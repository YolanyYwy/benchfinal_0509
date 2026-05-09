"""Policy model wrapper for GRPO training with open-source LLMs."""

import copy
import json
import os
import re
from typing import Optional, Tuple

import requests
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from AGentCL.agent.base import LocalAgent
from AGentCL.agent.llm_agent import LLMAgent, LLMAgentState
from AGentCL.data_model.message import AssistantMessage, Message, SystemMessage, UserMessage
from AGentCL.data_model.tasks import Task
from AGentCL.environment.environment import Environment
from AGentCL.orchestrator.orchestrator import Orchestrator
from AGentCL.registry import registry
from AGentCL.user.user_simulator import UserSimulator
from AGentCL.user.base import ValidUserInputMessage, UserState

from .config import GRPOConfig
from .reward_oracle import Trajectory


def strip_thinking_content(text: str) -> str:
    """Remove all <think>...</think> content from generated text."""
    if not text:
        return ""

    cleaned = text.strip()

    # Remove complete think blocks
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)

    # Handle incomplete think tag
    if '<think>' in cleaned:
        think_start = cleaned.find('<think>')
        if '</think>' not in cleaned[think_start:]:
            cleaned = cleaned[:think_start]

    # Remove any remaining artifacts
    cleaned = re.sub(r'</think>\s*', '', cleaned)
    cleaned = re.sub(r'<think>\s*', '', cleaned)

    return cleaned.strip()


class APIUserSimulator(UserSimulator):
    """User simulator that uses zidongtaichu API (SSE streaming)."""

    def __init__(
        self,
        tools,
        instructions,
        api_base: str,
        api_key: str,
        model: str = "gpt_oss_120b",
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(tools=tools, instructions=instructions, llm=None, llm_args={})
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._seed = seed

    def set_seed(self, seed: int):
        """Set the seed for reproducibility."""
        self._seed = seed

    def _call_streaming_api(self, payload: dict) -> str:
        """Call zidongtaichu API and collect SSE streaming chunks into full text."""
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload["stream"] = True

        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=120)
        resp.raise_for_status()

        content_parts = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    content_parts.append(delta["content"])
            except (json.JSONDecodeError, IndexError, KeyError):
                continue

        return "".join(content_parts)

    def _generate_next_message(self, message: ValidUserInputMessage, state: UserState) -> Tuple[UserMessage, UserState]:
        from AGentCL.data_model.message import MultiToolMessage

        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Sliding window: keep only recent messages to prevent role confusion in long conversations
        MAX_HISTORY_MESSAGES = 20
        flipped = state.flip_roles()
        if len(flipped) > MAX_HISTORY_MESSAGES:
            flipped = flipped[-MAX_HISTORY_MESSAGES:]
        messages = state.system_messages + flipped
        api_messages = self._messages_to_api_format(messages)

        # Build API call payload
        payload = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
        }

        # Add seed for reproducibility
        if self._seed is not None:
            payload["seed"] = self._seed

        try:
            content = self._call_streaming_api(payload)

            # Strip thinking content
            content = strip_thinking_content(content)

            if not content:
                content = "I understand. Please continue."

        except Exception as e:
            print(f"  [API Error] zidongtaichu API call failed: {e}")
            content = "I'm having trouble understanding. Could you please help me?"

        user_message = UserMessage(
            role="user",
            content=content,
            cost=0.0,
            usage=None,
            raw_data=None,
        )

        state.messages.append(user_message)
        return user_message, state

    def _messages_to_api_format(self, messages: list[Message]) -> list[dict]:
        """Convert messages to API dict format (roles already flipped by flip_roles)."""
        api_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content or ""})
            elif isinstance(msg, AssistantMessage):
                api_messages.append({"role": "assistant", "content": msg.content or ""})
            elif isinstance(msg, UserMessage):
                api_messages.append({"role": "user", "content": msg.content or ""})
            else:
                api_messages.append({"role": "user", "content": msg.content or ""})
        return api_messages


class LocalUserSimulator(UserSimulator):
    """User simulator that uses a local model instead of API."""

    def __init__(
        self,
        tools,
        instructions,
        model,
        tokenizer,
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
    ):
        super().__init__(tools=tools, instructions=instructions, llm=None, llm_args={})
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def _generate_next_message(self, message: ValidUserInputMessage, state: UserState) -> Tuple[UserMessage, UserState]:
        from AGentCL.data_model.message import MultiToolMessage

        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.flip_roles()
        prompt = self._messages_to_prompt(messages)

        # Always unwrap for generation
        model = self.model
        while hasattr(model, "module"):
            model = model.module

        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            # Limit max_new_tokens for faster generation
            max_tokens = min(self.max_new_tokens, 256)

            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }

            if self.temperature > 0:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = self.temperature
            else:
                generation_kwargs["do_sample"] = False

            outputs = model.generate(**inputs, **generation_kwargs)

            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

        # Strip thinking content
        cleaned_text = strip_thinking_content(generated_text)

        # Default response if nothing left
        if not cleaned_text:
            cleaned_text = "Please continue."

        user_message = UserMessage(
            role="user",
            content=cleaned_text,
            cost=0.0,
            usage=None,
            raw_data=None,
        )

        state.messages.append(user_message)
        return user_message, state

    def _messages_to_prompt(self, messages: list[Message]) -> str:
        """Convert messages to prompt format.

        Note: flip_roles() has already been called, so we just need to
        convert the message types to their role strings without flipping again.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            chat_messages = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    chat_messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, AssistantMessage):
                    # After flip_roles(), AssistantMessage = what user sim said
                    chat_messages.append({"role": "assistant", "content": msg.content or ""})
                elif isinstance(msg, UserMessage):
                    # After flip_roles(), UserMessage = what agent said
                    chat_messages.append({"role": "user", "content": msg.content or ""})
                else:
                    chat_messages.append({"role": "user", "content": msg.content or ""})

            # Try to use enable_thinking=False for Qwen3 models
            try:
                return self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        else:
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    prompt_parts.append(f"System: {msg.content}")
                elif isinstance(msg, AssistantMessage):
                    prompt_parts.append(f"Assistant: {msg.content or ''}")
                elif isinstance(msg, UserMessage):
                    prompt_parts.append(f"User: {msg.content or ''}")
                else:
                    prompt_parts.append(f"User: {msg.content or ''}")
            prompt_parts.append("Assistant:")
            return "\n\n".join(prompt_parts)


class PolicyLLMAgent(LLMAgent):
    """Custom LLM agent that uses a local model for generation."""

    def __init__(
        self,
        tools,
        domain_policy: str,
        model,
        tokenizer,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ):
        LocalAgent.__init__(self, tools=tools, domain_policy=domain_policy)
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate_next_message(self, message, state: LLMAgentState) -> tuple[AssistantMessage, LLMAgentState]:
        from AGentCL.data_model.message import MultiToolMessage, ToolCall
        import json

        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.messages
        prompt = self._messages_to_prompt(messages)

        # Always unwrap for generation
        model = self.model
        while hasattr(model, "module"):
            model = model.module

        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            generation_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }

            if self.temperature > 0:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = self.temperature
            else:
                generation_kwargs["do_sample"] = False

            outputs = model.generate(**inputs, **generation_kwargs)

            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

        # Strip thinking content from agent response
        cleaned_text = strip_thinking_content(generated_text)

        # If nothing left after stripping, return default message
        if not cleaned_text:
            assistant_message = AssistantMessage(
                role="assistant",
                content="How can I help you today?",
                tool_calls=None,
            )
            state.messages.append(assistant_message)
            return assistant_message, state

        tool_calls = None
        final_content = cleaned_text

        # Look for <tool_call>...</tool_call> format
        tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        tool_call_matches = re.findall(tool_call_pattern, cleaned_text, flags=re.DOTALL)

        if tool_call_matches:
            try:
                tool_data = json.loads(tool_call_matches[0])
                if "name" in tool_data and "arguments" in tool_data:
                    tool_calls = [ToolCall(
                        id=f"call_{hash(tool_call_matches[0]) % 10000}",
                        name=tool_data["name"],
                        arguments=tool_data["arguments"],
                        requestor="assistant"
                    )]
                    final_content = None
            except (json.JSONDecodeError, Exception):
                pass

        # Try legacy JSON format: {"action": "tool_name", "parameters": {...}}
        if not tool_calls:
            try:
                if cleaned_text.strip().startswith('{') and cleaned_text.strip().endswith('}'):
                    tool_data = json.loads(cleaned_text.strip())
                    if "action" in tool_data:
                        tool_name = tool_data.get("action")
                        tool_args = tool_data.get("parameters", {})
                        tool_calls = [ToolCall(
                            id=f"call_{hash(cleaned_text) % 10000}",
                            name=tool_name,
                            arguments=tool_args,
                            requestor="assistant"
                        )]
                        final_content = None
            except (json.JSONDecodeError, Exception):
                pass

        assistant_message = AssistantMessage(
            role="assistant",
            content=final_content if (final_content and not tool_calls) else None,
            tool_calls=tool_calls,
        )
        state.messages.append(assistant_message)
        return assistant_message, state

    def _messages_to_prompt(self, messages: list[Message]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            chat_messages = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    chat_messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, AssistantMessage):
                    chat_messages.append({"role": "assistant", "content": msg.content or ""})
                else:
                    chat_messages.append({"role": "user", "content": msg.content or ""})

            # Try to use enable_thinking=False for Qwen3 models
            try:
                return self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        else:
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    prompt_parts.append(f"System: {msg.content}")
                elif isinstance(msg, AssistantMessage):
                    prompt_parts.append(f"Assistant: {msg.content or ''}")
                else:
                    prompt_parts.append(f"User: {msg.content or ''}")
            prompt_parts.append("Assistant:")
            return "\n\n".join(prompt_parts)


class PolicyModel:
    """Wrapper around open-source LLM for policy learning with gradient access."""

    def __init__(self, config: GRPOConfig, device: torch.device):
        self.config = config
        self.device = device

        # Check if using DeepSpeed
        use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true"
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if local_rank == 0:
            print(f"Loading model: {config.model_name_or_path}")
            print(f"Using DeepSpeed: {use_deepspeed}")

        # When using DeepSpeed, don't use device_map - let DeepSpeed handle it
        # Use low_cpu_mem_usage to reduce memory footprint during loading
        if use_deepspeed:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                torch_dtype=getattr(torch, config.model_dtype),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                torch_dtype=getattr(torch, config.model_dtype),
                device_map={"": device},
                trust_remote_code=True,
            )

        self.model.train()
        if local_rank == 0:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Agent model: {trainable_params:,} / {total_params:,} parameters are trainable")

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Note: gradient_checkpointing is incompatible with DeepSpeed ZeRO
        # when compute_grpo_loss does multiple forward passes per backward.
        # We disable it here and let DeepSpeed handle memory optimization.
        # If you need gradient checkpointing, set world_size=1 or use FSDP.
        self._gradient_checkpointing_enabled = False
        if config.gradient_checkpointing and config.world_size == 1:
            self.model.gradient_checkpointing_enable()
            self._gradient_checkpointing_enabled = True
            if local_rank == 0:
                print("[Warning] Gradient checkpointing enabled (single GPU mode)")
        elif config.gradient_checkpointing:
            if local_rank == 0:
                print("[Warning] Gradient checkpointing disabled for DeepSpeed compatibility")

        # Reference model - only load on rank 0 and share, or skip for memory saving
        # For now, we load it but with low_cpu_mem_usage
        if local_rank == 0:
            print("Creating reference model for KL divergence...")

        if use_deepspeed:
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                torch_dtype=getattr(torch, config.model_dtype),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            # Move reference model to the correct device
            self.reference_model = self.reference_model.to(device)
        else:
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                torch_dtype=getattr(torch, config.model_dtype),
                device_map={"": device},
                trust_remote_code=True,
            )
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        if config.warmup_steps > 0:
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=config.warmup_steps)
        else:
            self.scheduler = None

        if config.use_local_user_model:
            if local_rank == 0:
                print(f"Loading user simulator model: {config.user_model}")

            # User model - also don't use device_map with DeepSpeed
            if use_deepspeed:
                self.user_model = AutoModelForCausalLM.from_pretrained(
                    config.user_model,
                    torch_dtype=getattr(torch, config.model_dtype),
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                self.user_model = self.user_model.to(device)
            else:
                self.user_model = AutoModelForCausalLM.from_pretrained(
                    config.user_model,
                    torch_dtype=getattr(torch, config.model_dtype),
                    device_map={"": device},
                    trust_remote_code=True,
                )
            self.user_model.eval()
            for p in self.user_model.parameters():
                p.requires_grad = False

            self.user_tokenizer = AutoTokenizer.from_pretrained(config.user_model, trust_remote_code=True)
            if self.user_tokenizer.pad_token is None:
                self.user_tokenizer.pad_token = self.user_tokenizer.eos_token

            if getattr(config, "local_rank", 0) == 0:
                print(f"User simulator model loaded on {device}")
        else:
            self.user_model = None
            self.user_tokenizer = None
            if getattr(config, "local_rank", 0) == 0:
                print(f"Using API for user simulator: {config.user_model}")

        if getattr(config, "local_rank", 0) == 0:
            print(f"Model loaded on {device}")

    def generate_responses(self, task: Task, environment: Environment, num_samples: int, domain: str, timeout_per_sample: float = 300.0) -> list[Trajectory]:
        import signal
        import threading

        trajectories = []

        # generation uses eval + no_grad
        self.model.eval()

        if self.config.verbose and getattr(self.config, "local_rank", 0) == 0:
            print(f"  → Generating {num_samples} trajectories...")

        for sample_idx in range(num_samples):
            agent = self._create_agent(environment, domain)
            user = self._create_user(task, domain)

            orchestrator = Orchestrator(
                domain=domain,
                agent=agent,
                user=user,
                environment=environment,
                task=task,
                max_steps=50,
            )

            try:
                orchestrator.initialize()
                simulation_run = orchestrator.run()

                trajectory = Trajectory(
                    task_id=task.id,
                    messages=simulation_run.messages,
                    termination_reason=simulation_run.termination_reason,
                    cost=0.0,
                )
                trajectories.append(trajectory)
            except Exception as e:
                if self.config.verbose and getattr(self.config, "local_rank", 0) == 0:
                    print(f"  ✗ Sample {sample_idx} failed: {e}")
                continue

        return trajectories

    def _create_agent(self, environment: Environment, domain: str) -> PolicyLLMAgent:
        tools_obj = environment.get_tools()
        tools = list(tools_obj.values()) if isinstance(tools_obj, dict) else list(tools_obj)
        domain_policy = environment.policy

        return PolicyLLMAgent(
            tools=tools,
            domain_policy=domain_policy,
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
        )

    def _create_user(self, task: Task, domain: str):
        user_tools = []
        if domain == "telecom":
            from AGentCL.domains.telecom.user_tools import TelecomUserTools
            from AGentCL.domains.telecom.user_data_model import TelecomUserDB
            user_db = TelecomUserDB()
            user_toolkit = TelecomUserTools(user_db)
            user_tools_obj = user_toolkit.get_tools()
            user_tools = list(user_tools_obj.values()) if isinstance(user_tools_obj, dict) else list(user_tools_obj)

        if self.config.use_local_user_model:
            return LocalUserSimulator(
                instructions=task.user_scenario,
                tools=user_tools,
                model=self.user_model,
                tokenizer=self.user_tokenizer,
                temperature=self.config.user_model_temperature,
                max_new_tokens=self.config.max_new_tokens,
            )
        else:
            # 使用 zidongtaichu API (SSE streaming)
            return APIUserSimulator(
                instructions=task.user_scenario,
                tools=user_tools,
                api_base=self.config.user_api_base,
                api_key=self.config.user_api_key,
                model=self.config.user_model,
                temperature=self.config.user_model_temperature,
                seed=getattr(self.config, 'seed', None),
            )

    def compute_log_probs(
        self,
        trajectory: Trajectory,
        use_reference: bool = False,
        max_messages: int = 5,  # 最多处理 5 条消息
        max_seq_len: int = 2048,  # 限制序列长度
        grad_messages: int = 2,  # 只对最后 2 条消息保留梯度
        debug: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute sum of log probabilities for assistant messages in trajectory.

        为了防止 OOM，只对最后 grad_messages 个 message 保留梯度，
        其他 message 的 log_prob 会 detach 掉。

        IMPORTANT: 为避免死锁，每条消息都会执行一次 forward（成功的做真实 forward，
        失败的做 dummy forward），确保 forward 次数 = min(len(assistant_messages), max_messages)。
        dummy forward 的结果会累加到 total_log_prob 中（乘以 0），保持计算图完整。

        Returns:
            Tuple of (log_prob tensor, num_forwards executed)
        """
        model = self.reference_model if use_reference else self.model

        # ✅ FIX: only unwrap reference model; never unwrap the trainable policy model.
        if use_reference and hasattr(model, "module"):
            model = model.module

        assistant_messages = [m for m in trajectory.messages if isinstance(m, AssistantMessage) and m.content]
        if not assistant_messages:
            if debug:
                print(f"      [DEBUG] No assistant messages with content found")
            return torch.tensor([], device=self.device), 0

        # 限制最多处理的消息数量
        if len(assistant_messages) > max_messages:
            assistant_messages = assistant_messages[-max_messages:]

        num_messages = len(assistant_messages)
        # 确定哪些 message 需要梯度（最后 grad_messages 个）
        grad_start_idx = max(0, num_messages - grad_messages) if not use_reference else num_messages

        # Accumulate log probs - 使用 None 初始化，后面会根据第一个有效值设置
        total_log_prob = None
        num_valid_messages = 0
        num_forwards = 0  # 实际执行的 forward 次数

        for msg_idx, msg in enumerate(assistant_messages):
            # 判断这个 message 是否需要梯度
            need_grad = (not use_reference) and (msg_idx >= grad_start_idx)
            did_forward = False  # 标记是否已做 forward

            try:
                history = []
                for m in trajectory.messages:
                    if m == msg:
                        break
                    history.append(m)

                prompt = self._messages_to_prompt(history)
                target_text = msg.content or ""
                if not target_text:
                    # 做 dummy forward 保持同步，并累加到 total_log_prob
                    if not use_reference:
                        dummy_val = self._dummy_forward()
                        total_log_prob = dummy_val if total_log_prob is None else total_log_prob + dummy_val
                        num_forwards += 1
                    continue

                # 截断 target_text 如果太长
                target_text_truncated = target_text[:1024] if len(target_text) > 1024 else target_text

                prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).to(self.device)
                full_text = prompt + target_text_truncated
                full_ids = self.tokenizer.encode(full_text, return_tensors="pt", truncation=True, max_length=max_seq_len).to(self.device)

                if prompt_ids.shape[1] >= full_ids.shape[1]:
                    # 做 dummy forward 保持同步
                    if not use_reference:
                        dummy_val = self._dummy_forward()
                        total_log_prob = dummy_val if total_log_prob is None else total_log_prob + dummy_val
                        num_forwards += 1
                    continue

                target_ids = full_ids[:, prompt_ids.shape[1]:]
                if target_ids.shape[1] == 0:
                    # 做 dummy forward 保持同步
                    if not use_reference:
                        dummy_val = self._dummy_forward()
                        total_log_prob = dummy_val if total_log_prob is None else total_log_prob + dummy_val
                        num_forwards += 1
                    continue

                # 只对需要梯度的 message 启用梯度
                with torch.set_grad_enabled(need_grad):
                    outputs = model(full_ids)
                    did_forward = True
                    num_forwards += 1
                    logits = outputs.logits

                    start_idx = prompt_ids.shape[1] - 1
                    end_idx = full_ids.shape[1] - 1
                    if start_idx < 0 or end_idx <= start_idx:
                        # 已做 forward，但无法计算 log_prob，累加 0
                        zero_val = logits.sum() * 0.0
                        total_log_prob = zero_val if total_log_prob is None else total_log_prob + zero_val
                        continue

                    target_logits = logits[:, start_idx:end_idx, :]
                    if target_logits.shape[1] != target_ids.shape[1]:
                        # 已做 forward，但无法计算 log_prob，累加 0
                        zero_val = logits.sum() * 0.0
                        total_log_prob = zero_val if total_log_prob is None else total_log_prob + zero_val
                        continue

                    log_probs_tokens = F.log_softmax(target_logits, dim=-1)
                    token_log_probs = log_probs_tokens.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
                    msg_log_prob = token_log_probs.sum()

                # 如果不需要梯度，detach 掉
                if not need_grad:
                    msg_log_prob = msg_log_prob.detach()

                total_log_prob = msg_log_prob if total_log_prob is None else total_log_prob + msg_log_prob
                num_valid_messages += 1

                # 清理中间变量
                del outputs, logits, target_logits, log_probs_tokens, token_log_probs
                del prompt_ids, full_ids, target_ids

            except Exception as e:
                # 如果还没做 forward，做 dummy forward 保持同步
                if not did_forward and not use_reference:
                    dummy_val = self._dummy_forward()
                    total_log_prob = dummy_val if total_log_prob is None else total_log_prob + dummy_val
                    num_forwards += 1
                continue

        # 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if num_valid_messages == 0:
            # 返回 total_log_prob（包含 dummy forward 的计算图）或空 tensor
            if total_log_prob is not None:
                return total_log_prob.unsqueeze(0), num_forwards
            return torch.tensor([], device=self.device), num_forwards

        # Return as 1-element tensor for compatibility
        return total_log_prob.unsqueeze(0), num_forwards

    def _messages_to_prompt(self, messages: list[Message]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            chat_messages = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    chat_messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, AssistantMessage):
                    # 移除 thinking 内容，减少 prompt 长度
                    content = strip_thinking_content(msg.content or "")
                    chat_messages.append({"role": "assistant", "content": content})
                else:
                    chat_messages.append({"role": "user", "content": msg.content or ""})
            return self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=False)
        else:
            parts = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    parts.append(f"System: {msg.content}")
                elif isinstance(msg, AssistantMessage):
                    # 移除 thinking 内容，减少 prompt 长度
                    content = strip_thinking_content(msg.content or "")
                    parts.append(f"Assistant: {content}")
                else:
                    parts.append(f"User: {msg.content or ''}")
            return "\n\n".join(parts)

    def compute_grpo_loss(
        self,
        trajectories: list[Trajectory],
        advantages: torch.Tensor,
        num_forwards: int = None,  # 同步后的 forward 次数，确保所有 rank 一致
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute GRPO loss for a batch of trajectories.

        IMPORTANT: 为避免 DeepSpeed ZeRO 死锁，所有 rank 必须执行相同次数的 forward。
        通过 num_forwards 参数控制，不足的用 dummy forward 补齐。
        当 log_prob 有效但 ref_log_prob 为空时，使用 ref_log_prob_val=0 继续计算，
        确保 log_prob 的计算图不被丢弃。

        Args:
            trajectories: 轨迹列表
            advantages: 优势值
            num_forwards: 所有 rank 同步后的 forward 次数。如果为 None，则不做同步（单 GPU 模式）

        Returns:
            loss: The computed loss tensor
            num_valid: Number of valid trajectories that contributed to the loss
        """
        total_loss = None  # 使用 None 初始化，确保计算图正确
        valid = 0
        rank = getattr(self.config, "local_rank", 0)
        actual_forwards = 0  # 实际执行的 forward 次数

        for traj_idx, (traj, adv) in enumerate(zip(trajectories, advantages)):
            try:
                assistant_messages = [m for m in traj.messages if isinstance(m, AssistantMessage) and m.content]

                # 使用 compute_log_probs 计算 log probs（会执行多次 forward）
                # 现在返回 (log_prob, num_forwards)
                log_prob, fwd_count = self.compute_log_probs(traj, use_reference=False, debug=True)
                actual_forwards += fwd_count

                with torch.no_grad():
                    ref_log_prob, _ = self.compute_log_probs(traj, use_reference=True, debug=False)

                # 如果 log_prob 为空，跳过（但计算图已在 compute_log_probs 中通过 dummy forward 保留）
                if log_prob.numel() == 0:
                    print(f"    [Rank {rank}] Traj {traj_idx}: INVALID - log_probs empty "
                          f"(assistant_msgs={len(assistant_messages)}, total_msgs={len(traj.messages)})")
                    continue

                log_prob_val = log_prob.squeeze()

                # 如果 ref_log_prob 为空，使用 0 作为 baseline（不跳过，保留 log_prob 的计算图）
                if ref_log_prob.numel() == 0:
                    print(f"    [Rank {rank}] Traj {traj_idx}: WARNING - ref_log_probs empty, using 0 as baseline")
                    ref_log_prob_val = torch.tensor(0.0, device=self.device)
                else:
                    ref_log_prob_val = ref_log_prob.squeeze()

                policy_loss = -log_prob_val * adv
                kl_div = log_prob_val - ref_log_prob_val
                traj_loss = policy_loss + self.config.kl_coef * kl_div

                if torch.isnan(traj_loss) or torch.isinf(traj_loss):
                    print(f"    [Rank {rank}] Traj {traj_idx}: INVALID - loss is nan/inf")
                    # 仍然累加一个 0 来保留计算图
                    zero_loss = log_prob_val * 0.0
                    total_loss = zero_loss if total_loss is None else total_loss + zero_loss
                    continue

                total_loss = traj_loss if total_loss is None else total_loss + traj_loss
                valid += 1
                print(f"    [Rank {rank}] Traj {traj_idx}: VALID - loss={traj_loss.item():.4f}, "
                      f"assistant_msgs={len(assistant_messages)}, log_prob={log_prob_val.item():.2f}")

            except Exception as e:
                print(f"    [Rank {rank}] Traj {traj_idx}: INVALID - exception: {e}")
                continue

        # 如果指定了 num_forwards，补齐 dummy forward 以保持同步
        if num_forwards is not None and actual_forwards < num_forwards:
            dummy_count = num_forwards - actual_forwards
            print(f"    [Rank {rank}] Padding {dummy_count} dummy forwards (actual={actual_forwards}, required={num_forwards})")
            for _ in range(dummy_count):
                dummy_val = self._dummy_forward()
                total_loss = dummy_val if total_loss is None else total_loss + dummy_val

        # 确保 total_loss 有计算图
        if total_loss is None:
            # 做一个 dummy forward 来创建计算图
            print(f"    [Rank {rank}] No valid loss, creating dummy loss")
            total_loss = self._dummy_forward()
        elif valid > 0:
            total_loss = total_loss / valid

        return total_loss, valid

    def _dummy_forward(self) -> torch.Tensor:
        """做一个 dummy forward，保持所有 rank 同步"""
        dummy_input = torch.tensor([[self.tokenizer.pad_token_id or 0]], device=self.device)
        dummy_output = self.model(dummy_input)
        return dummy_output.logits.sum() * 0.0

    def count_forwards_needed(self, trajectories: list[Trajectory], max_messages: int = 5) -> int:
        """计算处理这些 trajectories 需要的 forward 次数"""
        total = 0
        for traj in trajectories:
            assistant_messages = [m for m in traj.messages if isinstance(m, AssistantMessage) and m.content]
            # compute_log_probs 最多处理 max_messages 条消息
            total += min(len(assistant_messages), max_messages) if assistant_messages else 0
        return total

    def save_checkpoint(self, path: str, model_override=None, save_optimizer: bool = True):
        """
        保存 checkpoint。

        注意：DeepSpeed ZeRO 的 optimizer state 是按 rank 分片的，
        每个 rank 需要保存自己的 optimizer state 到不同文件。
        """
        os.makedirs(path, exist_ok=True)
        rank = getattr(self.config, "local_rank", 0)

        # 模型只需要 rank 0 保存（所有 rank 的模型权重是一样的）
        if rank == 0:
            if model_override is not None:
                model_to_save = model_override
            else:
                model_to_save = self.model.module if hasattr(self.model, "module") else self.model

            model_to_save.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

        # Optimizer state 每个 rank 都要保存（因为 ZeRO 分片）
        if save_optimizer:
            torch.save(
                {
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                },
                os.path.join(path, f"optimizer_rank{rank}.pt"),
            )

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """
        加载 checkpoint。

        注意：DeepSpeed ZeRO 的 optimizer state 是按 rank 分片的，
        每个 rank 需要加载自己的 optimizer state。
        """
        import glob

        local_rank = getattr(self.config, "local_rank", 0)

        # 获取目标模型（unwrap DeepSpeed wrapper）
        target = self.model.module if hasattr(self.model, "module") else self.model

        # 查找权重文件
        safetensor_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        bin_files = sorted(glob.glob(os.path.join(path, "model*.bin"))) or sorted(glob.glob(os.path.join(path, "pytorch_model*.bin")))

        if local_rank == 0:
            print(f"[load_checkpoint] path={path}")
            print(f"[load_checkpoint] safetensor_files={safetensor_files}")
            print(f"[load_checkpoint] bin_files={bin_files}")

        if safetensor_files:
            # 加载 safetensors 格式
            if local_rank == 0:
                print(f"[load_checkpoint] Loading {len(safetensor_files)} safetensor files...")
            from safetensors.torch import load_file
            state_dict = {}
            for sf in safetensor_files:
                state_dict.update(load_file(sf, device="cpu"))
            # 使用 strict=False 处理 tied weights（如 lm_head.weight）
            missing, unexpected = target.load_state_dict(state_dict, strict=False)
            if local_rank == 0 and missing:
                # 过滤掉 tied weights 的警告
                real_missing = [k for k in missing if not k.endswith('.weight') or 'lm_head' not in k]
                if real_missing:
                    print(f"[load_checkpoint] Warning: Missing keys (non-tied): {real_missing}")
            del state_dict
        elif bin_files:
            # 加载 bin 格式
            if local_rank == 0:
                print(f"[load_checkpoint] Loading {len(bin_files)} bin files...")
            state_dict = {}
            for bf in bin_files:
                state_dict.update(torch.load(bf, map_location="cpu", weights_only=True))
            # 使用 strict=False 处理 tied weights（如 lm_head.weight）
            missing, unexpected = target.load_state_dict(state_dict, strict=False)
            if local_rank == 0 and missing:
                # 过滤掉 tied weights 的警告
                real_missing = [k for k in missing if not k.endswith('.weight') or 'lm_head' not in k]
                if real_missing:
                    print(f"[load_checkpoint] Warning: Missing keys (non-tied): {real_missing}")
            del state_dict
        else:
            # 没有找到权重文件，报错而不是回退到 from_pretrained
            raise FileNotFoundError(
                f"No safetensors or bin files found in {path}. "
                f"Please check the checkpoint directory structure."
            )

        torch.cuda.empty_cache()

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if local_rank == 0:
            print(f"[load_checkpoint] Checkpoint loaded successfully from {path}")

        if not load_optimizer:
            return

        rank = getattr(self.config, "local_rank", 0)

        # 优先加载按 rank 分片的 optimizer state
        optimizer_path_rank = os.path.join(path, f"optimizer_rank{rank}.pt")
        optimizer_path_legacy = os.path.join(path, "optimizer.pt")

        # 调试信息
        print(f"[load_checkpoint] rank={rank}, optimizer_path_rank={optimizer_path_rank}, exists={os.path.exists(optimizer_path_rank)}")

        if os.path.exists(optimizer_path_rank):
            # 新格式：每个 rank 有自己的 optimizer state
            print(f"[load_checkpoint] rank={rank} loading optimizer from {optimizer_path_rank}")
            ckpt = torch.load(optimizer_path_rank, map_location="cpu", weights_only=False)

            # DeepSpeed ZeRO optimizer 需要特殊处理
            # 检查是否是 DeepSpeed wrapped optimizer
            if hasattr(self.optimizer, 'optimizer') and hasattr(self.optimizer.optimizer, 'param_groups'):
                # 这是 Accelerate wrapped DeepSpeed optimizer
                # 直接加载到底层 optimizer 的 state
                try:
                    # 尝试直接设置 optimizer state（绕过 DeepSpeed 的 load_state_dict）
                    base_optimizer = self.optimizer.optimizer
                    if hasattr(base_optimizer, 'optimizer'):
                        # DeepSpeed ZeRO optimizer
                        base_optimizer = base_optimizer.optimizer

                    saved_state = ckpt["optimizer_state_dict"]
                    # 如果保存的是完整的 DeepSpeed state，提取当前 rank 的部分
                    if isinstance(saved_state, dict) and "optimizer_state_dict" in saved_state:
                        saved_state = saved_state["optimizer_state_dict"]

                    base_optimizer.load_state_dict(saved_state)
                    print(f"[load_checkpoint] rank={rank} optimizer loaded successfully (direct)")
                except Exception as e:
                    print(f"[Warning] rank={rank} Could not load optimizer state: {e}")
                    print(f"[Warning] Model weights loaded, optimizer will start fresh.")
            else:
                # 普通 optimizer
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    print(f"[load_checkpoint] rank={rank} optimizer loaded successfully")
                except Exception as e:
                    print(f"[Warning] rank={rank} Could not load optimizer state: {e}")
                    print(f"[Warning] Model weights loaded, optimizer will start fresh.")

            if self.scheduler and ckpt.get("scheduler_state_dict") is not None:
                try:
                    self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception as e:
                    print(f"[Warning] rank={rank} Could not load scheduler state: {e}")
        elif os.path.exists(optimizer_path_legacy):
            # 旧格式：尝试加载，如果失败则跳过
            try:
                ckpt = torch.load(optimizer_path_legacy, map_location=self.device, weights_only=False)
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if self.scheduler and ckpt.get("scheduler_state_dict") is not None:
                    self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except (KeyError, RuntimeError) as e:
                if rank == 0:
                    print(f"[Warning] Could not load legacy optimizer state: {e}")
                    print(f"[Warning] Model weights loaded, optimizer will start fresh.")

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
