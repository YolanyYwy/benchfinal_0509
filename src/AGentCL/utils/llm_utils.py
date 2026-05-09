import json
import re
from typing import Any, Optional, Union

import requests as _requests
import torch
import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from AGentCL.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
    USER_LLM_API_BASE,
    USER_LLM_API_KEY,
    USER_LLM_MODEL,
)
from AGentCL.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from AGentCL.environment.tool import Tool

# litellm._turn_on_debug()

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def _is_zidongtaichu_model(model: str) -> bool:
    """Check if the model should be routed through zidongtaichu streaming API."""
    clean = model.replace("openai/", "")
    return clean == USER_LLM_MODEL and "zidongtaichu" in USER_LLM_API_BASE


def _call_zidongtaichu_streaming(messages: list[dict], tools: Optional[list[dict]] = None, **kwargs) -> AssistantMessage:
    """Call zidongtaichu API with SSE streaming and collect the full response.

    Since gpt_oss_120b does not support native function calling, tools are
    injected into the system prompt as text and the model is instructed to
    use <tool_call> tags.  The response is then parsed for tool calls.
    """
    import os

    # 支持多 API keys：根据 LOCAL_RANK 选择对应的 key
    api_key = USER_LLM_API_KEY
    user_api_keys_str = os.environ.get("USER_API_KEYS", "")
    if user_api_keys_str:
        user_api_keys = user_api_keys_str.split()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if user_api_keys:
            api_key = user_api_keys[local_rank % len(user_api_keys)]

    url = f"{USER_LLM_API_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Inject tools description into messages (same approach as local models)
    api_messages = []
    tools_desc = _build_tools_prompt(tools)

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")

        if role == "system":
            # Append tools description to system message
            api_messages.append({"role": "system", "content": content + tools_desc})
        elif role == "assistant":
            if tool_calls:
                # Format previous tool calls as <tool_call> text
                tc = tool_calls[0]
                func = tc.get("function", {})
                tc_content = f'<tool_call>\n{{"name": "{func.get("name", "")}", "arguments": {func.get("arguments", "{}")}}}\n</tool_call>'
                api_messages.append({"role": "assistant", "content": tc_content})
            else:
                api_messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            # Format tool response as user message
            tool_call_id = msg.get("tool_call_id", "unknown")
            api_messages.append({"role": "user", "content": f"[Tool Result for {tool_call_id}]: {content}"})
        else:
            api_messages.append({"role": role, "content": content})

    # If no system message had tools injected and tools exist, prepend one
    if tools_desc and not any(m.get("role") == "system" for m in api_messages):
        api_messages.insert(0, {"role": "system", "content": tools_desc})

    payload = {
        "model": USER_LLM_MODEL,
        "messages": api_messages,
        "stream": True,
    }

    temperature = kwargs.get("temperature")
    if temperature is not None:
        payload["temperature"] = temperature

    resp = _requests.post(url, json=payload, headers=headers, stream=True, timeout=120)
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

            # Collect content (skip reasoning_content)
            if "content" in delta and delta["content"]:
                content_parts.append(delta["content"])
        except (json.JSONDecodeError, IndexError, KeyError):
            continue

    generated_text = "".join(content_parts)

    # Strip thinking content
    generated_text = _strip_thinking_content(generated_text)

    # Parse tool calls from <tool_call> tags in the text
    parsed_tool_calls = _parse_tool_calls(generated_text) if generated_text else None

    # If tool calls found, content should be None (same as local model behavior)
    content = None if parsed_tool_calls else (generated_text or None)

    return AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=parsed_tool_calls,
        cost=0.0,
        usage=None,
    )


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use. Use "local:<name>" for local models registered
               via register_local_model().
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    # Check if using local model
    if model.startswith("local:"):
        local_model_name = model[6:]  # Remove "local:" prefix
        return generate_local(
            model_name=local_model_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    # Check if using zidongtaichu streaming API
    if _is_zidongtaichu_model(model):
        litellm_messages = to_litellm_messages(messages)
        tools_schema = [tool.openai_schema for tool in tools] if tools else None
        return _call_zidongtaichu_streaming(
            messages=litellm_messages,
            tools=tools_schema,
            tool_choice=tool_choice,
            **kwargs,
        )

    # Original API-based generation
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    litellm_messages = to_litellm_messages(messages)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"

    try:
        response = completion(
            model=model,
            messages=litellm_messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
    except Exception as e:
        logger.error(e)
        raise e
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]
    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content
    tool_calls = response.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in tool_calls
    ]
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage


# ============================================================================
# Local Model Support
# ============================================================================

# Global registry for local models
_LOCAL_MODEL_REGISTRY: dict[str, tuple[Any, Any]] = {}


def register_local_model(name: str, model: Any, tokenizer: Any) -> None:
    """
    Register a local model for use with generate().

    Args:
        name: The name to register the model under (e.g., "qwen3-4b")
        model: The HuggingFace model
        tokenizer: The HuggingFace tokenizer

    Usage:
        from AGentCL.utils.llm_utils import register_local_model
        register_local_model("qwen3-4b", model, tokenizer)

        # Then use in LLMAgent:
        agent = LLMAgent(tools=tools, domain_policy=policy, llm="local:qwen3-4b")
    """
    _LOCAL_MODEL_REGISTRY[name] = (model, tokenizer)
    logger.info(f"Registered local model: {name}")


def unregister_local_model(name: str) -> None:
    """Unregister a local model."""
    if name in _LOCAL_MODEL_REGISTRY:
        del _LOCAL_MODEL_REGISTRY[name]
        logger.info(f"Unregistered local model: {name}")


def get_local_model(name: str) -> tuple[Any, Any]:
    """Get a registered local model and tokenizer."""
    if name not in _LOCAL_MODEL_REGISTRY:
        raise ValueError(f"Local model '{name}' not registered. Use register_local_model() first.")
    return _LOCAL_MODEL_REGISTRY[name]


def _build_tools_prompt(tools: Optional[list[dict]]) -> str:
    """Build a text description of tools for the prompt."""
    if not tools:
        return ""

    tools_desc = "\n<available_tools>\n"
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        tools_desc += f"\n## {name}\n"
        tools_desc += f"Description: {desc}\n"

        if params.get("properties"):
            tools_desc += "Parameters:\n"
            for param_name, param_info in params["properties"].items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                required = param_name in params.get("required", [])
                req_str = " (required)" if required else " (optional)"
                tools_desc += f"  - {param_name}: {param_type}{req_str} - {param_desc}\n"

    tools_desc += "\n</available_tools>\n"
    tools_desc += """
To call a tool, respond with ONLY a JSON object in this exact format:
<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
</tool_call>

To send a message to the user (without calling a tool), just respond with plain text.
You cannot do both at the same time - either call a tool OR send a message.
"""
    return tools_desc


def _messages_to_local_prompt(
    messages: list[dict],
    tools: Optional[list[dict]],
    tokenizer: Any,
) -> str:
    """Convert messages to a prompt string for local model generation."""
    tools_desc = _build_tools_prompt(tools)

    # Build chat messages with tool info in system prompt
    chat_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")

        if role == "system":
            # Append tools description to system message
            chat_messages.append({"role": "system", "content": content + tools_desc})
        elif role == "assistant":
            if tool_calls:
                # Format previous tool calls
                tc = tool_calls[0]
                func = tc.get("function", {})
                tc_content = f'<tool_call>\n{{"name": "{func.get("name", "")}", "arguments": {func.get("arguments", "{}")}}}\n</tool_call>'
                chat_messages.append({"role": "assistant", "content": tc_content})
            else:
                chat_messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            # Format tool response as user message
            tool_call_id = msg.get("tool_call_id", "unknown")
            chat_messages.append({"role": "user", "content": f"[Tool Result for {tool_call_id}]: {content}"})
        else:
            chat_messages.append({"role": "user", "content": content})

    # Use tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # For Qwen3 models
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
    else:
        # Fallback to simple format
        parts = []
        for msg in chat_messages:
            role = msg["role"].capitalize()
            parts.append(f"{role}: {msg['content']}")
        parts.append("Assistant:")
        return "\n\n".join(parts)


def _strip_thinking_content(text: str) -> str:
    """Remove <think>...</think> content from generated text."""
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)

    if '<think>' in cleaned:
        think_start = cleaned.find('<think>')
        if '</think>' not in cleaned[think_start:]:
            cleaned = cleaned[:think_start]

    cleaned = re.sub(r'</think>\s*', '', cleaned)
    cleaned = re.sub(r'<think>\s*', '', cleaned)

    return cleaned.strip()


def _parse_tool_calls(text: str) -> Optional[list[ToolCall]]:
    """Parse tool calls from generated text."""
    # Look for <tool_call>...</tool_call> format
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(tool_call_pattern, text, flags=re.DOTALL)

    if matches:
        try:
            tool_data = json.loads(matches[0])
            if "name" in tool_data and "arguments" in tool_data:
                return [ToolCall(
                    id=f"call_{hash(matches[0]) % 100000}",
                    name=tool_data["name"],
                    arguments=tool_data["arguments"],
                )]
        except (json.JSONDecodeError, Exception):
            pass

    # Try legacy JSON format: {"action": "tool_name", "parameters": {...}}
    try:
        if text.strip().startswith('{') and text.strip().endswith('}'):
            tool_data = json.loads(text.strip())
            if "action" in tool_data:
                return [ToolCall(
                    id=f"call_{hash(text) % 100000}",
                    name=tool_data.get("action"),
                    arguments=tool_data.get("parameters", {}),
                )]
    except (json.JSONDecodeError, Exception):
        pass

    return None


def generate_local(
    model_name: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> AssistantMessage:
    """
    Generate a response using a local model.

    Args:
        model_name: The registered local model name (without "local:" prefix)
        messages: The messages to send to the model
        tools: The tools available to the model
        tool_choice: The tool choice (ignored for local models, handled via prompt)
        **kwargs: Additional arguments (temperature, max_new_tokens, etc.)

    Returns:
        AssistantMessage with the generated response
    """
    import time
    import os
    rank = int(os.environ.get("LOCAL_RANK", 0))

    model, tokenizer = get_local_model(model_name)

    # Unwrap DeepSpeed/DDP wrapper for generation
    unwrapped_model = model
    while hasattr(unwrapped_model, "module"):
        unwrapped_model = unwrapped_model.module

    # Convert messages and tools to prompt
    litellm_messages = to_litellm_messages(messages)
    tools_schema = [tool.openai_schema for tool in tools] if tools else None
    prompt = _messages_to_local_prompt(litellm_messages, tools_schema, tokenizer)

    # Generation parameters
    temperature = kwargs.get("temperature", 0.7)
    max_new_tokens = min(kwargs.get("max_new_tokens", 1024), 1024)  # Limit to 1024

    # Tokenize without truncation first to check length
    max_input_tokens = 28000  # Leave room for generation
    full_inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    full_len = full_inputs.input_ids.shape[1]

    if full_len > max_input_tokens:
        # Truncate from LEFT (keep the most recent messages)
        input_ids = full_inputs.input_ids[:, -max_input_tokens:].to(unwrapped_model.device)
        attention_mask = full_inputs.attention_mask[:, -max_input_tokens:].to(unwrapped_model.device) if full_inputs.attention_mask is not None else None
        input_len = max_input_tokens
    else:
        input_ids = full_inputs.input_ids.to(unwrapped_model.device)
        attention_mask = full_inputs.attention_mask.to(unwrapped_model.device) if full_inputs.attention_mask is not None else None
        input_len = full_len

    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "repetition_penalty": 1.1,  # 防止重复生成
        }

        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False

        # Build inputs dict for generate
        gen_inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            gen_inputs["attention_mask"] = attention_mask

        # Generate with local model (sequential path)
        outputs = unwrapped_model.generate(**gen_inputs, **generation_kwargs)

        generated_text = tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        )

    # Clean up thinking content
    cleaned_text = _strip_thinking_content(generated_text)

    # 检测异常输出（大量重复字符、换行符等）
    def is_degenerate_output(text: str) -> bool:
        if not text or len(text) < 5:
            return True
        # 检查是否有太多换行符
        if text.count('\n') > 50:
            return True
        # 检查是否有太多重复字符
        if len(set(text)) < 10 and len(text) > 50:
            return True
        # 检查是否主要是空白字符和引号
        non_whitespace = text.replace('\n', '').replace(' ', '').replace('"', '').replace("'", '')
        if len(non_whitespace) < len(text) * 0.3 and len(text) > 20:
            return True
        return False

    if is_degenerate_output(cleaned_text):
        # 输出异常，返回默认响应
        return AssistantMessage(
            role="assistant",
            content="I apologize, but I need more information to assist you. Could you please clarify your request?",
            tool_calls=None,
            cost=0.0,
            usage=None,
        )

    if not cleaned_text:
        return AssistantMessage(
            role="assistant",
            content="How can I help you today?",
            tool_calls=None,
            cost=0.0,
            usage=None,
        )

    # Parse tool calls
    tool_calls = _parse_tool_calls(cleaned_text)

    # If tool calls found, content should be None
    content = None if tool_calls else cleaned_text

    return AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=0.0,
        usage=None,
    )
