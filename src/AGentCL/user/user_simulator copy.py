import re
from typing import Optional, Tuple

from loguru import logger
from openai import OpenAI

from AGentCL.config import USER_LLM_API_BASE, USER_LLM_API_KEY, USER_LLM_MODEL
from AGentCL.data_model.message import (
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from AGentCL.data_model.tasks import UserInstructions
from AGentCL.environment.tool import Tool
from AGentCL.user.base import (
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    BaseUser,
    UserState,
    ValidUserInputMessage,
    is_valid_user_history_message,
)
from AGentCL.utils import DATA_DIR
from AGentCL.utils.llm_utils import generate

GLOBAL_USER_SIM_GUIDELINES_DIR = DATA_DIR / "tau2" / "user_simulator"


GLOBAL_USER_SIM_GUIDELINES_PATH = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines.md"
)

GLOBAL_USER_SIM_GUIDELINES_PATH_TOOLS = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines_tools.md"
)


def get_global_user_sim_guidelines(use_tools: bool = False) -> str:
    """
    Get the global user simulator guidelines.

    Args:
        use_tools: Whether to use the tools guidelines.

    Returns:
        The global user simulator guidelines.
    """
    if use_tools:
        with open(GLOBAL_USER_SIM_GUIDELINES_PATH_TOOLS, "r") as fp:
            user_sim_guidelines = fp.read()
    else:
        with open(GLOBAL_USER_SIM_GUIDELINES_PATH, "r") as fp:
            user_sim_guidelines = fp.read()
    return user_sim_guidelines


SYSTEM_PROMPT = """
{global_user_sim_guidelines}

<scenario>
{instructions}
</scenario>
""".strip()


class UserSimulator(BaseUser):
    """Stateless implementation of a user simulator."""

    def __init__(
        self,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[UserInstructions] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(instructions=instructions, llm=llm, llm_args=llm_args)
        self.tools = tools

    @property
    def global_simulation_guidelines(self) -> str:
        """
        The simulation guidelines for the user simulator.
        """
        use_tools = self.tools is not None
        return get_global_user_sim_guidelines(use_tools=use_tools)

    @property
    def system_prompt(self) -> str:
        """
        The system prompt for the user simulator.
        """
        if self.instructions is None:
            logger.warning("No instructions provided for user simulator")

        system_prompt = SYSTEM_PROMPT.format(
            global_user_sim_guidelines=self.global_simulation_guidelines,
            instructions=self.instructions,
        )
        return system_prompt

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        """
        Get the initial state of the user simulator.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_user_history_message(m) for m in message_history), (
            "Invalid user message history. User messages must be of type UserMessage, AssistantMessage, or ToolMessage to User."
        )

        user_state = UserState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
        return user_state

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        """
        Check if the message is a stop message.
        """
        if message.is_tool_call():
            return False
        assert message.content is not None
        return (
            STOP in message.content
            or TRANSFER in message.content
            or OUT_OF_SCOPE in message.content
        )

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        return self._generate_next_message(message, state)

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """Get the response from the user simulator.

        Args:
            message: The assistant or tool message.
            state: The user simulator's state.

        Returns:
            A tuple containing the user message and the updated user state.
        """
        # Updating state with new message
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.flip_roles()

        # Generate response
        assistant_message = generate(
            model=self.llm,
            messages=messages,
            tools=self.tools,
            **self.llm_args,
        )

        user_response = assistant_message.content
        logger.debug(f"Response: {user_response}")

        user_message = UserMessage(
            role="user",
            content=user_response,
            cost=assistant_message.cost,
            usage=assistant_message.usage,
            raw_data=assistant_message.raw_data,
        )

        # flip the requestor of the tool calls
        if assistant_message.tool_calls is not None:
            user_message.tool_calls = []
            for tool_call in assistant_message.tool_calls:
                user_message.tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                        requestor="user",
                    )
                )

        # Updating state with response
        state.messages.append(user_message)
        return user_message, state


class DummyUser(UserSimulator):
    """A dummy user to run a agent solo simulation."""

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        return UserState(messages=[], system_messages=[])

    def is_stop(cls, message: UserMessage) -> bool:
        raise NotImplementedError("DummyUser does not support stop messages")

    def set_seed(self, seed: int):
        pass

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> tuple[UserMessage, UserState]:
        raise NotImplementedError("DummyUser does not support generate_next_message")


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


class GPT41UserSimulator(BaseUser):
    """User simulator that uses zidongtaichu API."""

    def __init__(
        self,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[UserInstructions] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(instructions=instructions, llm=llm, llm_args=llm_args)
        self.tools = tools
        # Initialize OpenAI client with 中转 API
        self.client = OpenAI(
            base_url=USER_LLM_API_BASE,
            api_key=USER_LLM_API_KEY,
        )
        self.model = USER_LLM_MODEL
        self._seed = None

    @property
    def global_simulation_guidelines(self) -> str:
        use_tools = self.tools is not None
        return get_global_user_sim_guidelines(use_tools=use_tools)

    @property
    def system_prompt(self) -> str:
        if self.instructions is None:
            logger.warning("No instructions provided for user simulator")

        system_prompt = SYSTEM_PROMPT.format(
            global_user_sim_guidelines=self.global_simulation_guidelines,
            instructions=self.instructions,
        )
        return system_prompt

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        if message_history is None:
            message_history = []
        assert all(is_valid_user_history_message(m) for m in message_history), (
            "Invalid user message history."
        )

        user_state = UserState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
        return user_state

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        if message.is_tool_call():
            return False
        assert message.content is not None
        return (
            STOP in message.content
            or TRANSFER in message.content
            or OUT_OF_SCOPE in message.content
        )

    def set_seed(self, seed: int):
        """Set the seed for reproducibility."""
        self._seed = seed
        if self.llm_args is None:
            self.llm_args = {}
        self.llm_args["seed"] = seed

    def _messages_to_openai_format(self, messages: list[Message]) -> list[dict]:
        """Convert messages to OpenAI API format."""
        openai_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content or ""})
            elif hasattr(msg, 'role'):
                # For user simulator, we flip roles: assistant -> user, user -> assistant
                role = msg.role
                if role == "assistant":
                    role = "user"
                elif role == "user":
                    role = "assistant"
                openai_messages.append({"role": role, "content": msg.content or ""})
        return openai_messages

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        return self._generate_next_message(message, state)

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """Get the response from the user simulator using zidongtaichu API."""
        # Updating state with new message
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.flip_roles()

        # Convert to OpenAI format
        openai_messages = self._messages_to_openai_format(messages)

        # Build API call kwargs
        api_kwargs = {
            "model": self.model,
            "messages": openai_messages,
        }

        # Add temperature if specified
        temperature = self.llm_args.get("temperature", 0.0) if self.llm_args else 0.0
        api_kwargs["temperature"] = temperature

        # Add seed for reproducibility
        if self._seed is not None:
            api_kwargs["seed"] = self._seed

        try:
            response = self.client.chat.completions.create(**api_kwargs)
            content = response.choices[0].message.content or ""

            # Strip thinking content
            content = strip_thinking_content(content)

            # Calculate usage info
            usage = None
            if response.usage:
                usage = {
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                }

            logger.debug(f"zidongtaichu Response: {content}")

        except Exception as e:
            logger.error(f"Error calling zidongtaichu API: {e}")
            raise e

        user_message = UserMessage(
            role="user",
            content=content,
            cost=0.0,  # Cost calculation not available for 中转 API
            usage=usage,
            raw_data={"model": self.model, "response": str(response)},
        )

        # Updating state with response
        state.messages.append(user_message)
        return user_message, state
