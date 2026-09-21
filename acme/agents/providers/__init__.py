"""LLM provider adapters.

Each provider implements the LLMProvider protocol:
    generate(messages: list[dict]) -> str

Providers are intentionally thin — all agent logic (looping, parsing,
error recovery) lives in `acme/agents/llm_agent.py`.
"""

from acme.agents.providers.openai import OpenAIProvider
from acme.agents.providers.deepseek import DeepSeekProvider
from acme.agents.providers.openai_compatible import OpenAICompatibleProvider

__all__ = [
    "OpenAIProvider",
    "DeepSeekProvider",
    "OpenAICompatibleProvider",
]