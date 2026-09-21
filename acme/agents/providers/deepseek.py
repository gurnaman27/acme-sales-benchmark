"""DeepSeek LLM provider adapter using OpenAI client."""

from __future__ import annotations

from acme.agents.providers.openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider adapter."""

    provider_name = "deepseek"
    default_model = "deepseek-chat"
    default_base_url = "https://api.deepseek.com"
    api_key_env = "DEEPSEEK_API_KEY"

