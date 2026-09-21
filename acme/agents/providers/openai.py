"""OpenAI LLM provider adapter."""

from __future__ import annotations

from acme.agents.providers.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider adapter."""

    provider_name = "openai"
    default_model = "gpt-4o-mini"
    default_base_url = None
    api_key_env = "OPENAI_API_KEY"

