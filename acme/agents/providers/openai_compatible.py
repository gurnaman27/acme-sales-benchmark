import os
import time


class OpenAICompatibleProvider:
    """Base provider class for OpenAI-compatible HTTP APIs."""

    provider_name = "compatible"
    default_model = "llama3"
    default_base_url: str | None = "http://localhost:11434/v1"
    api_key_env = "OPENAI_COMPATIBLE_API_KEY"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or self.default_model
        self.api_key = api_key or os.environ.get(self.api_key_env)
        if not self.api_key and self.provider_name == "compatible":
            self.api_key = "nokey"
        self.base_url = base_url if base_url is not None else self.default_base_url
        self.name = f"{self.provider_name}_{self.model}"

    def generate(self, messages: list[dict]) -> str:
        if not self.api_key:
            raise ValueError(
                f"API key missing. Set {self.api_key_env} environment variable."
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is not installed. Install with `pip install openai`."
            )

        kwargs: dict[str, str] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        client = OpenAI(**kwargs)
        formatted_messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        ]

        max_retries = 5
        backoff = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=0.0,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                err_msg = str(e)
                if ("429" in err_msg or "rate" in err_msg.lower()) and attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise
        return ""

