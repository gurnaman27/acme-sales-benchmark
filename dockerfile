FROM python:3.11-slim

WORKDIR /app

# System deps for compiling numpy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first (better layer caching)
COPY pyproject.toml README.md ./

# Copy source code and trained models
COPY acme/ ./acme/
COPY models/ ./models/

# Install package with LLM extras (openai for OpenAI + DeepSeek)
RUN pip install --no-cache-dir -e ".[llm]"

# Expose the A2A server port
EXPOSE 9009

# Run the A2A server on all interfaces (required inside containers)
CMD ["uvicorn", "acme.green_agent.a2a_server:app", "--host", "0.0.0.0", "--port", "9009"]