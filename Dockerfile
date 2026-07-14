FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_MODEL=qwen2.5:3b
ENV OLLAMA_URL=http://127.0.0.1:11434

# 1. Install system dependencies & official Ollama binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    curl \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy application code and scripts
COPY . .
RUN chmod +x entrypoint.sh && mkdir -p data .data outputs .mission_state /root/.ollama /root/.cache/huggingface

# 4. Prefetch HuggingFace embedding & reranker models into container cache
RUN python scripts/prefetch_models.py

# 5. Start Ollama in background during build and pull the default model so weights are baked in
RUN (ollama serve >/dev/null 2>&1 &) && \
    sleep 5 && \
    ollama pull ${OLLAMA_MODEL} && \
    pkill ollama || true

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["cli"]
