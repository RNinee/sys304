FROM oven/bun:1.3-debian

RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY milestone/3/backend/package.json milestone/3/backend/bun.lock ./
RUN bun install --frozen-lockfile --production

COPY milestone/3/backend/python/requirements.txt ./python/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
  && pip3 install --break-system-packages --no-cache-dir -r python/requirements.txt

COPY milestone/3/backend/src ./src
COPY milestone/3/backend/python ./python
COPY milestone/1/models/qwen2.5-1.5b-disaster-lora /models/qwen2.5-1.5b-disaster-lora

ENV QWEN_ADAPTER_DIR=/models/qwen2.5-1.5b-disaster-lora \
    PYTHON=python3 \
    PORT=8000 \
    HOST=0.0.0.0 \
    INFER_HOST=127.0.0.1 \
    INFER_PORT=9377 \
    INFER_BACKEND=int8 \
    INFER_WORKERS=1 \
    BATCH_MAX_SIZE=8 \
    BATCH_MAX_WAIT_MS=15 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/cache/huggingface

EXPOSE 8000
CMD ["bun", "run", "src/index.ts"]
