FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    HOUSE_PRICER_VAR=/data/var \
    COOKIE_SECURE=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

ARG CLIP_MODEL_ID=google/siglip2-base-patch16-224
RUN python -c "from transformers import AutoModel, AutoProcessor; \
AutoModel.from_pretrained('${CLIP_MODEL_ID}'); AutoProcessor.from_pretrained('${CLIP_MODEL_ID}')"

COPY . .

RUN useradd --create-home --uid 1000 house-pricer \
    && mkdir -p /data/var \
    && chown -R house-pricer:house-pricer /app /data
USER house-pricer

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fs "http://localhost:${PORT:-8501}/healthz" || exit 1

CMD ["./start.sh"]
