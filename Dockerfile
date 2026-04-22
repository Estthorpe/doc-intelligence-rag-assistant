FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
RUN python -m spacy download en_core_web_lg

FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

RUN mkdir -p data/raw data/processed logs models

ENV PATH=/root/.local/bin:$PATH

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]