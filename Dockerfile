FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

COPY app ./app
COPY tests ./tests
COPY architecture ./architecture
COPY models ./models
COPY requirements.txt .
COPY .env.example ./.env.example
COPY README.md ./README.md

RUN pip install --no-cache-dir -r requirements.txt

RUN python - <<'PY'
import importlib
m = importlib.import_module("architecture.multimodal_arch")
print("OK architecture:", hasattr(m, "build_model"), m.ATMOS_KEYS)
PY


EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
