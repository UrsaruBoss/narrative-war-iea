# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (minimal, but enough for common ML/text libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps first (better build cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project code
COPY . /app

# Defaults for HF / sentence-transformers cache (mounted via compose)
ENV HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf \
    SENTENCE_TRANSFORMERS_HOME=/cache/sentence-transformers

# Default command just prints help
CMD ["python", "-m", "pip", "--version"]
