FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Upgrade packaging tools
RUN pip install --upgrade pip setuptools wheel

# Install pinned numpy first (avoid ABI issues)
RUN pip install numpy==1.26.4

# Install CPU-only PyTorch explicitly
RUN pip install \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
RUN pip install -r requirements.txt

# VERIFY critical packages during build (important)
RUN python -c "import chromadb; print('chromadb OK')"
RUN python -c "import sentence_transformers; print('sentence_transformers OK')"
RUN python -c "import torch; print('torch OK')"

# Copy source code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
