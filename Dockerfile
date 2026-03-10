FROM python:3.10-slim

# --------------------------------------------------
# Install system dependencies
# --------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libreoffice \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Set working directory
# --------------------------------------------------

WORKDIR /app

COPY requirements.txt requirements.txt

# --------------------------------------------------
# Upgrade pip tools
# --------------------------------------------------

RUN pip install --upgrade pip setuptools wheel

# --------------------------------------------------
# Install PyTorch CPU
# --------------------------------------------------

RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    torchaudio==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# --------------------------------------------------
# Install Whisper
# --------------------------------------------------

RUN pip install --no-cache-dir openai-whisper

# --------------------------------------------------
# Install YouTube downloader
# --------------------------------------------------

RUN pip install yt-dlp

# --------------------------------------------------
# Install project requirements
# --------------------------------------------------

RUN pip install -r requirements.txt

# --------------------------------------------------
# Fix dependency conflicts
# --------------------------------------------------

RUN pip uninstall -y numpy requests && \
    pip install numpy==1.26.4 requests==2.31.0

# --------------------------------------------------
# Copy project files
# --------------------------------------------------

COPY . /app

ENV PYTHONPATH=/app

# --------------------------------------------------
# Expose API
# --------------------------------------------------

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]