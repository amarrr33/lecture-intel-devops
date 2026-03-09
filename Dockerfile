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
# Install PyTorch CPU (small build)
# --------------------------------------------------

RUN pip install --no-cache-dir torch==2.2.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# --------------------------------------------------
# Install Whisper
# --------------------------------------------------

RUN pip install --no-cache-dir openai-whisper

# --------------------------------------------------
# Fix numpy compatibility
# --------------------------------------------------

RUN pip install "numpy<2"

# --------------------------------------------------
# Install other torch libs (CPU only)
# --------------------------------------------------

RUN pip install torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# --------------------------------------------------
# Install YouTube downloader
# --------------------------------------------------

RUN pip install yt-dlp

# --------------------------------------------------
# Install project requirements
# --------------------------------------------------

RUN pip install -r requirements.txt

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