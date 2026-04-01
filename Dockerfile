FROM python:3.10

RUN pip install numpy==1.26.4
# --------------------------------------------------
# Install system dependencies
# --------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libreoffice \
    tesseract-ocr \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*



# --------------------------------------------------
# Set working directory
# --------------------------------------------------
WORKDIR /app

COPY requirements.txt .

# --------------------------------------------------
# Upgrade pip tools
# --------------------------------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel

# --------------------------------------------------
# Install PyTorch CPU
# --------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    torchaudio==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# --------------------------------------------------
# Install Whisper
# --------------------------------------------------
RUN python3 -m pip install --no-cache-dir openai-whisper

# --------------------------------------------------
# Install project requirements
# --------------------------------------------------
RUN python3 -m pip install --no-cache-dir -r requirements.txt

RUN python -c "import whisper; whisper.load_model('base')"
# --------------------------------------------------
# Copy project files
# --------------------------------------------------
COPY . .

ENV PYTHONPATH=/app

# --------------------------------------------------
# Expose API
# --------------------------------------------------
EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]