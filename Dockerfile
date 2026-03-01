FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Install system build tools first
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU-only (avoids massive CUDA download)
RUN pip install --no-cache-dir torch==2.6.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir openai-whisper

# Now install rest of your requirements
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY run_dataset.py /app/run_dataset.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
