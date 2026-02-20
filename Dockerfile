FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel

# CPU-only torch (prevents CUDA downloads)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Whisper after CPU torch
RUN pip install --no-cache-dir openai-whisper

# rest of deps
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
