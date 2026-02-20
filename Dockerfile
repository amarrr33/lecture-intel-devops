FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/requirements.txt /app/requirements.txt

# Install system build tools first
RUN pip install --upgrade pip setuptools wheel

# Install Whisper separately (CPU only, avoids massive CUDA download)
RUN pip install --no-cache-dir torch==2.2.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir openai-whisper

# Now install rest of your requirements
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]