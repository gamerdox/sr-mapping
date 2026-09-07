FROM python:3.12-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements with CPU-optimized PyTorch
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application source code and checkpoints
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY checkpoints/ ./checkpoints/
COPY data/ ./data/
COPY src/ ./src/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import requests; r=requests.get('http://localhost:8000/health'); exit(0 if r.ok else 1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
