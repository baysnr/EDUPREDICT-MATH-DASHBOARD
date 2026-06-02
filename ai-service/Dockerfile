# ==============================================================================
# Production Dockerfile for EduPredict DKT Inference Engine
# ==============================================================================
# Uses python-slim for a lightweight container footprint.
# Recommends tensorflow-cpu for standard servers to avoid GPU CUDA package overhead.

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Install system utilities needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
# We use tensorflow-cpu for highly optimized, lightweight server deployment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    numpy \
    tensorflow-cpu \
    google-generativeai

# Copy application files
COPY inference_api.py .
COPY top_category.json .

# Copy model artifacts
COPY final/ final/

# Expose port (Hugging Face requires port 7860)
EXPOSE 7860

# Start application via Uvicorn on port 7860
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "7860"]
