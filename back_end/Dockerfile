# Use CUDA-enabled base image with PyTorch support
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create directory for model files
RUN mkdir -p /app/models/ltx-video

# Copy the application code
COPY . .

# Download and chunk the model files
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Lightricks/LTX-Video', local_dir='/app/models/ltx-video', local_dir_use_symlinks=False, repo_type='model')"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uvicorn with production settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]