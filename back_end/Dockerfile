# First stage: Download model files
FROM python:3.10-slim as model_downloader

# Install huggingface_hub and diffusers
RUN pip3 install --no-cache-dir huggingface_hub diffusers

# Create directory for model files
RUN mkdir -p /app/back_end/models/ltx-video

# Create a Python script to download the model with better error handling and progress tracking
RUN echo 'import os\nimport sys\nimport time\nfrom huggingface_hub import snapshot_download\nfrom tqdm import tqdm\n\ndef download_with_progress():\n    max_retries = 5\n    retry_delay = 30  # seconds\n    \n    for attempt in range(max_retries):\n        try:\n            model_path = "/app/back_end/models/ltx-video"\n            print(f"Starting model download (attempt {attempt + 1}/{max_retries})...")\n            \n            # Download with progress bar\n            snapshot_download(\n                "Lightricks/LTX-Video",\n                local_dir=model_path,\n                repo_type="model",\n                ignore_patterns=["*.md", "*.txt", "*.gif", "*.png", "*.jpg", "*.jpeg", "*.bin", "*.pt", "*.pth"],\n                local_files_only=False,\n                resume_download=True\n            )\n            \n            print("Model download successful")\n            return True\n        except Exception as e:\n            print(f"Error downloading model (attempt {attempt + 1}/{max_retries}): {str(e)}", file=sys.stderr)\n            if attempt < max_retries - 1:\n                print(f"Retrying in {retry_delay} seconds...")\n                time.sleep(retry_delay)\n            else:\n                print("Max retries reached. Download failed.", file=sys.stderr)\n                return False\n\nif __name__ == "__main__":\n    success = download_with_progress()\n    sys.exit(0 if success else 1)' > /download_model.py

# Run the Python script with increased timeout
RUN python3 /download_model.py || (echo "Model download failed" && exit 1)

# Second stage: Main application
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app/back_end

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create directory for model files
RUN mkdir -p /app/back_end/models/ltx-video

# Copy the application code
COPY . .

# Copy model files from the downloader stage
COPY --from=model_downloader /app/back_end/models/ltx-video /app/back_end/models/ltx-video

# Clean up any temporary files
RUN rm -rf /root/.cache/pip /root/.cache/huggingface

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uvicorn with production settings and memory optimization
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info --timeout-keep-alive 120"]