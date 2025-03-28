# First stage: Download model files
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime as model_downloader

# Install huggingface_hub
RUN pip3 install --no-cache-dir huggingface_hub

# Create directory for model files
RUN mkdir -p /app/back_end/models/ltx-video

# Download model files
RUN python3 -c "import os; from huggingface_hub import snapshot_download; try: model_path = '/app/back_end/models/ltx-video'; snapshot_download('Lightricks/LTX-Video', local_dir=model_path, local_dir_use_symlinks=False, repo_type='model', ignore_patterns=['*.md', '*.txt']); print('Model download successful'); except Exception as e: print(f'Error downloading model: {str(e)}'); exit(1)"

# Second stage: Main application
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

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

# Copy each model file individually to leverage layer caching
COPY --from=model_downloader /app/back_end/models/ltx-video/model.safetensors /app/back_end/models/ltx-video/
COPY --from=model_downloader /app/back_end/models/ltx-video/config.json /app/back_end/models/ltx-video/
COPY --from=model_downloader /app/back_end/models/ltx-video/scheduler /app/back_end/models/ltx-video/scheduler/
COPY --from=model_downloader /app/back_end/models/ltx-video/text_encoder /app/back_end/models/ltx-video/text_encoder/
COPY --from=model_downloader /app/back_end/models/ltx-video/tokenizer /app/back_end/models/ltx-video/tokenizer/
COPY --from=model_downloader /app/back_end/models/ltx-video/unet /app/back_end/models/ltx-video/unet/
COPY --from=model_downloader /app/back_end/models/ltx-video/vae /app/back_end/models/ltx-video/vae/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uvicorn with production settings and memory optimization
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info", "--timeout-keep-alive", "120"]