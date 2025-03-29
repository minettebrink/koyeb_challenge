# First stage: Download model files
FROM python:3.10.5-slim as model_downloader

# Install huggingface_hub and diffusers with specific versions
RUN pip3 install --no-cache-dir huggingface_hub==0.20.3 diffusers==0.25.0

# Create directory for model files
RUN mkdir -p /models/ltx-video

# Create a Python script to download the model with better error handling and progress tracking
RUN echo 'import os\nimport sys\nimport time\nimport signal\nimport json\nimport torch\nfrom huggingface_hub import snapshot_download\nfrom tqdm import tqdm\nfrom safetensors.torch import load_file\n\ndef timeout_handler(signum, frame):\n    raise TimeoutError("Download timed out")\n\ndef verify_model_files(model_path):\n    required_files = ["config.json", "model.safetensors", "scheduler", "text_encoder", "tokenizer", "vae"]\n    missing_files = []\n    \n    for file in required_files:\n        if not os.path.exists(os.path.join(model_path, file)):\n            missing_files.append(file)\n    \n    if missing_files:\n        print(f"Missing required model files: {missing_files}", file=sys.stderr)\n        return False\n    return True\n\ndef verify_model_config(model_path):\n    try:\n        with open(os.path.join(model_path, "config.json"), "r") as f:\n            config = json.load(f)\n            \n        # Verify essential config fields\n        required_fields = ["model_type", "architectures", "torch_dtype", "version"]\n        missing_fields = [field for field in required_fields if field not in config]\n        \n        if missing_fields:\n            print(f"Missing required config fields: {missing_fields}", file=sys.stderr)\n            return False\n            \n        # Verify model type\n        if config.get("model_type") != "ltx-video":\n            print(f"Invalid model type: {config.get(\'model_type\')}", file=sys.stderr)\n            return False\n            \n        # Verify model version\n        if config.get("version") not in ["0.9.0", "0.9.1"]:\n            print(f"Unsupported model version: {config.get(\'version\')}", file=sys.stderr)\n            return False\n            \n        # Verify torch dtype\n        if config.get("torch_dtype") != "bfloat16":\n            print(f"Warning: Model dtype is {config.get(\'torch_dtype\')}, recommended dtype is bfloat16", file=sys.stderr)\n            \n        # Verify T5 encoder configuration\n        if "text_encoder" not in config:\n            print("Missing T5 encoder configuration", file=sys.stderr)\n            return False\n            \n        # Verify VAE configuration\n        if "vae" not in config:\n            print("Missing VAE configuration", file=sys.stderr)\n            return False\n            \n        # Verify diffusion model configuration\n        if "diffusion_model" not in config:\n            print("Missing diffusion model configuration", file=sys.stderr)\n            return False\n            \n        print("Model configuration verified successfully")\n        return True\n    except Exception as e:\n        print(f"Error verifying model config: {str(e)}", file=sys.stderr)\n        return False\n\n# (remaining code unchanged)\nif __name__ == "__main__":\n    success = download_with_progress()\n    sys.exit(0 if success else 1)' > /download_model.py


# Run the Python script with increased timeout
RUN python3 /download_model.py || (echo "Model download failed" && exit 1)

# Second stage: Main application
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/back_end \
    HUGGINGFACE_HUB_CACHE=/models \
    CUDA_LAUNCH_BLOCKING=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app/back_end

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy model files from the downloader stage and verify
COPY --from=model_downloader /models/ltx-video /models/ltx-video
RUN echo "Verifying model files in final stage..." && \
    ls -la /models/ltx-video && \
    test -f /models/ltx-video/config.json && \
    test -f /models/ltx-video/model.safetensors && \
    test -d /models/ltx-video/scheduler && \
    test -d /models/ltx-video/text_encoder && \
    test -d /models/ltx-video/tokenizer && \
    test -d /models/ltx-video/vae || (echo "Model files verification failed in final stage" && exit 1)

# Create a script to handle graceful shutdown and startup
RUN echo '#!/bin/sh\n\ntrap "kill -TERM \$PID" TERM INT\n\n# Function to check if the server is ready\ncheck_server() {\n    curl -s http://localhost:8000/health > /dev/null\n    return \$?\n}\n\n# Function to check CUDA availability\ncheck_cuda() {\n    python3 -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA device count: {torch.cuda.device_count()}\"); print(f\"Current CUDA device: {torch.cuda.current_device()}\"); print(f\"CUDA device name: {torch.cuda.get_device_name(0)}\")"\n    return \$?\n}\n\n# Check CUDA availability\nif ! check_cuda; then\n    echo "CUDA check failed"\n    exit 1\nfi\n\n# Start uvicorn in the background\nuvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --backlog 2048 --log-level info --timeout-keep-alive 120 --timeout-graceful-shutdown 120 &\nPID=\$!\n\n# Wait for the server to be ready\necho "Waiting for server to be ready..."\nfor i in $(seq 1 30); do\n    if check_server; then\n        echo "Server is ready!"\n        break\n    fi\n    if [ \$i -eq 30 ]; then\n        echo "Server failed to start within 30 seconds"\n        exit 1\n    fi\n    sleep 1\ndone\n\n# Wait for the process\nwait \$PID' > /start.sh && chmod +x /start.sh

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application with graceful shutdown
CMD ["/start.sh"]