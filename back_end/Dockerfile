# First stage: Download model files
FROM python:3.11-slim as model_downloader

# Install huggingface_hub and diffusers with specific versions
RUN pip3 install --no-cache-dir huggingface_hub


# Copy the download script
COPY download_model.py /download_model.py

# Run the Python script with increased timeout and retries
RUN python3 /download_model.py || (echo "Model download failed" && exit 1)

# Second stage: Main application
# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies (removed curl since it's no longer needed for healthcheck)
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files from the downloader stage and verify
COPY --from=model_downloader /models/ltx-video-2b-v0.9.safetensors /models/ltx-video-2b-v0.9.safetensors 

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uvicorn with debug logging
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]



