from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import Union, Optional
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import base64
from io import BytesIO
import torch
from diffusers import LTXPipeline, LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import tempfile
from PIL import Image

app = FastAPI()

# Get allowed origins from environment variable or use default
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,https://*.koyeb.app,https://definite-weasel-challenge-f82bb88d.koyeb.app"
).split(",")

# Add CORS middleware with more flexible configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Allow both local and Koyeb domains
    allow_credentials=False,  # We don't need credentials
    allow_methods=["GET", "POST", "OPTIONS"],  # Specify allowed methods
    allow_headers=["Content-Type", "Accept"],  # Specify allowed headers
    expose_headers=["Content-Type"],  # Specify exposed headers
)

# Initialize the model
MODEL_PATH = "/app/models/ltx-video"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize pipelines
text_to_video_pipe = LTXPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
).to(device)

image_to_video_pipe = LTXImageToVideoPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
).to(device)

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": device}

class VideoGenerationRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    seed: Optional[int] = None
    inference_steps: int = 50  # Required parameter with default
    guidance_scale: float = 7.5  # Required parameter with default
    width: int = 704  # Required parameter with default
    height: int = 480  # Required parameter with default
    num_frames: int = 161  # Required parameter with default

    def validate_dimensions(self):
        if self.width % 32 != 0 or self.height % 32 != 0:
            raise ValueError("Width and height must be divisible by 32")
        if (self.num_frames - 1) % 8 != 0:
            raise ValueError("Number of frames must be divisible by 8 plus 1")
        if self.inference_steps < 1:
            raise ValueError("Inference steps must be positive")
        if self.guidance_scale < 0:
            raise ValueError("Guidance scale must be non-negative")

@app.post("/generate-video")
async def generate_video(request: VideoGenerationRequest):
    try:
        # Validate dimensions and parameters
        request.validate_dimensions()
        
        # Create a temporary directory for the output video
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            
            if request.image_url is not None or request.image_base64 is not None:
                # Image-to-video generation
                if request.image_base64:
                    # Decode base64 image
                    image_data = base64.b64decode(request.image_base64)
                    image = Image.open(BytesIO(image_data))
                else:
                    # Load image from URL
                    image = load_image(request.image_url)
                
                video = image_to_video_pipe(
                    image=image,
                    prompt=request.prompt,
                    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
                    width=request.width,
                    height=request.height,
                    num_frames=request.num_frames,
                    num_inference_steps=request.inference_steps,
                    guidance_scale=request.guidance_scale,
                    generator=torch.manual_seed(request.seed) if request.seed else None,
                ).frames[0]
            else:
                # Text-to-video generation
                video = text_to_video_pipe(
                    prompt=request.prompt,
                    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
                    width=request.width,
                    height=request.height,
                    num_frames=request.num_frames,
                    num_inference_steps=request.inference_steps,
                    guidance_scale=request.guidance_scale,
                    generator=torch.manual_seed(request.seed) if request.seed else None,
                ).frames[0]
            
            # Export video
            export_to_video(video, output_path, fps=24)
            
            # Read the video file and convert to base64
            with open(output_path, "rb") as video_file:
                video_base64 = base64.b64encode(video_file.read()).decode()
            
            return {
                "status": "success",
                "video_base64": video_base64,
                "parameters": {
                    "prompt": request.prompt,
                    "has_image": request.image_url is not None or request.image_base64 is not None,
                    "seed": request.seed,
                    "inference_steps": request.inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "width": request.width,
                    "height": request.height,
                    "num_frames": request.num_frames
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))