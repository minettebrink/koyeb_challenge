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

# Global variables for model and device
MODEL_VERSION = "Lightricks/LTX-Video-0.9.1"  # Using latest stable version
device = "cuda" if torch.cuda.is_available() else "cpu"
text_to_video_pipe = None
image_to_video_pipe = None

@app.on_event("startup")
async def startup_event():
    global text_to_video_pipe, image_to_video_pipe
    try:
        print(f"Loading models on device: {device}")
        print(f"Loading models from: {MODEL_VERSION}")
        
        # Load text-to-video pipeline
        print("Loading text-to-video pipeline...")
        text_to_video_pipe = LTXPipeline.from_pretrained(
            MODEL_VERSION,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        ).to(device)
        print("Text-to-video model loaded successfully")
        
        # Load image-to-video pipeline
        print("Loading image-to-video pipeline...")
        image_to_video_pipe = LTXImageToVideoPipeline.from_pretrained(
            MODEL_VERSION,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        ).to(device)
        print("Image-to-video model loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load models: {str(e)}"
        )

# Add a health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Check if models are loaded
        models_loaded = text_to_video_pipe is not None and image_to_video_pipe is not None
        
        # Check CUDA memory if available
        cuda_memory = None
        if device == "cuda":
            cuda_memory = {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
            }
        
        return {
            "status": "healthy" if models_loaded else "unhealthy",
            "device": device,
            "models_loaded": models_loaded,
            "model_version": MODEL_VERSION,
            "cuda_memory": cuda_memory
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

class VideoGenerationRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    seed: Optional[int] = None
    inference_steps: int = 40  # Default from documentation
    guidance_scale: float = 7.5  # Default from documentation
    width: int = 768  # Default from documentation
    height: int = 512  # Default from documentation
    num_frames: int = 161  # Default from documentation

    def validate_dimensions(self):
        if self.width % 32 != 0 or self.height % 32 != 0:
            raise ValueError("Width and height must be divisible by 32")
        if (self.num_frames - 1) % 8 != 0:
            raise ValueError("Number of frames must be divisible by 8 plus 1")
        if self.inference_steps < 1:
            raise ValueError("Inference steps must be positive")
        if self.guidance_scale < 0:
            raise ValueError("Guidance scale must be non-negative")
        if self.width < 512 or self.width > 1024:
            raise ValueError("Width must be between 512 and 1024")
        if self.height < 512 or self.height > 1024:
            raise ValueError("Height must be between 512 and 1024")
        if self.num_frames < 17 or self.num_frames > 161:
            raise ValueError("Number of frames must be between 17 and 161")

@app.post("/generate-video")
async def generate_video(request: VideoGenerationRequest):
    if text_to_video_pipe is None or image_to_video_pipe is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet")
        
    try:
        # Validate dimensions and parameters
        request.validate_dimensions()
        
        # Create a temporary directory for the output video
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            
            if request.image_url is not None or request.image_base64 is not None:
                # Image-to-video generation
                try:
                    if request.image_base64:
                        # Decode base64 image
                        image_data = base64.b64decode(request.image_base64)
                        image = Image.open(BytesIO(image_data))
                    else:
                        # Load image from URL
                        image = load_image(request.image_url)
                    
                    # Validate image dimensions
                    if image.width % 32 != 0 or image.height % 32 != 0:
                        raise ValueError("Input image dimensions must be divisible by 32")
                    
                    print(f"Generating video from image with dimensions: {image.width}x{image.height}")
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
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
                    raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
            else:
                # Text-to-video generation
                print(f"Generating video from text prompt: {request.prompt}")
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
            
            print(f"Exporting video to: {output_path}")
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
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))