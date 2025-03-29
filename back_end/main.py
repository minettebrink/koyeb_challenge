from fastapi import FastAPI, HTTPException
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os


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

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

class VideoGenerationRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    seed: Optional[int] = None
    inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None

@app.post("/generate-video")
async def generate_video(request: VideoGenerationRequest):
    try:
        if request.image_url is None and request.image_base64 is None:
            raise HTTPException(status_code=400, detail="Either image_url or image_base64 must be provided")

        # Here you would eventually add the LTX-Video model integration
        # For now, just return the parameters that would be used
        return {
            "status": "success",
            "parameters": {
                "prompt": request.prompt,
                "image_url": request.image_url,
                "has_image_base64": request.image_base64 is not None,
                "seed": request.seed,
                "inference_steps": request.inference_steps,
                "guidance_scale": request.guidance_scale
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))