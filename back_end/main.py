from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import Union, Optional
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/generate-video")
async def generate_video(
    prompt: str = Form(...),
    image_url: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    seed: Optional[int] = Form(None),
    inference_steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None)
):
    try:
        if image_url is None and image is None:
            raise HTTPException(status_code=400, detail="Either image_url or image file must be provided")

        # Here you would eventually add the LTX-Video model integration
        # For now, just return the parameters that would be used
        return {
            "status": "success",
            "parameters": {
                "prompt": prompt,
                "image_url": image_url,
                "image_filename": image.filename if image else None,
                "seed": seed,
                "inference_steps": inference_steps,
                "guidance_scale": guidance_scale
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))