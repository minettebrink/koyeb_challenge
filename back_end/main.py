from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import Union, Optional
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # SvelteKit dev server origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

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