from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from PIL import Image
import requests
import io
import torch
import asyncio
from itertools import cycle
import re
import os
import random
from generate_image_data_v2 import OmegaSpectraFlashInference  # Ensure this module is correctly set up

app = FastAPI()

# Determine the number of GPUs available
num_gpus = torch.cuda.device_count()
gpus = [f'cuda:{i}' for i in range(num_gpus)]
models = {gpu: OmegaSpectraFlashInference(device=gpu) for gpu in gpus}
gpu_semaphore = asyncio.Semaphore(num_gpus)  # Semaphore to limit concurrent requests to the number of GPUs

# Set the queue size to 4 times the number of GPUs
queue_size = num_gpus * 4
queue = asyncio.Queue(maxsize=queue_size)  # Limit queue size dynamically

# Cycle through available GPUs
model_cycle = cycle(models.keys())

# Check if all required environment variables are available, not null, and non-empty
api_keys_raw = os.getenv("API_KEYS", "")
API_KEYS = {key: {'type': 'standard'} for key in api_keys_raw.split(',') if len(key) == 32 and re.match(r'^[a-zA-Z0-9]+$', key)}

api_key_header = APIKeyHeader(name="X-API-Key")

request_counter = 0

def get_api_key(api_key: str = Depends(api_key_header)):
    if API_KEYS and (api_key not in API_KEYS):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

class RequestBody(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = Field(default="", max_length=50)
    steps: Optional[int] = Field(default=8, ge=4, le=50)
    num_of_images: Optional[int] = Field(default=1, ge=1, le=4)
    resolution: Optional[str] = Field(default="512x768")
    seed: Optional[int] = Field(default_factory=lambda: random.randint(1, 2147483647), ge=0, le=2147483647)
    guidance_scale: Optional[float] = Field(default=3.0, ge=0.1, le=6.0)

    @validator("resolution")
    def validate_resolution(cls, v):
        valid_resolutions = ["512|512", "768|768", "1024|1024", "1280|1280", "1536|1536", "2048|2048", "768|512", "1024|768", "1536|1024", "1280|960", "1536|1152", "2048|1536", "2048|1152", "1280|1024", "512|768", "768|1024", "1024|1536", "960|1280", "1152|1536", "1536|2048", "1152|2048", "1024|1280"]
        if v not in valid_resolutions:
            raise ValueError(f"Resolution must be one of {valid_resolutions}")
        return v

@app.get("/health")
async def health_check():
    return {
        "status": "online",
    }

async def process_request(model, request_body):
    images = await asyncio.to_thread(
        model.generate_image_data,
        request_body.prompt,
        negative_prompt=request_body.negative_prompt,
        num_of_images=request_body.num_of_images,
        seed=request_body.seed,
        resolution=request_body.resolution,
        guidance_scale=request_body.guidance_scale,
        steps=request_body.steps,
    )
    return images

async def handle_request(request_body):
    async with gpu_semaphore:
        selected_gpu = next(model_cycle)
        model = models[selected_gpu]
        try:
            images = await process_request(model, request_body)
            return JSONResponse(status_code=200, content={"status": 1, "message": "Image generated", "data": images, "code": 200})
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": 0, "message": "Something went wrong", "data": [], "code": 500, "error": str(e)})

@app.post("/generate-images/")
async def generate_image(request_body: RequestBody, api_key: str = Depends(get_api_key) if API_KEYS else None):
    global request_counter

    if queue.qsize() >= queue_size:
        return JSONResponse(status_code=429, content={"status": 0, "message": "Queue is full. Please try again later.", "data": [], "code": 429})

    await queue.put(request_body)
    request_counter += 1

    response = await handle_request(request_body)
    await queue.get()
    queue.task_done()

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)