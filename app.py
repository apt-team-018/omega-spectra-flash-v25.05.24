from fastapi import FastAPI, HTTPException, Header, Depends, Request, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from PIL import Image
import requests
import io
import torch
import torch.nn as nn
import asyncio
from itertools import cycle
import re
import os
import random
from diffusers import OmegaSpectraV2Pipeline, DPMSolverSinglestepScheduler
from io import BytesIO
import shutil
import math
from huggingface_hub import HfFolder, snapshot_download
import uuid
import time
import json
from datetime import datetime

app = FastAPI()

# Check if all required environment variables are available, not null, and non-empty
required_env_vars = ["MODEL_PATH", "IMAGE_UPLOAD_ENDPOINT", "IMAGE_UPLOAD_APIKEY", "UPLOAD_IMAGE_STATIC_PATH"]

for var in required_env_vars:
    value = os.getenv(var)
    if not value:
        raise RuntimeError(f"Environment variable {var} is not set or is empty")

try:
    queue_batch_size = int(os.getenv("QUEUE_BATCH_SIZE", 10))
except ValueError:
    queue_batch_size = 10  # Default value if conversion fails

# Retrieve API keys from environment variable, split by comma, filter valid keys
api_keys_raw = os.getenv("API_KEYS", "")
API_KEYS = {key: {'type': 'standard'} for key in api_keys_raw.split(',') if len(key) == 32 and re.match(r'^[a-zA-Z0-9]+$', key)}

image_upload_endpoint = os.getenv("IMAGE_UPLOAD_ENDPOINT", "")
image_upload_apikey = os.getenv("IMAGE_UPLOAD_APIKEY", "")
upload_image_static_path = os.getenv("UPLOAD_IMAGE_STATIC_PATH", "")

api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Depends(api_key_header)):
    if API_KEYS and (api_key not in API_KEYS):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

class RequestBody(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = Field(default="", max_length=210)
    steps: Optional[int] = Field(default=8, ge=4, le=50)
    num_of_images: Optional[int] = Field(default=1, ge=1, le=4)
    resolution: Optional[str] = Field(default="1024|1024")
    seed: Optional[int] = Field(default_factory=lambda: random.randint(1, 2147483647), ge=0, le=2147483647)
    guidance_scale: Optional[float] = Field(default=3.0, ge=0.1, le=6.0)

    @validator("resolution")
    def validate_resolution(cls, v):
        valid_resolutions = ["512|512", "768|768", "1024|1024", "1280|1280", "1536|1536", "2048|2048", "768|512", "1024|768", "1536|1024", "1280|960", "1536|1152", "2048|1536", "2048|1152", "1280|1024", "512|768", "768|1024", "1024|1536", "960|1280", "1152|1536", "1536|2048", "1152|2048", "1024|1280"]
        if v not in valid_resolutions:
            raise ValueError(f"Resolution must be one of {valid_resolutions}")
        return v

class OmegaSpectraFlashInference:
    def __init__(self, device='cuda:0'):
        """
        Initialize the inference class with specified device.
        Input:
            device (str): The device (e.g., 'cuda:0' or 'cpu') where the model will run.
        """
        model_path = os.getenv("MODEL_PATH", "")
        token = os.getenv("TOKEN", "")
        print("Args: ", {"MODEL_PATH": model_path, "device": device})
        
        if token != None and len(token) > 0:
            hf_folder = HfFolder()
            hf_folder.save_token(token)

        # Check for GPU availability and set up DataParallel if multiple GPUs are available
        self.device = device

        MODEL_CACHE = "cache"
        if os.path.exists(MODEL_CACHE):
            shutil.rmtree(MODEL_CACHE)
        os.makedirs(MODEL_CACHE, exist_ok=True)

        try:
            self.model = OmegaSpectraV2Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
        except:
            print("Model not found in cache. Downloading...")
            self.model = OmegaSpectraV2Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            self.model.save_pretrained(save_directory=MODEL_CACHE, safe_serialization=True)

        self.scheduler = DPMSolverSinglestepScheduler.from_config(self.model.scheduler.config, timestep_spacing="trailing")

        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.device)
        self.model = self.model.to(device)
    
    def base(self, x):
        return int(8 * math.floor(int(x)/8))
    
    def generate_image_data(self, prompt, negative_prompt,steps, num_of_images, resolution, seed, guidance_scale):
        if seed == None or seed == "" or seed == 0:
            seed = random.randint(1, 2147483647)
        seed_generator = torch.Generator('cuda').manual_seed(seed)
        
        resolution_split = resolution.split("|")
        width = self.base(resolution_split[0])
        height = self.base(resolution_split[1])
        
        negative_prompt = f"(deformed iris, deformed pupils, nude, sex, adult, pornography, porn, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck,{negative_prompt}"

        result = self.model(
            prompt=prompt,
            negative_prompt= negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            scheduler = self.scheduler,
            num_images_per_prompt = num_of_images,
            width=width,
            height=height,
            generator=seed_generator,
            censored=False
        )

        return result.images

# Determine the number of GPUs available
num_gpus = torch.cuda.device_count()
gpus = [f'cuda:{i}' for i in range(num_gpus)]
models = {gpu: OmegaSpectraFlashInference(device=gpu) for gpu in gpus}
gpu_semaphore = asyncio.Semaphore(num_gpus)  # Semaphore to limit concurrent requests to the number of GPUs

# Set the queue size to queue_batch_size times the number of GPUs
queue_size = num_gpus * queue_batch_size
queue = asyncio.Queue(maxsize=queue_size)  # Limit queue size dynamically

print("Configs: ", {"QUEUE SIZE": queue_size})

# Cycle through available GPUs
model_cycle = cycle(models.keys())


def upload_images(image_objects, file_ids):
    """
    Uploads multiple PIL Image objects to a specified server and returns a list of URLs.
    Input:
        image_objects (list of PIL.Image): The images to be uploaded.
    Output:
        list of str: URLs of the uploaded images.
    """
    try:
        # Prepare the data for the API request
        files = []
    
        # Iterate over the image objects and append each image to the files
        for idx, image_object in enumerate(image_objects):
            img_byte_arr = BytesIO()
            image_object.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            files.append(('images', ('image.png', img_byte_arr, 'image/png')))

        payload = {
            'fileIds': file_ids
        }
        
        # Set the headers for the API request
        headers = {
            'x-api-key': image_upload_apikey
        }

        # Send the images to the API endpoint
        response = requests.post(
            f"{image_upload_endpoint}?requestId={str(uuid.uuid4())}",
            headers=headers,
            files=files,
            data=payload
        )

        if response.status_code == 200:
            response_json = response.json()
            if response_json.get('success') == True:
                image_urls = [item['fileUrl'] for item in response_json.get('data', [])]
                
                return {
                    'status': 1,
                    'message': "Response",
                    'data': image_urls,
                    'code': 200,
                }
            else:
                return {
                    'status': 0,
                    'message': "Error: Failed to upload images",
                    'data': {},
                    'code': 400,
                }
        else:
            return {
                'status': 0,
                'message': "Error: Failed to upload images",
                'data': {},
                'code': 400,
            }
    except Exception as e:
        return {
            'status': 0,
            'message': "Error: Failed to upload images",
            'data': {},
            'code': 400,
        }

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
            return images
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": 0, "message": "Something went wrong", "data": {}, "code": 500, "error_id": "image_generation"})

@app.post("/generate-images/")
async def generate_image(request_body: RequestBody, api_key: str = Depends(get_api_key) if API_KEYS else None):
    try:
        global upload_image_static_path
        
        if queue.qsize() >= queue_size:
            return JSONResponse(status_code=429, content={"status": 0, "message": "Queue is full. Please try again later.", "data": [], "code": 429, "error_id": "queue_full"})
    
        await queue.put(request_body)
        
        images = await handle_request(request_body)
        await queue.get()
        queue.task_done()
    
        if isinstance(images, JSONResponse):
            return images

        current_date = datetime.utcnow().strftime('%d-%m-%Y')
        unique_ids = [f"{current_date}/{uuid.uuid4()}_{int(time.time())}" for _ in images]

        uploaded_image_urls = []
        for item in unique_ids:
            uploaded_image_urls.append(f"{upload_image_static_path}{item}.png")

        background_tasks = BackgroundTasks()
        background_tasks.add_task(upload_images, images, json.dumps(unique_ids))

        return JSONResponse(status_code=200, content={"status": 1, "message": "Image generated", "data": uploaded_image_urls, "code": 200}, background=background_tasks)
            
        # upload_images_result =  upload_images(images, json.dumps(unique_ids))
        
        # if upload_images_result['status'] == 1:
        #     return JSONResponse(status_code=200, content={"status": 1, "message": "Image generated", "data": uploaded_image_urls, "code": 200})
        # else:
        #     return JSONResponse(status_code=400, content={"status": 0, "message": "Error: Failed to upload images", "data": {}, "code": 400, "error_id": "image_upload"})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": 0, "message": "Something went wrong", "data": {}, "code": 500, "error_id": "unknown"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)