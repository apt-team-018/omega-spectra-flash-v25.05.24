import torch
import torch.nn as nn
from diffusers import OmegaSpectraV2Pipeline, DPMSolverSinglestepScheduler
import requests
from io import BytesIO
import os
import re
from pydantic import BaseModel, Field
from typing import Optional
import shutil
import random
import math
from huggingface_hub import HfFolder, snapshot_download
import uuid

class OmegaSpectraFlashInference:
    def __init__(self, device='cuda:0'):
        """
        Initialize the inference class with specified device.
        Input:
            device (str): The device (e.g., 'cuda:0' or 'cpu') where the model will run.
        """
        model_path = os.getenv("MODEL_PATH", "")
        self.image_upload_endpoint = os.getenv("IMAGE_UPLOAD_ENDPOINT", "")
        self.image_upload_apikey = os.getenv("IMAGE_UPLOAD_APIKEY", "")
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

        print("prompt - ", prompt)
        print("negative_prompt - ", negative_prompt)
        print("steps - ", steps)
        print("guidance_scale - ", guidance_scale)
        print("width - ", width)
        print("seed_generator - ", seed_generator)

        result = self.model(
            prompt=prompt,
            negative_prompt= negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            scheduler = self.scheduler,
            width=width,
            height=height,
            # generator=seed_generator,
            censored=False
        )

        image = result.images[0]
        uploaded_image_data = self.upload_image(image)
        return [
                {
                    "index": 0,
                    "image": uploaded_image_data,
                }
            ]

    def upload_image(self, image_object):
        """
               Uploads a PIL Image object to a specified server and returns the URL.
               Input:
                   image_object (PIL.Image): The image to be uploaded.
               Output:
                   str: URL of the uploaded image.
        """
        headers = {
                   'accept': 'multipart/form-data',
                   'x-api-key': self.image_upload_apikey
                   }
        img_byte_arr = BytesIO()
        image_object.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        files = {'image': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(f"{self.image_upload_endpoint}?requestId={str(uuid.uuid4())}", headers=headers, files=files)
        return response.json()['data']