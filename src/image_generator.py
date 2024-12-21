import requests
import os
import random
from utils import *
from config import *
from status import *
from uuid import uuid4  # Import uuid4 for generating unique IDs

class ImageGenerator:
    def __init__(self):
        pass

    def generate_image(self, prompt: str) -> str:
        image_gen = get_image_gen()
        image_model = get_image_model()
        
        info(f"Using: {image_gen} With Model: {image_model}")
        if image_gen == "prodia":
            s = requests.Session()
            headers = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            resp = s.get(
                "https://api.prodia.com/generate",
                params={
                    "new": "true",
                    "prompt": prompt,
                    "model": image_model,
                    "negative_prompt": "verybadimagenegative_v1.3",
                    "steps": "20",
                    "cfg": "7",
                    "seed": random.randint(1, 10000),
                    "sample": "DPM++ 2M Karras",
                    "aspect_ratio": "square"
                },
                headers=headers
            )
            
            job_id = resp.json()['job']
            info(f"Job created with ID: {job_id}")
            
            while True:
                time.sleep(5)
                status = s.get(f"https://api.prodia.com/job/{job_id}", headers=headers).json()
                if status["status"] == "succeeded":
                    img_data = s.get(f"https://images.prodia.xyz/{job_id}.png?download=1", headers=headers).content
                    image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
                    with open(image_path, "wb") as f:
                        f.write(img_data)
                    success(f"Image saved to: {image_path}")
                    return image_path

        elif image_gen == "hercai":
            url = f"https://hercai.onrender.com/{image_model}/text2image?prompt={prompt}"
            r = requests.get(url)
            parsed = r.json()

            if "url" in parsed and parsed["url"]:
                image_url = parsed["url"]
                image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
                with open(image_path, "wb") as f:
                    image_data = requests.get(image_url).content
                    f.write(image_data)
                success(f"Image saved to: {image_path}")
                return image_path
            else:
                warning("No image URL in Hercai response")

        elif image_gen == "pollinations":
            response = requests.get(f"https://image.pollinations.ai/prompt/{prompt}{random.randint(1,10000)}")
            if response.status_code == 200:
                image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
                with open(image_path, "wb") as f:
                    f.write(response.content)
                success(f"Image saved to: {image_path}")
                return image_path
            else:
                warning(f"Pollinations request failed with status code: {response.status_code}")

        warning(f"Image generation with {image_gen} was unsuccessful")
        return None