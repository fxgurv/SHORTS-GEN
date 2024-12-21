import re
import json
import g4f
import requests
from utils import *
from config import *
from status import *
from constants import *
from typing import List
from uuid import uuid4  # Import uuid4 for generating unique IDs

def parse_model(model_name: str) -> any:
    if model_name == "gpt4":  # 018
        return g4f.models.gpt_4
    elif model_name == "gpt_4o":  # okay
        return g4f.models.gpt_4o
    elif model_name == "gigachat":  # 0 api key
        return g4f.models.gigachat
    elif model_name == "meta":  # 500
        return g4f.models.meta
    elif model_name == "llama3_8b_instruct":  # 018
        return g4f.models.llama3_8b_instruct
    elif model_name == "llama3_70b_instruct":
        return g4f.models.llama3_70b_instruct
    elif model_name == "codellama_34b_instruct":  # 500
        return g4f.models.codellama_34b_instruct
    elif model_name == "codellama_70b_instruct":  # 018
        return g4f.models.codellama_70b_instruct
    elif model_name == "mixtral_8x7b":  # 500
        return g4f.models.mixtral_8x7b
    elif model_name == "mistral_7b":  # 500
        return g4f.models.mistral_7b
    elif model_name == "mistral_7b_v02":  # 500
        return g4f.models.mistral_7b_v02
    elif model_name == "claude_v2":  # 018
        return g4f.models.claude_v2
    elif model_name == "claude_3_opus":  # 500
        return g4f.models.claude_3_opus
    elif model_name == "claude_3_sonnet":  # 500
        return g4f.models.claude_3_sonnet
    elif model_name == "claude_3_haiku":
        return g4f.models.claude_3_haiku
    elif model_name == "pi":  # 500
        return g4f.models.pi
    elif model_name == "dbrx_instruct":  # 018
        return g4f.models.dbrx_instruct
    elif model_name == "command_r_plus":  # 500
        return g4f.models.command_r_plus
    elif model_name == "blackbox":
        return g4f.models.blackbox
    elif model_name == "reka_core":  # 0 cookie
        return g4f.models.reka_core
    elif model_name == "nemotron_4_340b_instruct":
        return g4f.models.nemotron_4_340b_instruct
    elif model_name == "Phi_3_mini_4k_instruct":
        return g4f.models.Phi_3_mini_4k_instruct
    elif model_name == "Yi_1_5_34B_Chat":
        return g4f.models.Yi_1_5_34B_Chat
    elif model_name == "Nous_Hermes_2_Mixtral_8x7B_DPO":
        return g4f.models.Nous_Hermes_2_Mixtral_8x7B_DPO
    elif model_name == "llama_2_70b_chat":
        return g4f.models.llama_2_70b_chat
    elif model_name == "gemma_2_9b_it":
        return g4f.models.gemma_2_9b_it
    elif model_name == "gemma_2_27b_it":
        return g4f.models.gemma_2_27b_it
    else:
        # Default model is gpt3.5-turbo
        return g4f.models.gpt_35_turbo

class TextGenerator:
    def __init__(self, niche: str, language: str):
        self.niche = niche
        self.language = language
        self.subject = ""
        self.script = ""
        self.metadata = {}
        self.image_prompts = []
    
    def generate_response(self, prompt: str, model: any = None) -> str:
        info(f"Generating response for prompt: {prompt[:50]}...")
        if get_model() == "google":
            info("Using Google's Gemini model")
            import google.generativeai as genai
            genai.configure(api_key=get_gemini_api_key())
            model = genai.GenerativeModel('gemini-pro')
            response: str = model.generate_content(prompt).text
        else:
            info(f"Using model: {parse_model(get_model()) if not model else model}")
            response = g4f.ChatCompletion.create(
                model=parse_model(get_model()) if not model else model,
                messages=[{"role": "user", "content": prompt}]
            )
        info(f"Response generated successfully, length: {len(response)} characters")
        return response

    def generate_topic(self) -> str:
        info("Generating topic for YouTube video")
        completion = self.generate_response(f"Please generate a specific video idea that takes about the following topic: {self.niche}. Make it exactly one sentence. Only return the topic, nothing else.")

        if not completion:
            error("Failed to generate Topic.")
        else:
            self.subject = completion
            success(f"Generated topic: {completion}")

        return completion

    def generate_script(self) -> str:
        info("Generating script for YouTube video")
        prompt = f"""
        Generate a script for a video in 4 sentences, depending on the subject of the video.

        The script is to be returned as a string with the specified number of paragraphs.

        Here is an example of a string:
        "This is an example string."

        Do not under any circumstance reference this prompt in your response.

        Get straight to the point, don't start with unnecessary things like, "welcome to this video".

        Obviously, the script should be related to the subject of the video.
        
        YOU MUST NOT EXCEED THE 4 SENTENCES LIMIT. MAKE SURE THE 4 SENTENCES ARE SHORT.
        YOU MUST NOT INCLUDE ANY TYPE OF MARKDOWN OR FORMATTING IN THE SCRIPT, NEVER USE A TITLE.
        YOU MUST WRITE THE SCRIPT IN THE LANGUAGE SPECIFIED IN [LANGUAGE].
        ONLY RETURN THE RAW CONTENT OF THE SCRIPT. DO NOT INCLUDE "VOICEOVER", "NARRATOR" OR SIMILAR INDICATORS OF WHAT SHOULD BE SPOKEN AT THE BEGINNING OF EACH PARAGRAPH OR LINE. YOU MUST NOT MENTION THE PROMPT, OR ANYTHING ABOUT THE SCRIPT ITSELF. ALSO, NEVER TALK ABOUT THE AMOUNT OF PARAGRAPHS OR LINES. JUST WRITE THE SCRIPT
        
        Subject: {self.subject}
        Language: {self.language}
        """
        completion = self.generate_response(prompt)

        # Apply regex to remove *
        completion = re.sub(r"\*", "", completion)
        
        if not completion:
            error("The generated script is empty.")
            return
        
        if len(completion) > 5000:
            warning("Generated Script is too long. Retrying...")
            return self.generate_script()
        
        self.script = completion
        success(f"Generated script: {completion[:100]}... Length: {len(completion)} characters")
    
        return completion

    def generate_metadata(self) -> dict:
        info("Generating metadata for YouTube video")
        title = self.generate_response(f"Please generate a YouTube Video Title for the following subject, including hashtags: {self.subject}. Only return the title, nothing else. Limit the title under 100 characters.")

        if len(title) > 100:
            warning("Generated Title is too long. Retrying...")
            return self.generate_metadata()

        description = self.generate_response(f"Please generate a YouTube Video Description for the following script: {self.script}. Only return the description, nothing else.")
        
        self.metadata = {
            "title": title,
            "description": description
        }
        success(f"Generated metadata: {self.metadata}, Title length: {len(title)} characters, Description length: {len(description)} characters")

        return self.metadata
    
    def generate_prompts(self) -> List[str]:
        info("Generating image prompts for YouTube video")
        n_prompts = 3
        info(f"Number of prompts requested: {n_prompts}")

        prompt = f"""
        Generate {n_prompts} Image Prompts for AI Image Generation,
        depending on the subject of a video.
        Subject: {self.subject}

        The image prompts are to be returned as
        a JSON-Array of strings.

        Each search term should consist of a full sentence,
        always add the main subject of the video.

        Be emotional and use interesting adjectives to make the
        Image Prompt as detailed as possible.
        
        YOU MUST ONLY RETURN THE JSON-ARRAY OF STRINGS.
        YOU MUST NOT RETURN ANYTHING ELSE. 
        YOU MUST NOT RETURN THE SCRIPT.
        
        The search terms must be related to the subject of the video.
        Here is an example of a JSON-Array of strings:
        ["image prompt 1", "image prompt 2", "image prompt 3"]

        For context, here is the full text:
        {self.script}
        """
        completion = str(self.generate_response(prompt, model=parse_model(get_image_prompt_llm())))

        image_prompts = []

        if "image_prompts" in completion:
            image_prompts = json.loads(completion)["image_prompts"]
        else:
            image_prompts = json.loads(completion)
            info(f"Generated Image Prompts: {image_prompts}")

        self.image_prompts = image_prompts

        if len(image_prompts) > n_prompts:
            image_prompts = image_prompts[:n_prompts]

        success(f"Generated {len(image_prompts)} Image Prompts.")

        return image_prompts