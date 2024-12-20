import os
import sys
import json

from termcolor import colored

ROOT_DIR = os.path.dirname(sys.path[0])

def assert_folder_structure() -> None:
    """Ensure the .mp folder exists."""
    if not os.path.exists(os.path.join(ROOT_DIR, ".mp")):
        if get_verbose():
            print(colored(f"=> Creating .mp folder at {os.path.join(ROOT_DIR, '.mp')}", "green"))
        os.makedirs(os.path.join(ROOT_DIR, ".mp"))

def get_first_time_running() -> bool:
    """Check if this is the first time the application is running."""
    return not os.path.exists(os.path.join(ROOT_DIR, ".mp"))

# General Configuration
def get_verbose() -> bool:
    """Get the verbose setting."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["verbose"]

def get_browser() -> str:
    """Get the browser setting."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["browser"]

def get_headless() -> bool:
    """Get the headless setting."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["headless"]

def get_profile_path() -> str:
    """Get the browser profile path."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["profile_path"]

def get_threads() -> int:
    """Get the number of threads."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["threads"]

def get_is_for_kids() -> bool:
    """Get the is_for_kids setting."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["is_for_kids"]

# Language and Localization
def get_twitter_language() -> str:
    """Get the Twitter language setting."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["twitter_language"]

# URLs and Paths
def get_zip_url() -> str:
    """Get the zip URL."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["zip_url"]

def get_imagemagick_path() -> str:
    """Get the ImageMagick path."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["imagemagick_path"]

# API Keys
def get_gemini_api_key() -> str:
    """Get the Gemini API key."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["gemini_api_key"]

def get_assemblyai_api_key() -> str:
    """Get the AssemblyAI API key."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["assembly_ai_api_key"]

def get_elevenlabs_api_key() -> str:
    """Get the ElevenLabs API key."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["elevenlabs_api_key"]

def get_openai_api_key() -> str:
    """Get the OpenAI API key."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["openai_api_key"]

def get_stability_api_key() -> str:
    """Get the Stability AI API key."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["stability_api_key"]

def get_segmind_api_key() -> str:
    """Get the Segmind API key."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["segmind_api_key"]

# Text Generation
def get_text_gen() -> str:
    """Get the text generation service."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["text_gen"]

def get_text_gen_model() -> str:
    """Get the text generation model."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["text_gen_model"]

# Image Generation
def get_image_gen() -> str:
    """Get the image generation service."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["image_gen"]

def get_image_model() -> str:
    """Get the image generation model."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["image_model"]

# TTS (Text-to-Speech)
def get_tts_engine() -> str:
    """Get the TTS engine."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["tts_engine"]

def get_tts_voice() -> str:
    """Get the TTS voice."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["tts_voice"]

# Subtitles and Display Settings
def get_subtitles() -> bool:
    """Get the subtitles setting."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["subtitles"] == "true"

def get_color() -> str:
    """Get the text color."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["color"]

def get_highlight_bg() -> bool:
    """Get the highlight background setting."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["highlight_bg"] == "true"

def get_bg_color() -> str:
    """Get the background color."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["bg_color"]

def get_font() -> str:
    """Get the font."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["font"]

def get_font_size() -> int:
    """Get the font size."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return int(json.load(file)["font_size"])

def get_max_chars() -> int:
    """Get the maximum number of characters per line."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return int(json.load(file)["max_chars"])

def get_max_lines() -> int:
    """Get the maximum number of lines for subtitles."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return int(json.load(file)["max_lines"])

def get_stroke_color() -> str:
    """Get the stroke color."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["stroke_color"]

def get_stroke_width() -> int:
    """Get the stroke width."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return int(json.load(file)["stroke_width"])

# Additional Settings
def get_auto_upload() -> str:
    """Get the auto upload setting."""
    with open(os.path.join(ROOT_DIR, "config.json"), "r") as file:
        return json.load(file)["auto_upload"]
