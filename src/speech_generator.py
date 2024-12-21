import requests
import os
import asyncio
import re
from utils import *
from config import *
from status import *
from uuid import uuid4  # Import uuid4 for generating unique IDs

class SpeechGenerator:
    def __init__(self):
        pass

    def generate_speech(self, text: str, output_format: str = 'mp3') -> str:
        info("Generating speech from text")
        
        text = re.sub(r'[^\w\s.?!]', '', text)
        
        tts_engine = get_tts_engine()
        tts_voice = get_tts_voice()
        info(f"Using TTS Engine: {tts_engine}, Voice: {tts_voice}")
        
        audio_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.{output_format}")
        
        if tts_engine == "elevenlabs":
            ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": get_elevenlabs_api_key()
            }
            
            payload = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{tts_voice}",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
                success(f"Speech generated successfully using ElevenLabs at {audio_path}")
                return audio_path
            else:
                error(f"ElevenLabs API error: {response.text}")
                return None
                
        elif tts_engine == 'bark':
            from bark import SAMPLE_RATE, generate_audio, preload_models
            preload_models()
            audio_array = generate_audio(text)
            import soundfile as sf
            sf.write(audio_path, audio_array, SAMPLE_RATE)
            
        elif tts_engine == "gtts":
            info("Using Google TTS provider for speech generation")
            from gtts import gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(audio_path)
            
        elif tts_engine == "openai":
            info("Using OpenAI provider for speech generation")
            from openai import OpenAI
            client = OpenAI(api_key=get_openai_api_key())
            response = client.audio.speech.create(
                model="tts-1",
                voice=tts_voice,
                input=text
            )
            response.stream_to_file(audio_path)
            
        elif tts_engine == "edge":
            info("Using Edge TTS provider for speech generation")
            import edge_tts
            import asyncio
            async def generate():
                communicate = edge_tts.Communicate(text, tts_voice)
                await communicate.save(audio_path)
            asyncio.run(generate())
            
        elif tts_engine == "local_tts":
            info("Using Local TTS provider for speech generation")
            import requests
            
            url = "https://imseldrith-tts-openai-free.hf.space/v1/audio/speech"
            
            payload = {
                "model": "tts-1",
                "input": text,
                "voice": tts_voice,
                "response_format": "mp3",
                "speed": 0.60
            }
            
            headers = {
                "accept": "*/*",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                with open(audio_path, "wb") as f:
                    f.write(response.content)
                success(f"Speech generated successfully at {audio_path}")
            else:
                error(f"Failed to generate speech: {response.text}")
                return None
                
        elif tts_engine == "xtts":
            info("Using XTTS-v2 provider for speech generation")
            from TTS.api import TTS
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            tts.tts_to_file(
                text=text,
                file_path=audio_path,
                speaker=tts_voice,
                language="en"
            )
            
        elif tts_engine == "rvc":
            info("Using RVC provider for speech generation")
            from rvc_engine import RVCEngine
            
            # First generate base audio using GTTS
            temp_path = os.path.join(ROOT_DIR, ".mp", f"temp_{uuid4()}.wav")
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_path)
            
            # Convert using RVC
            rvc = RVCEngine(model_path=get_rvc_model_path())
            rvc.convert(
                input_path=temp_path,
                output_path=audio_path,
                f0_method='dio'  # CPU-friendly method
            )
            
            # Cleanup temp file
            os.remove(temp_path)
            
        else:
            error(f"Unsupported TTS engine: {tts_engine}")
            return None
            
        success(f"Speech generated and saved to: {audio_path}")
        self.tts_path = audio_path
        return audio_path
