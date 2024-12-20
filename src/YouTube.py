import re
import g4f
import json
import time
import shutil
import asyncio
import requests
import string
from utils import *
from cache import *
from config import *
from status import *
from uuid import uuid4
from constants import *
from typing import List
import assemblyai as aai
from moviepy.editor import *
from uploader import Uploader
from datetime import datetime
from termcolor import colored
from moviepy.video.fx.all import crop
from moviepy.audio.fx.all import volumex
from moviepy.config import change_settings
from moviepy.video.tools.subtitles import SubtitlesClip

# Set ImageMagick Path
change_settings({"IMAGEMAGICK_BINARY": get_imagemagick_path()})

# Constants
FONT = 'Helvetica-Bold'
FONTSIZE = 80
COLOR = 'white'
BG_COLOR = 'blue'
FRAME_SIZE = (1080, 1920)
MAX_CHARS = 30
MAX_DURATION = 3.0
MAX_GAP = 2.5

class YouTube:
    def __init__(self, account_uuid: str, account_name: str, profile_path: str, niche: str, language: str) -> None:
        info(f"Initializing YouTube class for account: {account_name}")
        self._account_uuid: str = account_uuid
        self._account_name: str = account_name
        self._profile_path: str = profile_path
        self._niche: str = niche
        self._language: str = language
        self.images = []
        self.uploader = Uploader(profile_path)
        info(f"Niche: {niche}, Language: {language}")

    @property
    def niche(self) -> str:
        return self._niche
    
    @property
    def language(self) -> str:
        return self._language
    
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

    def generate_image(self, prompt: str) -> str:
        image_gen = get_image_gen()
        image_model = get_image_model()
        
        info(f"Using: {image_gen} With Model: {image_model}")
        if image_gen == "prodia":
            info("Using Prodia provider for image generation")
            s = requests.Session()
            headers = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            # Generate job
            info("Sending generation request to Prodia API")
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
                    info("Image generation successful, downloading result")
                    img_data = s.get(f"https://images.prodia.xyz/{job_id}.png?download=1", headers=headers).content
                    image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
                    with open(image_path, "wb") as f:
                        f.write(img_data)
                    self.images.append(image_path)
                    success(f"Image saved to: {image_path}")
                    return image_path

        elif image_gen == "hercai":
            info("Using Hercai provider for image generation")
            url = f"https://hercai.onrender.com/{image_model}/text2image?prompt={prompt}"
            r = requests.get(url)
            parsed = r.json()

            if "url" in parsed and parsed["url"]:
                info("Image URL received from Hercai")
                image_url = parsed["url"]
                image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
                with open(image_path, "wb") as f:
                    image_data = requests.get(image_url).content
                    f.write(image_data)
                self.images.append(image_path)
                success(f"Image saved to: {image_path}")
                return image_path
            else:
                warning("No image URL in Hercai response")

        elif image_gen == "pollinations":
            info("Using Pollinations provider for image generation")
            response = requests.get(f"https://image.pollinations.ai/prompt/{prompt}{random.randint(1,10000)}")
            if response.status_code == 200:
                info("Image received from Pollinations")
                image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
                with open(image_path, "wb") as f:
                    f.write(response.content)
                self.images.append(image_path)
                success(f"Image saved to: {image_path}")
                return image_path
            else:
                warning(f"Pollinations request failed with status code: {response.status_code}")

        warning(f"Image generation with {image_gen} was unsuccessful")
        return None

    def generate_speech(self, text: str, output_format: str = 'mp3') -> str:
        """Generate speech using multiple TTS engines"""
        info("Generating speech from text")
        
        # Clean text
        text = re.sub(r'[^\w\s.?!]', '', text)
        
        tts_engine = get_tts_engine()
        tts_voice = get_tts_voice()
        info(f"Using TTS Engine: {tts_engine}, Voice: {tts_voice}")
        
        audio_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.{output_format}")
        
        if tts_engine == "elevenlabs":
            # Latest ElevenLabs API implementation

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
                url = "https://api.elevenlabs.io/v1/text-to-speech/{tts_voice}",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
                success(f"Speech generated successfully using ElevenLabs at {audio_path}")
                self.tts_path = audio_path
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

    def generate_subtitles(self, audio_path: str):
        """Generate word-highlighted subtitles for the video."""
        info("Starting subtitle generation process")

        try:
            # Transcribe audio to get word-level information
            aai.settings.api_key = get_assemblyai_api_key()
            config = aai.TranscriptionConfig(speaker_labels=False, word_boost=[], format_text=True)
            transcriber = aai.Transcriber(config=config)
            transcript = transcriber.transcribe(audio_path)

            # Process word-level information
            wordlevel_info = []
            for word in transcript.words:
                word_data = {
                    "word": word.text.strip(),
                    "start": word.start / 1000.0,
                    "end": word.end / 1000.0
                }
                wordlevel_info.append(word_data)

            # Save word-level JSON
            wordlevel_json = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}_wordlevel.json")
            with open(wordlevel_json, 'w') as f:
                json.dump(wordlevel_info, f, indent=4)
            info(f"Word-level JSON saved to: {wordlevel_json}")

            # Split text into lines based on character count, duration, and gap
            subtitles = []
            line = []
            line_duration = 0

            for idx, word_data in enumerate(wordlevel_info):
                word = word_data["word"]
                start = word_data["start"]
                end = word_data["end"]

                line.append(word_data)
                line_duration += end - start
                temp = " ".join(item["word"] for item in line)
                new_line_chars = len(temp)
                duration_exceeded = line_duration > MAX_DURATION
                chars_exceeded = new_line_chars > MAX_CHARS
                if idx > 0:
                    gap = word_data['start'] - wordlevel_info[idx - 1]['end']
                    maxgap_exceeded = gap > MAX_GAP
                else:
                    maxgap_exceeded = False

                # Check if any condition is exceeded to finalize the current line
                if duration_exceeded or chars_exceeded or maxgap_exceeded:
                    if line:
                        subtitle_line = {
                            "text": " ".join(item["word"] for item in line),
                            "start": line[0]["start"],
                            "end": line[-1]["end"],
                            "words": line
                        }
                        subtitles.append(subtitle_line)
                        line = []
                        line_duration = 0

            # Add the remaining words as the last subtitle line if any
            if line:
                subtitle_line = {
                    "text": " ".join(item["word"] for item in line),
                    "start": line[0]["start"],
                    "end": line[-1]["end"],
                    "words": line
                }
                subtitles.append(subtitle_line)

            # Save line-level JSON
            linelevel_json = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}_linelevel.json")
            with open(linelevel_json, 'w') as f:
                json.dump([line for line in subtitles], f, indent=4)
            info(f"Line-level JSON saved to: {linelevel_json}")

            # Create subtitle clips
            all_subtitle_clips = []

            for subtitle in subtitles:
                full_duration = subtitle['end'] - subtitle['start']

                word_clips = []
                xy_textclips_positions = []

                # Dynamic vertical positioning (moved to bottom)
                frame_width, frame_height = FRAME_SIZE
                x_pos = 0
                y_pos = frame_height * 0.85  # Position at 85% of frame height
                x_buffer = frame_width * 1 / 10
                y_buffer = 10  # Small vertical buffer

                line_height = 0
                current_line_height = 0

                for index, wordJSON in enumerate(subtitle['words']):
                    duration = wordJSON['end'] - wordJSON['start']
                    word_clip = TextClip(wordJSON['word'], font=FONT, fontsize=FONTSIZE, color=COLOR).set_start(subtitle['start']).set_duration(full_duration)
                    word_width, word_height = word_clip.size

                    # Track line height for multi-line support
                    line_height = max(line_height, word_height)

                    # Check if the current word exceeds the frame width, move to the next line
                    if x_pos + word_width > frame_width - 2 * x_buffer:
                        x_pos = 0
                        y_pos += line_height + y_buffer
                        current_line_height += line_height + y_buffer

                    # Store the position and other details for highlighting
                    xy_textclips_positions.append({
                        "x_pos": x_pos + x_buffer,
                        "y_pos": y_pos + y_buffer,
                        "width": word_width,
                        "height": word_height,
                        "word": wordJSON['word'],
                        "start": wordJSON['start'],
                        "end": wordJSON['end'],
                        "duration": duration
                    })

                    # Set the position of the word clip
                    word_clip = word_clip.set_position((x_pos + x_buffer, y_pos + y_buffer))
                    word_clips.append(word_clip)
                    x_pos = x_pos + word_width + 10

                # Create highlighted word clips
                for highlight_word in xy_textclips_positions:
                    word_clip_highlight = TextClip(highlight_word['word'], font=FONT, fontsize=FONTSIZE, color=COLOR, bg_color=BG_COLOR).set_start(highlight_word['start']).set_duration(highlight_word['duration'])
                    word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
                    word_clips.append(word_clip_highlight)

                # Add all word clips to the list of subtitle clips
                all_subtitle_clips.extend(word_clips)

            info(f"Generated {len(all_subtitle_clips)} subtitle clips")
            return all_subtitle_clips

        except Exception as e:
            error(f"Subtitle generation failed: {str(e)}")
            return []

    def combine(self) -> str:
        """Combine all elements into final video."""
        info("Starting to combine all elements into the final video")
        combined_image_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.mp4")
        threads = get_threads()
        
        tts_clip = AudioFileClip(self.tts_path)
        max_duration = tts_clip.duration
        req_dur = max_duration / len(self.images)
        
        clips = []
        tot_dur = 0
        while tot_dur < max_duration:
            for image_path in self.images:
                clip = ImageClip(image_path)
                clip.duration = req_dur
                clip = clip.set_fps(30)

                # Intelligent cropping for different aspect ratios
                aspect_ratio = 9/16  # Standard vertical video ratio
                if clip.w / clip.h < aspect_ratio:
                    clip = crop(
                        clip, 
                        width=clip.w, 
                        height=round(clip.w / aspect_ratio), 
                        x_center=clip.w / 2, 
                        y_center=clip.h / 2
                    )
                else:
                    clip = crop(
                        clip, 
                        width=round(aspect_ratio * clip.h), 
                        height=clip.h, 
                        x_center=clip.w / 2, 
                        y_center=clip.h / 2
                    )

                clip = clip.resize((1080, 1920))

                clips.append(clip)
                tot_dur += clip.duration

        final_clip = concatenate_videoclips(clips)
        final_clip = final_clip.set_fps(30)
        
        random_Music = choose_random_music()
        random_Music_clip = AudioFileClip(random_Music)
        random_Music_clip = random_Music_clip.fx(volumex, 0.1)
        random_Music_clip = random_Music_clip.set_duration(max_duration)
        
        word_highlighted_clips = self.generate_subtitles(self.tts_path)
        
        comp_audio = CompositeAudioClip([
            tts_clip,
            random_Music_clip
        ])

        final_clip = final_clip.set_audio(comp_audio)
        final_clip = final_clip.set_duration(tts_clip.duration)

        final_clip = CompositeVideoClip([
            final_clip
        ] + word_highlighted_clips)

        final_clip.write_videofile(combined_image_path, threads=threads)

        success(f"Video successfully created at: {combined_image_path}")
        return combined_image_path

    def save_metadata(self):
        """Save metadata and copy video to output directory"""
        info("Creating active_folder and saving metadata")
        videos_dir = os.path.join(ROOT_DIR, "Videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        existing_folders = [f for f in os.listdir(videos_dir) if os.path.isdir(os.path.join(videos_dir, f))]
        last_number = max([int(f.split('.')[0]) for f in existing_folders if f.split('.')[0].isdigit()] or [0])
        
        new_folder_number = last_number + 1
        sanitized_subject = ''.join(c for c in self.subject if c.isalnum() or c.isspace())
        folder_name = f"{new_folder_number}. {sanitized_subject}"
        
        active_folder = os.path.join(videos_dir, folder_name)
        os.makedirs(active_folder, exist_ok=True)

        metadata_file = os.path.join(active_folder, "metadata.txt")
        with open(metadata_file, "w") as f:
            f.write(f"Title: {self.metadata['title']}\n")
            f.write(f"Description: {self.metadata['description']}")

        shutil.copy2(self.video_path, os.path.join(active_folder, os.path.basename(self.video_path)))
        success(f"Metadata and video saved to: {active_folder}")

    def generate_video(self) -> str:
        """Generate complete video with all components"""
        info("Starting video generation process")
        
        info("Generating topic")
        self.generate_topic()
        
        info("Generating script")
        self.generate_script()
        
        info("Generating metadata")
        self.generate_metadata()
        
        info("Generating image prompts")
        self.generate_prompts()
        
        info("Generating images")
        for i, prompt in enumerate(self.image_prompts, 1):
            info(f"Generating image {i}/{len(self.image_prompts)}")
            self.generate_image(prompt)
        
        info("Generating speech")
        self.generate_speech(self.script)
        
        info("Combining all elements into final video")
        path = self.combine()
        
        info(f"Video generation complete. File saved at: {path}")
        self.video_path = os.path.abspath(path)

        info("Saving metadata and video to active_folder")
        self.save_metadata()
        
        return path