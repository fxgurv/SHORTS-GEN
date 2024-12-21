from moviepy.editor import *
import os
import shutil
from utils import *
from config import *
from status import *
from uuid import uuid4
from moviepy.video.fx.all import crop
from moviepy.audio.fx.all import volumex
from moviepy.config import change_settings
from subtitle_generator import SubtitleGenerator
from moviepy.video.tools.subtitles import SubtitlesClip

# Set ImageMagick Path
change_settings({"IMAGEMAGICK_BINARY": get_imagemagick_path()})


class VideoGenerator:
    def __init__(self, images: list, tts_path: str):
        self.images = images
        self.tts_path = tts_path

    def combine(self) -> str:
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

                aspect_ratio = 9/16
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
        
        word_highlighted_clips = SubtitleGenerator().generate_subtitles(self.tts_path)
        
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

    def save_metadata(self, metadata: dict, video_path: str, subject: str):
        info("Creating active_folder and saving metadata")
        videos_dir = os.path.join(ROOT_DIR, "Videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        existing_folders = [f for f in os.listdir(videos_dir) if os.path.isdir(os.path.join(videos_dir, f))]
        last_number = max([int(f.split('.')[0]) for f in existing_folders if f.split('.')[0].isdigit()] or [0])
        
        new_folder_number = last_number + 1
        sanitized_subject = ''.join(c for c in subject if c.isalnum() or c.isspace())
        folder_name = f"{new_folder_number}. {sanitized_subject}"
        
        active_folder = os.path.join(videos_dir, folder_name)
        os.makedirs(active_folder, exist_ok=True)

        metadata_file = os.path.join(active_folder, "metadata.txt")
        with open(metadata_file, "w") as f:
            f.write(f"Title: {metadata['title']}\n")
            f.write(f"Description: {metadata['description']}")

        shutil.copy2(video_path, os.path.join(active_folder, os.path.basename(video_path)))
        success(f"Metadata and video saved to: {active_folder}")