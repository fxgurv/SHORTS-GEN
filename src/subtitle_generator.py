import os
import json
from utils import *
from config import *
from status import *
import assemblyai as aai
from moviepy.editor import TextClip
from uuid import uuid4  # Import uuid4 for generating unique IDs


FONT = 'Helvetica-Bold'
FONTSIZE = 80
COLOR = 'white'
BG_COLOR = 'blue'
FRAME_SIZE = (1080, 1920)
MAX_CHARS = 30
MAX_DURATION = 3.0
MAX_GAP = 2.5


class SubtitleGenerator:
    def __init__(self):
        pass

    def generate_subtitles(self, audio_path: str):
        # Log the start of the subtitle generation process
        info("Starting subtitle generation process")

        # Set the API key for AssemblyAI and configure transcription settings
        aai.settings.api_key = get_assemblyai_api_key()
        config = aai.TranscriptionConfig(speaker_labels=False, word_boost=[], format_text=True)
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_path)

        # Extract word-level information from the transcript
        wordlevel_info = []
        for word in transcript.words:
            word_data = {
                "word": word.text.strip(),
                "start": word.start / 1000.0,
                "end": word.end / 1000.0
            }
            wordlevel_info.append(word_data)

        # Save the word-level information to a JSON file
        wordlevel_json = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}_wordlevel.json")
        with open(wordlevel_json, 'w') as f:
            json.dump(wordlevel_info, f, indent=4)
        info(f"Word-level JSON saved to: {wordlevel_json}")

        # Initialize lists to hold subtitles and lines
        subtitles = []
        line = []
        line_duration = 0

        # Process each word to form lines based on time, character count, and gaps
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

            # If any condition is met, finalize the current line and start a new one
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

        # Add any remaining words as a final subtitle line
        if line:
            subtitle_line = {
                "text": " ".join(item["word"] for item in line),
                "start": line[0]["start"],
                "end": line[-1]["end"],
                "words": line
            }
            subtitles.append(subtitle_line)

        # Save the line-level information to a JSON file
        linelevel_json = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}_linelevel.json")
        with open(linelevel_json, 'w') as f:
            json.dump([line for line in subtitles], f, indent=4)
        info(f"Line-level JSON saved to: {linelevel_json}")

        # Create subtitle clips using MoviePy
        all_subtitle_clips = []

        for subtitle in subtitles:
            full_duration = subtitle['end'] - subtitle['start']

            word_clips = []
            xy_textclips_positions = []

            frame_width, frame_height = FRAME_SIZE
            x_pos = 0
            y_pos = frame_height * 0.85
            x_buffer = frame_width * 1 / 10
            y_buffer = 10

            line_height = 0
            current_line_height = 0

            # Create text clips for each word in the subtitle line
            for index, wordJSON in enumerate(subtitle['words']):
                duration = wordJSON['end'] - wordJSON['start']
                word_clip = TextClip(wordJSON['word'], font=FONT, fontsize=FONTSIZE, color=COLOR).set_start(subtitle['start']).set_duration(full_duration)
                word_width, word_height = word_clip.size

                line_height = max(line_height, word_height)

                # Handle line wrapping if the word exceeds the frame width
                if x_pos + word_width > frame_width - 2 * x_buffer:
                    x_pos = 0
                    y_pos += line_height + y_buffer
                    current_line_height += line_height + y_buffer

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

                word_clip = word_clip.set_position((x_pos + x_buffer, y_pos + y_buffer))
                word_clips.append(word_clip)
                x_pos = x_pos + word_width + 10

            # Create highlighted text clips for each word
            for highlight_word in xy_textclips_positions:
                word_clip_highlight = TextClip(highlight_word['word'], font=FONT, fontsize=FONTSIZE, color=COLOR, bg_color=BG_COLOR).set_start(highlight_word['start']).set_duration(highlight_word['duration'])
                word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
                word_clips.append(word_clip_highlight)

            all_subtitle_clips.extend(word_clips)

        # Log the number of subtitle clips generated
        info(f"Generated {len(all_subtitle_clips)} subtitle clips")
        return all_subtitle_clips