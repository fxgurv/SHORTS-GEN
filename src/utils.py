import os
import random
import zipfile
import requests
import platform
from status import *
from config import *

def close_running_selenium_instances() -> None:
    try:
        info(" => Closing running Selenium instances...")

        if platform.system() == "Windows":
            os.system("taskkill /f /im firefox.exe")
        else:
            os.system("pkill firefox")

        success(" => Closed running Selenium instances.")

    except Exception as e:
        error(f"Error occurred while closing running Selenium instances: {str(e)}")

def build_url(youtube_video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={youtube_video_id}"

def rem_temp_files() -> None:
    mp_dir = os.path.join(ROOT_DIR, ".mp")
    files = os.listdir(mp_dir)

    for file in files:
        if not file.endswith(".json"):
            os.remove(os.path.join(mp_dir, file))

def fetch_Music() -> None:
    try:
        info(f" => Fetching Music...")

        files_dir = os.path.join(ROOT_DIR, "Music")
        if not os.path.exists(files_dir):
            os.mkdir(files_dir)
            if get_verbose():
                info(f" => Created directory: {files_dir}")
        else:
            return

        response = requests.get(get_zip_url() or "https://huggingface.co/AZILS/BG/resolve/main/BG.zip?download=true")

        with open(os.path.join(files_dir, "Music.zip"), "wb") as file:
            file.write(response.content)

        with zipfile.ZipFile(os.path.join(files_dir, "Music.zip"), "r") as file:
            file.extractall(files_dir)

        os.remove(os.path.join(files_dir, "Music.zip"))

        success(" => Downloaded Music to ../Music.")

    except Exception as e:
        error(f"Error occurred while fetching Music: {str(e)}")

def choose_random_music() -> str:
    try:
        music_files = os.listdir(os.path.join(ROOT_DIR, "Music"))
        chosen_music = random.choice(music_files)
        success(f"Successfully chose random background Music: {chosen_music}")
        return os.path.join(ROOT_DIR, "Music", chosen_music)
    except Exception as e:
        error(f"Error occurred while choosing random Music: {str(e)}")


    def add_video(self, video: dict) -> None:
        """
        Adds a video to the cache.

        Args:
            video (dict): The video to add

        Returns:
            None
        """
        videos = self.get_videos()
        videos.append(video)

        cache = get_youtube_cache_path()

        with open(cache, "r") as file:
            previous_json = json.loads(file.read())
            
            # Find our account
            accounts = previous_json["accounts"]
            for account in accounts:
                if account["id"] == self._account_uuid:
                    account["videos"].append(video)
            
            # Commit changes
            with open(cache, "w") as f:
                f.write(json.dumps(previous_json))



    def get_videos(self) -> List[dict]:
        """
        Gets the uploaded videos from the YouTube Channel.

        Returns:
            videos (List[dict]): The uploaded videos.
        """
        if not os.path.exists(get_youtube_cache_path()):
            # Create the cache file
            with open(get_youtube_cache_path(), 'w') as file:
                json.dump({
                    "videos": []
                }, file, indent=4)
            return []

        videos = []
        # Read the cache file
        with open(get_youtube_cache_path(), 'r') as file:
            previous_json = json.loads(file.read())
            # Find our account
            accounts = previous_json["accounts"]
            for account in accounts:
                if account["id"] == self._account_uuid:
                    videos = account["videos"]

        return videos
