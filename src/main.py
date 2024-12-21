import schedule
import subprocess

from art import *
from cache import *
from utils import *
from config import *
from status import *
from uuid import uuid4
from constants import *
from termcolor import colored
from text_generator import TextGenerator
from image_generator import ImageGenerator
from speech_generator import SpeechGenerator
from subtitle_generator import SubtitleGenerator
from video_generator import VideoGenerator
from prettytable import PrettyTable

def main():
    info("Starting main function")

    valid_input = False
    while not valid_input:
        try:
            info("Displaying options to user")
            info("\n============ Main Menu ============", False)

            for idx, option in enumerate(MAIN_MENU):
                print(colored(f" {idx + 1}. {option}", "cyan"))

            info("=================================\n", False)
            user_input = input("Select an option: ").strip()
            if user_input == '':
                warning("Empty input received")
                print("\n" * 100)
                raise ValueError("Empty input is not allowed.")
            user_input = int(user_input)
            valid_input = True
            info(f"User selected option: {user_input}")
        except ValueError as e:
            error(f"Invalid input: {e}")
            print("\n" * 100)

    if user_input == 1:
        info("Starting Shorts Automater...")

        cached_accounts = get_accounts("youtube")
        info(f"Retrieved {len(cached_accounts)} cached Shorts accounts")

        if len(cached_accounts) == 0:
            warning("No accounts found in cache. Prompting to create one.")
            user_input = question("y/n: ")

            if user_input.lower() == "y":
                generated_uuid = str(uuid4())
                info(f"Generated new UUID: {generated_uuid}")

                success(f" => Generated ID: {generated_uuid}")
                name = question(" => Enter a name for this account: ")
                profile_path = question(" => Enter the path to the Firefox profile: ")
                niche = question(" => Enter the account niche: ")
                language = question(" => Enter the account language: ")

                add_account("youtube", {
                    "id": generated_uuid,
                    "name": name,
                    "profile_path": profile_path,
                    "niche": niche,
                    "language": language,
                    "videos": []
                })
                success(f"Added new Shorts account: {name}")
        else:
            info("Displaying cached Shorts accounts")
            table = PrettyTable()
            table.field_names = ["ID", "UUID", "name", "Niche"]

            for account in cached_accounts:
                table.add_row([cached_accounts.index(account) + 1, colored(account["id"], "cyan"), colored(account["name"], "blue"), colored(account["niche"], "green")])

            print(table)

            user_input = question("Select an account to start: ")

            selected_account = None

            for account in cached_accounts:
                if str(cached_accounts.index(account) + 1) == user_input:
                    selected_account = account

            if selected_account is None:
                error("Invalid account selected. Restarting main function.")
                main()
            else:
                info(f"Selected YouTube account: {selected_account['name']}")
                text_gen = TextGenerator(selected_account["niche"], selected_account["language"])
                image_gen = ImageGenerator()
                speech_gen = SpeechGenerator()
                video_gen = VideoGenerator([], "")

                while True:
                    rem_temp_files()
                    info("Removed temporary files")
                    info("\n============ SHORTS MENU ============", False)

                    for idx, youtube_option in enumerate(SHORTS_MENU):
                        print(colored(f" {idx + 1}. {youtube_option}", "cyan"))

                    info("=================================\n", False)

                    user_input = int(question("Select an option: "))
                    info(f"User selected option: {user_input}")

                    if user_input == 1:
                        info("Generating Short video")
                        
                        # Generate topic, script, and metadata
                        text_gen.generate_topic()
                        text_gen.generate_script()
                        metadata = text_gen.generate_metadata()
                        
                        # Generate image prompts
                        image_prompts = text_gen.generate_prompts()
                        
                        # Generate images
                        images = [image_gen.generate_image(prompt) for prompt in image_prompts]
                        
                        # Generate speech
                        tts_path = speech_gen.generate_speech(text_gen.script)
                        
                        # Combine all elements into final video
                        video_gen.images = images
                        video_gen.tts_path = tts_path
                        video_path = video_gen.combine()
                        
                        # Save metadata and video to active_folder
                        video_gen.save_metadata(metadata, video_path, text_gen.subject)
                        
                        upload_to_yt = question("Do you want to upload this video to YouTube? (y/n): ")
                        if upload_to_yt.lower() == "y":
                            info("Uploading video to YouTube")
                            uploader = Uploader(selected_account["profile_path"])
                            uploader.upload(video_path, metadata["title"], metadata["description"])
                    elif user_input == 2:
                        info("Retrieving Short videos")
                        videos = YouTube.get_videos(selected_account["id"])

                        if len(videos) > 0:
                            info(f"Displaying {len(videos)} videos")
                            videos_table = PrettyTable()
                            videos_table.field_names = ["ID", "Date", "Title"]

                            for video in videos:
                                videos_table.add_row([
                                    videos.index(video) + 1,
                                    colored(video["date"], "blue"),
                                    colored(video["title"][:60] + "...", "green")
                                ])

                            print(videos_table)
                        else:
                            warning("No videos found.")
                    elif user_input == 3:
                        info("Setting up CRON job for Shorts uploads")
                        info("How often do you want to upload?")

                        info("\n============ OPTIONS ============", False)
                        for idx, cron_option in enumerate(CRON_MENU):
                            print(colored(f" {idx + 1}. {cron_option}", "cyan"))

                        info("=================================\n", False)

                        user_input = int(question("Select an Option: "))

                        cron_script_path = os.path.join(ROOT_DIR, "src", "cron.py")
                        command = f"python {cron_script_path} youtube {selected_account['id']}"

                        def job():
                            info("Executing CRON job for Shorts upload")
                            subprocess.run(command)

                        if user_input == 1:
                            info("Setting up daily upload")
                            schedule.every(1).day.do(job)
                            success("Set up CRON Job for daily upload.")
                        elif user_input == 2:
                            info("Setting up twice daily upload")
                            schedule.every().day.at("10:00").do(job)
                            schedule.every().day.at("16:00").do(job)
                            success("Set up CRON Job for twice daily upload.")
                        else:
                            info("Returning to main menu")
                            break
                    elif user_input == 4:
                        if get_verbose():
                            info(" => Climbing Options Ladder...", False)
                        info("Returning to main menu")
                        break

    elif user_input == 2:
        info("Starting Twitter Bot...")
        # ... (existing code for Twitter bot)
        pass

    elif user_input == 3:
        info("Starting Affiliate Marketing...")
        # ... (existing code for Affiliate Marketing)
        pass

    elif user_input == 4:
        info("Starting Outreach...")
        # ... (existing code for Outreach)
        pass

    elif user_input == 5:
        if get_verbose():
            info(" => Quitting...", False)
        info("Exiting application")
        sys.exit(0)
    else:
        error("Invalid option selected. Restarting main function.")
        main()

if __name__ == "__main__":
    info("Starting application")
    print_banner()
    info("Printed ASCII banner")

    first_time = get_first_time_running()
    info(f"First time running: {first_time}")

    if first_time:
        info("First time setup initiated")
        print(colored("Ah! first time? No worries! Let's get you setup first!", "yellow"))

    info("Setting up file tree")
    assert_folder_structure()

    info("Removing temporary files")
    rem_temp_files()

    info("Fetching MP3 files")
    fetch_Music()

    while True:
        main()
