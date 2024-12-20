# RUN THIS N AMOUNT OF TIMES
import os
import sys

from status import *
from cache import get_accounts
from config import get_verbose
from Twitter import Twitter
from YouTube import YouTube

def main():
    purpose = str(sys.argv[1])
    account_id = str(sys.argv[2])

    verbose = get_verbose()

    if purpose == "twitter":
        accounts = get_accounts("twitter")

        if not account_id:
            error("Account UUID cannot be empty.")

        for acc in accounts:
            if acc["id"] == account_id:
                if verbose:
                    info("Initializing Twitter...")
                twitter = Twitter(
                    acc["id"],
                    acc["name"],
                    acc["profile_path"],
                    acc["topic"]
                )
                twitter.post()
                if verbose:
                    success("Done posting.")
                break
    elif purpose == "youtube":

        accounts = get_accounts("youtube")

        if not account_id:
            error("Account UUID cannot be empty.")

        for acc in accounts:
            if acc["id"] == account_id:
                if verbose:
                    info("Initializing YouTube...")
                youtube = YouTube(
                    acc["id"],
                    acc["name"],
                    acc["profile_path"],
                    acc["niche"],
                    acc["language"]
                )
                youtube.generate_video()
                youtube.upload_video()
                if verbose:
                    success("Uploaded Short.")
                break
    else:
        error("Invalid Purpose, exiting...")
        sys.exit(1)

if __name__ == "__main__":
    main()
