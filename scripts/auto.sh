#!/usr/bin/env bash

# Script to generate & Upload videos to YT Shorts automatically (3 times per day)

# Normalize line endings
sed -i.bak 's/\r$//' "$0" && rm "$0.bak"

# Check which interpreter to use (python)
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "Python not found. Please install Python 3."
  exit 1
fi

while true; do
    # Read .mp/youtube.json file and get all account IDs
    youtube_ids=$($PYTHON -c "import json; print('\n'.join([account['id'] for account in json.load(open('.mp/youtube.json'))['accounts']]))")

    # Convert string of IDs to array
    IFS=$'\n' read -d '' -r -a id_array <<< "$youtube_ids"

    echo "Starting automated video generation and upload for ${#id_array[@]} accounts..."

    # Process each account
    for id in "${id_array[@]}"; do
        echo "Processing account ID: $id"
        $PYTHON src/cron.py youtube "$id"
        echo "Completed processing for account ID: $id"
    done

    echo "All accounts processed. Waiting 8 hours until next upload..."
    
    # 8-hour countdown (28800 seconds)
    for ((i=28800; i>0; i--)); do
        hours=$((i/3600))
        minutes=$(((i%3600)/60))
        seconds=$((i%60))
        printf "\rNext upload in: %02d:%02d:%02d" $hours $minutes $seconds
        sleep 1
    done
    echo -e "\nRestarting process..."
done
