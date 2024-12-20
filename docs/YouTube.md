# YouTube Shorts Automater

MPV2 uses a similar implementation of V1 (see [MPV1](https://github.com/FujiwaraChoki/MoneyPrinter)), to generate Video-Files and upload them to YouTube Shorts.

In contrast to V1, V2 uses AI generated images as the visuals for the video, instead of using stock footage. This makes the videos more unique and less likely to be flagged by YouTube. V2 also supports music right from the get-go.

## Relevant Configuration

In your `config.json`, you need the following attributes filled out, so that the bot can function correctly.

```json
{
  "firefox_profile": "The path to your Firefox profile (used to log in to YouTube)",
  "headless": true,
  "llm": "The Large Language Model you want to use to generate the video script.",
  "image_model": "What AI Model you want to use to generate images.",
  "threads": 4,
  "is_for_kids": true
}
```

After Accounts Created and 2 Videos uploaded! then the workflow file Example:
{
    "accounts": [
        {
            "id": "fde05e6d-327a-46a9-9648-cd83fba2f3e5",
            "name": "Tech Tips Channel",
            "profile_path": "/home/username/.mozilla/firefox/09aycyzs.default-release",
            "niche": "TECHNOLOGY",
            "language": "English",
            "videos": [
                {
                    "title": "5 Hidden iPhone Features You Need to Know #tech #iphone #tips",
                    "description": "Discover amazing hidden features on your iPhone that will make your life easier. These pro tips will help you master your device.",
                    "url": "https://www.youtube.com/watch?v=video_id_1",
                    "date": "2024-01-10 14:30:25"
                },
                {
                    "title": "Best Tech Gadgets Under $50 in 2024 #technology #gadgets #budget",
                    "description": "Looking for affordable tech gadgets? Here are the best budget-friendly tech accessories you can buy right now.",
                    "url": "https://www.youtube.com/watch?v=video_id_2",
                    "date": "2024-01-11 16:45:12"
                }
            ]
        },
        {
            "id": "abc123de-456f-789g-hij0-klmnopqrstuv",
            "name": "Fitness Motivation",
            "profile_path": "/home/username/.mozilla/firefox/xyz789.default-release",
            "niche": "FITNESS",
            "language": "English",
            "videos": [
                {
                    "title": "10-Minute Ab Workout for Beginners #fitness #abs #workout",
                    "description": "Quick and effective ab workout that you can do at home. Perfect for beginners looking to start their fitness journey.",
                    "url": "https://www.youtube.com/watch?v=video_id_3",
                    "date": "2024-01-12 09:15:30"
                },
                {
                    "title": "5 Best Exercises for Building Muscle Fast #gym #fitness #muscle",
                    "description": "Learn the most effective exercises to build muscle and strength. These compound movements will transform your physique.",
                    "url": "https://www.youtube.com/watch?v=video_id_4",
                    "date": "2024-01-13 11:20:45"
                }
            ]
        }
    ]
}

## Roadmap

Here are some features that are planned for the future:

- [ ] Subtitles (using either AssemblyAI or locally assembling them)
