from termcolor import colored

def error(message: str, show_emoji: bool = True) -> None:
    emoji = "[ERROR]" if show_emoji else ""
    print(colored(f"{emoji} {message}", "white"))

def success(message: str, show_emoji: bool = True) -> None:
    emoji = "[SUCCESS]" if show_emoji else ""
    print(colored(f"{emoji} {message}", "green"))

def info(message: str, show_emoji: bool = True) -> None:
    emoji = "[INFO]" if show_emoji else ""
    print(colored(f"{emoji} {message}", "white"))

def warning(message: str, show_emoji: bool = True) -> None:
    emoji = "[WARNING]" if show_emoji else ""
    print(colored(f"{emoji} {message}", "yellow"))

def question(message: str, show_emoji: bool = True) -> str:
    emoji = "[QUERY]" if show_emoji else ""
    return input(colored(f"{emoji} {message}", "magenta"))
