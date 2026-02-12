# -- coding: utf-8 --
import argparse
import os
import signal
import sys

from dotenv import load_dotenv

load_dotenv()


def init_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="py base demo.")

    parser.add_argument("-m", "--mode", type=str, default="main", help="run mode")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose output"
    )
    return parser.parse_args()


def signal_handler(signum, frame):
    _ = frame
    print(f"\n\nget interrupt system signal: sign_num={signum}")
    sys.exit(0)


def chat():
    # if set, process by signal_handler instead of interrupt exception
    signal.signal(signal.SIGINT, signal_handler)

    user_input: str = ""
    try:
        while 1:
            user_input = input("pls input:").strip()
            print("your input:", user_input)
            if user_input.lower() in ("exit", "quit"):
                print("bye")
                return
    except KeyboardInterrupt:
        print("\n\nget system signal ctrl+c")
    except EOFError:
        print("\n\nget system signal ctrl+d")
    finally:
        print("clearup")


def main():
    print("python root:", os.getenv("PYROOT", "unknown"))
    print("python project home:", os.getenv("PYHOME", "unknown"))
    print(f"run env: {os.getenv("ENV", "unknown")}")


# run cli:
# uv run main.py -v
# uv run main.py -m chat

if __name__ == "__main__":
    args = init_args()
    if args.verbose:
        print("verbose mode enabled")
        print(f"run mode: {args.mode}")
        print()

    match args.mode:
        case "main":
            main()
        case "chat":
            chat()
        case _:
            print("unknown run mode")

    print("python demo finished")
