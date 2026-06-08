# coding: utf-8

from dotenv import load_dotenv

from utils import get_logger

load_dotenv()


# playwright chromium env:
# uv run playwright install chromium


def main():
    logger = get_logger()
    logger.info("demo: ui automation")


if __name__ == "__main__":
    main()
