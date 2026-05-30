from dotenv import load_dotenv

from utils import init_logger

load_dotenv()


# playwright chromium env:
# uv run playwright install chromium


def main():
    logger = init_logger()
    logger.info("demo: ui automation")


if __name__ == "__main__":
    main()
