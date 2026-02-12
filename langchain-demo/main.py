import os
import sys

from dotenv import load_dotenv

from sandbox import pkg_help, playwright_sandbox_help

load_dotenv()


def main():
    print("python version:", sys.version)
    print("python root:", os.getenv("PYROOT", "unknown"))
    print("python project home:", os.getenv("PYHOME", "unknown"))

    print()
    pkg_help()
    playwright_sandbox_help()


if __name__ == "__main__":
    main()
