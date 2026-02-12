import os

from sandbox.play_wright.hello_world import playwright_sandbox_help


def pkg_help():
    print("project home:", os.getenv("PYHOME", "null"))
    print("this is a langchain browser sandbox package.")


print("loaded sandbox package.\n")
