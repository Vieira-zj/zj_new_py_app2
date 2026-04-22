from .cli import cli
from .io_file import IOFile

__all__ = ["IOFile", "pkg_help"]


def pkg_help():
    print("hello from py-pkg-demo")
