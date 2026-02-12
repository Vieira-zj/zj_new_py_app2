from copy import replace  # available: py >= 3.13
from dataclasses import dataclass
from typing import Iterator, List

from pydantic import SecretStr
from rich.console import Console

# common


def test_call_package_fn():
    from langchain_base import langchain_help

    langchain_help()


def test_secret_str():
    console = Console()
    s = SecretStr("hello")
    console.print(f"mark secret str: {s}")
    console.print(f"raw secret str: [bold red]{s.get_secret_value()}[/bold red]")


# demo: typing


def test_py_typing_list():
    def print_list(values: List[str | int]):
        for val in values:
            if isinstance(val, str):
                print("string:", val)
            elif isinstance(val, int):
                print("number:", val)
            else:
                print("invalid value")

    print_list(
        [
            "a",
            "b",
            1,
            "d",
        ]
    )


def test_py_typing_iter():
    def print_iter(values: Iterator[str | int]):
        for val in values:
            if isinstance(val, str):
                print("string:", val)
            elif isinstance(val, int):
                print("number:", val)
            else:
                print("invalid value")

    print_iter(iter(["a", "b", 1, "d"]))
    print()

    def get_iter(total: int) -> Iterator[str | int]:
        for i in range(1, total):
            if i % 2 == 0:
                yield str(i)
            else:
                yield i

    print_iter(get_iter(6))


# demo: dataclass


@dataclass(frozen=True, kw_only=True)
class Point:
    x: int = 0
    y: int = 0


def test_update_immutable_dataclass():
    p = Point(x=1, y=2)
    p_moved = replace(p, x=10)
    print(f"p={p}, p_moved={p_moved}")


if __name__ == "__main__":
    # test_call_package_fn()
    test_secret_str()

    # test_py_typing_list()
    # test_py_typing_iter()

    # test_update_immutable_dataclass()
