import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from functools import cache, lru_cache

# example: cache


def test_tool_cache():
    @cache
    def add1(x: int, y: int) -> int:
        print(f"calculating {x} + {y}")
        return x + y

    print("cache:")
    for inputs in [(3, 4), (5, 6), (3, 4), (5, 6)]:
        print(add1(*inputs))

    @lru_cache(maxsize=16)
    def add2(x: int, y: int) -> int:
        print(f"calculating {x} + {y}")
        return x + y

    print("\nlru_cache:")
    for inputs in [(3, 4), (5, 6), (3, 4), (5, 6)]:
        print(add2(*inputs))


# example: context manager


@contextmanager
def temporary_workspace():
    ws_path = tempfile.mkdtemp()
    try:
        print(f"current workspace: {ws_path}")
        yield ws_path  # 将控制权交给 with block
    finally:
        shutil.rmtree(ws_path)
        print(f"clear workspace: {ws_path}")


def test_context_manager():
    with temporary_workspace() as ws:
        src = "/tmp/test/output.json"
        dst = os.path.join(ws, "output.bak")

        print(f"copy file: from {src} to {dst}")
        shutil.copy(src, dst)

        print("load json")
        with open(dst, mode="r", encoding="utf-8") as f:
            obj: dict = json.load(f)
            print(f"error code: {obj.get('error', 1)}")


if __name__ == "__main__":
    test_tool_cache()
    # test_context_manager()
