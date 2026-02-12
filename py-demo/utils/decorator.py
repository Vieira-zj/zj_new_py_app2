import time
from functools import wraps
from typing import Callable


def exec_time(fn: Callable):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start) * 1000  # milli sec
            print(f"[{fn.__name__}] exec time: {elapsed:.2f}ms")

    return wrapped


class CacheDecorator:
    def __init__(self, max_size: int = 100) -> None:
        self.cache = {}
        self.max_size = max_size

    def __call__(self, fn: Callable):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            if key in self.cache:
                return self.cache[key]

            result = fn(*args, **kwargs)

            if len(self.cache) >= self.max_size:
                # remove oldest cache item
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = result
            return result

        return wrapped


# Decorator Test


def test_exec_time():
    @exec_time
    def data_processing(data: list):
        time.sleep(0.1)
        print(f"len: {len(data)}")

    data_processing(list(range(1, 10)))


def test_cache_decorator():
    @CacheDecorator(max_size=10)
    def computation(n: int) -> int:
        time.sleep(1)
        print(f"{n} * {n}")
        return n * n

    print(computation(10))
    print(computation(10))


if __name__ == "__main__":
    # test_exec_time()
    test_cache_decorator()
