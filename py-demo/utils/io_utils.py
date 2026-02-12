import csv
import re
from pathlib import Path
from typing import Generator


def read_lines_lazy(f_path: str) -> Generator[str]:
    with open(f_path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()


def read_chunk_csv(f_path: str, chunk_size: int = 1000) -> Generator[list]:
    with open(f_path, mode="r", encoding="utf-8") as f:
        chunk = []
        reader = csv.DictReader(f)
        for row in reader:
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
            chunk.append(row)

        if len(chunk) > 0:
            yield chunk


def batch_rename_files(dir_path: str, pattern: str, replacement: str) -> None:
    for f_path in Path(dir_path).iterdir():
        if f_path.is_file():
            old_name = f_path.name
            new_name = re.sub(pattern=pattern, repl=replacement, string=old_name)
            if old_name != new_name:
                new_path = f_path.parent / new_name
                f_path.rename(new_path)
                print(f"rename: {old_name} -> {new_name}")


# Utils Test


def test_read_chunk_csv():
    f_path = "/tmp/test/output.csv"
    print("read csv:", f_path)
    for chunk in read_chunk_csv(f_path):
        for row in chunk:
            print(row)


def test_batch_rename_files():
    # update test_*.py to pytest_*.py
    batch_rename_files("/tmp/test/py_project", r"^test_", "pytest_")


if __name__ == "__main__":
    pass
