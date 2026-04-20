import csv
import os
import re
from pathlib import Path
from typing import Generator


def read_lines_lazy(f_path: str) -> Generator[str, None, None]:
    with open(f_path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()


def read_chunk_csv(
    f_path: str, chunk_size: int = 1000
) -> Generator[list[str], None, None]:
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


def scan_files(root_path: str, extension: str) -> list[str]:
    """Recursively scan root_path for files with spec extension."""
    if not extension:
        raise ValueError("extension is required")

    root = Path(root_path).expanduser().resolve()
    if root.is_file() and has_extension(root, extension):
        return [str(root)]

    results: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if has_extension(fpath, extension):
                results.append(str(fpath))

    return results


def has_extension(path: Path, extension: str) -> bool:
    return path.suffix.lower() == extension


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


def test_scan_files():
    path = "~/Downloads/tmps"
    results = scan_files(path, extension=".py")
    print(f"all python files in [{path}]:")
    for p in results:
        print(f"{p}")


if __name__ == "__main__":
    test_scan_files()
