from pathlib import Path

from py_pkg_demo import IOFile


def test_pathlib():
    p = Path("/tmp/test/output.txt")
    assert p.name == "output.txt"


def test_list_dir():
    path = "/tmp/test"
    io_file = IOFile(path)
    results = io_file.ls()
    assert len(results) > 0

    print(f"\nls {path}")
    for item in results:
        print(item)


def test_get_file_size():
    path = "/tmp/test/output.json"
    io_file = IOFile(path)
    size = io_file.size()
    assert len(size) > 0
    print(f"\n{path}: {size}")


def test_find_file():
    path = "~/Downloads/tmps"
    io_file = IOFile(path)

    key = "tool"
    results = io_file.find(key)
    assert len(results) > 0

    print(f"\nfind [{key}] in {path}")
    for item in results:
        print(item)
