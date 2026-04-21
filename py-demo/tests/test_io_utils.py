from utils import scan_files


def test_scan_files():
    path = "~/Downloads/tmps"
    results = scan_files(path, extension=".py")
    assert len(results) > 0

    print(f"all python files in [{path}]:")
    for p in results:
        print(f"{p}")
