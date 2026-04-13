from pathlib import Path

import requests
from requests.models import Response


def download_file(url: str, saved_fpath: str) -> None:
    out = Path(saved_fpath)
    if out.exists():
        size_kb = out.stat().st_size / 1e3
        print(f"{out}: {size_kb:.1f} KB (cached)")
        return

    resp: Response = requests.get(url, timeout=60)
    resp.raise_for_status()

    if resp.content:
        out.write_bytes(bytes(resp.content))
        size_kb = out.stat().st_size / 1e3
        print(f"{out}: {size_kb:.1f} KB")


if __name__ == "__main__":
    pass
