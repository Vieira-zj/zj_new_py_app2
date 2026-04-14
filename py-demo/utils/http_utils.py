import json
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


def post_with_stream(url: str, data: dict) -> str:
    resp: Response = requests.post(url, json=data, stream=True, timeout=30)
    with resp as r:
        r.raise_for_status()
        resp_data = ""
        for line in r.iter_lines(decode_unicode=True):
            if line:
                resp_dict: dict = json.loads(line)
                if "message" in resp_dict:
                    resp_data += resp_dict["message"]["content"]

    return resp_data


if __name__ == "__main__":
    pass
