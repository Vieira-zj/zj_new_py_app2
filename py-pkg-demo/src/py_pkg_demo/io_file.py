import os
from pathlib import Path


class IOFile:
    def __init__(self, path: str) -> None:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"path is not exist: {path}")

        self.path = p

    def ls(self) -> list[str]:
        if self.path.is_file():
            return [str(self.path.absolute())]
        return [str(item.absolute()) for item in self.path.iterdir()]

    def size(self) -> str:
        if self.path.is_dir():
            return "0B"
        return f"{self.path.stat().st_size/1e3:.2f}B"

    def find(self, key: str) -> list[str]:
        if self.path.is_file():
            if key in self.path.name:
                return [str(self.path.absolute())]
            return []

        results: list[str] = []
        for dirpath, _, filenames in os.walk(self.path):
            for fname in filenames:
                if key in fname:
                    results.append(os.path.join(dirpath, fname))
        return results
