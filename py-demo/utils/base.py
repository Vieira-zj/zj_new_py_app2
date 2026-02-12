from typing import Any


class FlexibleDict(dict):
    def __getattr__(self, key: str) -> Any:
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value


# Test Utils


def test_flexible_dict():
    config = FlexibleDict(db_host="localhost", db_port=5432)
    if config.db_host:
        print("db_host:", config.db_host)
    if config.db_name:
        print("db_name:", config.db_name)


if __name__ == "__main__":
    test_flexible_dict()
