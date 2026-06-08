from functools import wraps


# pylint: disable=W0613:unused-argument
def case_meta(
    key: str,
    priority: str,
    desc: str | None = None,
    ticket_id: str | None = None,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    pass
