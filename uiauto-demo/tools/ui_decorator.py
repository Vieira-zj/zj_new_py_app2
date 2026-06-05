import ast
from functools import wraps

from playwright.sync_api import Locator, Page


def _get_func_name(node):
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        prefix = _get_func_name(node.value)
        return f"{prefix}.{node.attr}"

    if isinstance(node, ast.Call):
        return _get_func_name(node.func)

    raise ValueError(f"not support ast node type: {type(node).__name__}")


def _parse_ui_locator(code: str):
    tree = ast.parse(code, mode="eval")
    call = tree.body

    if not isinstance(call, ast.Call):
        raise ValueError("not function call")

    func_name = _get_func_name(call.func)
    args = [ast.literal_eval(arg) for arg in call.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}
    return func_name, args, kwargs


def _web_locator(page: Page | None, locator: str, **kwargs) -> Locator | None:
    if page is None:
        raise ValueError("page object is null")
    if not isinstance(page, Page):
        raise ValueError("page object is not playwright page")

    if len(kwargs) > 0:
        locator = locator.format(**kwargs)

    loc_name, loc_args, loc_kwargs = _parse_ui_locator(locator)
    match (loc_name):
        case "get_by_text":
            return page.get_by_text(*loc_args, **loc_kwargs)
        case "get_by_role":
            return page.get_by_role(*loc_args, **loc_kwargs)
        case "locator":
            return page.locator(*loc_args, **loc_kwargs)
        case _:
            raise ValueError("not support playwright web locator")


def pw_web(locator: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            page = vars(args[0])["page"]
            ele = _web_locator(page=page, locator=locator, **kwargs)
            return ele

        return wrapper

    return decorator


def appium_android(locator: str):
    _ = locator
    # to impl


def appium_ios(locator: str):
    _ = locator
    # to impl


# Testing


def parse_ui_locator_test():
    locator = 'get_by_role("menuitem", name="setting Settings")'
    # locator = 'page.get_by_role("menuitem", name="setting Settings")'
    fn_name, args, kwargs = _parse_ui_locator(locator)
    print(f"fn_name={fn_name}, args=[{args}], kwargs=[{kwargs}]")


def pw_web_decorator_test():
    class Login:
        def __init__(self) -> None:
            self.page = {"title": "login"}

        @pw_web(locator="locator(id='{title_id}')")
        def get_title(self, title_id: str):
            _ = title_id

    login = Login()
    login.get_title(title_id="abc")


if __name__ == "__main__":
    # parse_ui_locator_test()
    pw_web_decorator_test()
