def pkg_help():
    print("py unit test package.")


# 使用 from tests import * 导入时, 只会导入 pkg_help 函数.
__all__ = ["pkg_help"]
