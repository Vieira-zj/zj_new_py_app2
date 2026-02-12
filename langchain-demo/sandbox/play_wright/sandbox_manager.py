import os
from typing import Any, Dict, Optional

from agentrun.sandbox import BrowserSandbox, Sandbox, TemplateType
from dotenv import load_dotenv

load_dotenv()

_global_manager: Optional[SandboxManager] = None


def get_global_manager() -> SandboxManager:
    global _global_manager
    if _global_manager is None:
        _global_manager = SandboxManager()
    return _global_manager


def reset_global_manager():
    global _global_manager
    if _global_manager:
        _global_manager.destroy()
    _global_manager = None


class SandboxManager:
    def __init__(self):
        self._sandbox: Optional[BrowserSandbox] = None
        self._sandbox_id: Optional[str] = None
        self._cdp_url: Optional[str] = None
        self._vnc_url: Optional[str] = None

    def create(
        self, template_name: Optional[str] = None, idle_timeout: int = 3000
    ) -> Dict[str, Any]:
        try:

            if self._sandbox is not None:
                return self.get_info()
            if template_name is None:
                template_name = os.getenv(
                    "BROWSER_TEMPLATE_NAME", "sandbox-browser-demo"
                )
            self._sandbox = Sandbox.create(
                template_type=TemplateType.BROWSER,
                template_name=template_name,
                sandbox_idle_timeout_seconds=idle_timeout,
            )
            self._sandbox_id = self.get_sandbox_id()
            self._cdp_url = self.get_cdp_url()
            self._vnc_url = self.get_vnc_url()
            return self.get_info()
        except ImportError as e:
            print(e)
            raise RuntimeError(
                "agentrun-sdk does not install, pls run: pip install agentrun-sdk[playwright,server]"
            ) from e
        except Exception as e:
            raise RuntimeError(f"create Sandbox failed: {str(e)}") from e

    def get_info(self) -> Dict[str, Any]:
        if self._sandbox is None:
            raise RuntimeError("no active sandbox, pls create")
        return {
            "sandbox_id": self._sandbox_id,
            "cdp_url": self._cdp_url,
            "vnc_url": self._vnc_url,
        }

    def get_cdp_url(self) -> Optional[str]:
        if self._sandbox is None:
            raise RuntimeError("no active sandbox, pls create")
        return self._sandbox.get_cdp_url()

    def get_vnc_url(self) -> Optional[str]:
        if self._sandbox is None:
            raise RuntimeError("no active sandbox, pls create")
        return self._sandbox.get_vnc_url()

    def get_sandbox_id(self) -> Optional[str]:
        return self._sandbox_id

    def is_active(self) -> bool:
        return self._sandbox is not None

    def destroy(self) -> str:
        if self._sandbox is None:
            raise RuntimeError("no active sandbox, pls create")

        try:
            sandbox_id = self.get_sandbox_id()
            if hasattr(self._sandbox, "delete"):
                self._sandbox.delete()
            elif hasattr(self._sandbox, "stop"):
                self._sandbox.stop()
            return f"Sandbox has been destoryed: {sandbox_id}"
        except Exception as e:
            return f"destory Sandbox failed: {str(e)}"
        finally:
            self._sandbox = None
            self._sandbox_id = None
            self._cdp_url = None
            self._vnc_url = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        return False
