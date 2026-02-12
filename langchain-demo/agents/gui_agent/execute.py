import time

import mss
import pyautogui
import pyperclip

# 允许鼠标移动到屏幕角落 (默认会触发 fail-safe)
pyautogui.FAILSAFE = False


class Operation:
    """GUI 操作工具类"""

    def click(self, x: int, y: int):
        """点击指定坐标"""
        print(f"🖱️  点击坐标 ({x}, {y})")
        pyautogui.click(x=x, y=y)

    def double_click(self, x: int, y: int):
        """点击指定坐标"""
        print(f"🖱️  点击坐标 ({x}, {y})")
        pyautogui.doubleClick(x=x, y=y)

    def input(self, text: str):
        """输入文本 (使用粘贴方式, 支持中文)"""
        print(f"⌨️  输入: {text}")
        pyperclip.copy(text)
        pyautogui.hotkey("command", "v")

    def screenshot(self, save_path: str):
        """截图并保存"""
        with mss.mss() as sct:
            sct.shot(output=save_path)
        print(f"📸 截图已保存: {save_path}")

    def hotkey(self, *keys):
        """按下组合键 (如 ctrl+c)"""
        print(f"⌨️  按下组合键: {' + '.join(keys)}")
        pyautogui.hotkey(*keys)

    def wait(self, seconds: float = 1.0):
        """等待指定时间"""
        print(f"⏱️  等待 {seconds} 秒...")
        time.sleep(seconds)
