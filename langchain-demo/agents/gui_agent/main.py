import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

import pyautogui
from execute import Operation
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from model import LVMChat
from prompts import COMPUTER_USE_UITARS


class AgentState(TypedDict):
    instruction: str  # 用户指令
    screenshot_path: str  # 当前截图路径
    step: int  # 当前步骤
    thought: str  # 模型思考
    action: str  # 模型输出的动作
    finished: bool  # 是否完成


class GUIAgent:
    """GUI 自动化 Agent"""

    def __init__(self, instruction: str, model_name: Optional[str]):
        self.instruction = instruction
        self.operation = Operation()
        self.lvm_chat = LVMChat(model_name=model_name)
        self.s_dir = Path("/tmp/gui_auto")
        self.s_dir.mkdir(exist_ok=True)

        # 获取屏幕尺寸用于坐标映射

        self.screen_width, self.screen_height = pyautogui.size()
        print(f"🖥️  屏幕尺寸: {self.screen_width}x{self.screen_height}")

    def normalize_coords(self, x: int, y: int) -> tuple[int, int]:
        """将归一化坐标 (0-1000) 转换为实际像素坐标"""
        actual_x = int(x / 1000.0 * self.screen_width)
        actual_y = int(y / 1000.0 * self.screen_height)
        print(f"   归一化坐标 ({x}, {y}) -> 实际坐标 ({actual_x}, {actual_y})")
        return actual_x, actual_y

    def take_screenshot(self, state: AgentState) -> AgentState:
        """步骤1: 截图并保存"""
        step = state.get("step", 0) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = str(self.s_dir / f"step_{step}_{timestamp}.png")

        self.operation.screenshot(screenshot_path)

        return {
            **state,
            "instruction": self.instruction,
            "screenshot_path": screenshot_path,
            "step": step,
            "finished": False,
        }

    def model_decide(self, state: AgentState) -> AgentState:
        """步骤2: 模型决策 (自动使用会话历史)"""
        prompt = COMPUTER_USE_UITARS.format(instruction=state["instruction"])

        # 调用多模态模型 (use_history=True 自动保留上下文)
        response = self.lvm_chat.get_multimodal_response(
            text=prompt,
            image_paths=state["screenshot_path"],
            use_history=True,
        )
        print(f"\n📸 Step {state['step']} - 模型响应:\n{response}\n")

        # 解析 JSON 响应
        try:
            result = json.loads(response)
            thought = result.get("Thought", "")
            action = result.get("Action", "")
        except json.JSONDecodeError:
            # 如果不是 JSON 格式，尝试正则提取
            thought_match = re.search(r'"Thought":\s*"([^"]*)"', response)
            action_match = re.search(r'"Action":\s*"([^"]*)"', response)
            thought = thought_match.group(1) if thought_match else ""
            action = action_match.group(1) if action_match else ""

        return {**state, "thought": thought, "action": action}

    def execute_action(self, state: AgentState) -> AgentState:
        """步骤3: 解析并执行动作"""
        action = state["action"]
        if not action:
            print("⚠️ 没有可执行的动作")
            return {**state, "finished": True}

        # 检查是否完成
        if action.startswith("finished("):
            content_match = re.search(r"finished\(content='([^']*)'\)", action)
            content = content_match.group(1) if content_match else "任务完成"
            print(f"✅ 任务完成: {content}")
            return {**state, "finished": True}

        # 解析并执行动作
        try:
            self._parse_and_execute(action)
        except Exception as e:
            print(f"❌ 执行动作失败: {e}")
            print(f"   动作: {action}")

        return state

    def _parse_and_execute(self, action: str):
        print(f"🔧 执行动作: {action}")

        if action.startswith("click("):
            self._execute_click(action)
        elif action.startswith("left_double("):
            self._execute_double_click(action)
        elif action.startswith("type("):
            self._execute_input(action)
        elif action.startswith("hotkey("):
            self._execute_hot_key(action)
        elif action.startswith("scroll("):
            self._execute_scroll(action)
        elif action.startswith("wait("):
            self.operation.wait(seconds=5)
        elif action.startswith("drag("):
            self._execute_drag(action)

        # 等待一下让界面响应
        self.operation.wait(seconds=1)

    def _execute_click(self, action: str):
        # 尝试带标签的格式 click(point='<point>x y</point>')
        point_match = re.search(r"<point>(\d+)\s+(\d+)</point>", action)
        if not point_match:
            # 尝试不带标签的格式 click(point='x y')
            point_match = re.search(r"point=['\"](\d+)\s+(\d+)['\"]", action)

        if point_match:
            x, y = int(point_match.group(1)), int(point_match.group(2))
            actual_x, actual_y = self.normalize_coords(x, y)
            self.operation.click(actual_x, actual_y)
        else:
            print(f"⚠️ 无法解析点击坐标: {action}")

    def _execute_double_click(self, action: str):
        # 尝试带标签的格式 left_double(point='<point>x y</point>')
        point_match = re.search(r"<point>(\d+)\s+(\d+)</point>", action)
        if not point_match:
            # 尝试不带标签的格式 double_click(point='x y')
            point_match = re.search(r"point=['\"](\d+)\s+(\d+)['\"]", action)

        if point_match:
            x, y = int(point_match.group(1)), int(point_match.group(2))
            actual_x, actual_y = self.normalize_coords(x, y)
            self.operation.double_click(actual_x, actual_y)
        else:
            print(f"⚠️ 无法解析双击坐标: {action}")

    def _execute_input(self, action: str):
        # type(content='xxx')
        content_match = re.search(r"content=['\"]([^'\"]*)['\"]", action)
        if content_match:
            text = content_match.group(1)
            # 处理转义字符
            text = text.replace(r"\'", "'").replace(r"\"", '"').replace(r"\n", "\n")
            self.operation.input(text)

    def _execute_hot_key(self, action: str):
        # hotkey(key='ctrl c')
        key_match = re.search(r"key=['\"]([^'\"]*)['\"]", action)
        if key_match:
            keys = key_match.group(1).split()
            self.operation.hotkey(*keys)

    def _execute_scroll(self, action: str):
        # 尝试带标签的格式 scroll(point='<point>x y</point>', direction='down')
        point_match = re.search(r"<point>(\d+)\s+(\d+)</point>", action)
        if not point_match:
            # 尝试不带标签的格式 scroll(point='x y', direction='down')
            point_match = re.search(r"point=['\"](\d+)\s+(\d+)['\"]", action)

        direction_match = re.search(r"direction=['\"]([^'\"]*)['\"]", action)
        if point_match and direction_match:
            x, y = int(point_match.group(1)), int(point_match.group(2))
            actual_x, actual_y = self.normalize_coords(x, y)
            direction = direction_match.group(1)
            # 移动到位置并滚动
            pyautogui.moveTo(actual_x, actual_y)
            scroll_amount = 3 if direction in ["up", "left"] else -3
            pyautogui.scroll(scroll_amount)

    def _execute_drag(self, action: str):
        # drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
        # 尝试带标签的格式
        start_match = re.search(
            r"start_point=['\"]<point>(\d+)\s+(\d+)</point>['\"]", action
        )
        end_match = re.search(
            r"end_point=['\"]<point>(\d+)\s+(\d+)</point>['\"]", action
        )
        if not start_match:
            # 尝试不带标签的格式
            start_match = re.search(r"start_point=['\"](\d+)\s+(\d+)['\"]", action)
            end_match = re.search(r"end_point=['\"](\d+)\s+(\d+)['\"]", action)

        if start_match and end_match:
            x1, y1 = int(start_match.group(1)), int(start_match.group(2))
            x2, y2 = int(end_match.group(1)), int(end_match.group(2))
            actual_x1, actual_y1 = self.normalize_coords(x1, y1)
            actual_x2, actual_y2 = self.normalize_coords(x2, y2)
            pyautogui.moveTo(actual_x1, actual_y1)
            pyautogui.drag(actual_x2 - actual_x1, actual_y2 - actual_y1, duration=0.5)

    def should_continue(self, state: AgentState) -> str:
        """判断是否继续循环"""
        return "end" if state.get("finished", False) else "continue"

    def run(self):
        """运行 Agent"""
        # 构建 graph
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("screenshot", self.take_screenshot)
        workflow.add_node("decide", self.model_decide)
        workflow.add_node("execute", self.execute_action)

        # 添加边
        workflow.set_entry_point("screenshot")
        workflow.add_edge("screenshot", "decide")
        workflow.add_edge("decide", "execute")
        workflow.add_conditional_edges(
            "execute", self.should_continue, {"continue": "screenshot", "end": END}
        )

        # 编译并运行
        app = workflow.compile()

        print(f"🚀 开始执行任务: {self.instruction}\n")

        # 设置递归限制为 20 步
        param = {"instruction": self.instruction, "step": 0}
        final_state = app.invoke(
            AgentState(**param), config=RunnableConfig(recursion_limit=20)
        )

        print(f"\n🎉 任务完成! 共执行 {final_state['step']} 步")
        return final_state


def main():
    instruction = "打开 Chrome 浏览器查询 GUI, 找到 wikipedia 的介绍页面进行查看"
    agent = GUIAgent(instruction=instruction, model_name=None)
    agent.run()


if __name__ == "__main__":
    main()
