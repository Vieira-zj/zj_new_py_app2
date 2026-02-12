from typing import Any, Dict, List

from tools.project_tools import ProjectTools


class PerceptionModule:

    def __init__(self, project_path: str):
        self.project_path = project_path
        self.project_tools = ProjectTools()

    def collect_information(self, user_input: str, task_type: str) -> Dict[str, Any]:
        """
        收集信息: 用户输入+项目信息+对话历史
        task_type: 任务类型, "generate" (代码生成) 或 "optimize" (代码优化)
        """
        # 1. 收集项目信息 (跨源: 文件系统+代码文件)
        code_style = self.project_tools.extract_code_style(self.project_path)

        # 2. 收集用户输入, 检查是否缺失关键信息
        collected_info = {
            "user_input": user_input,
            "code_style": code_style,
            "task_type": task_type,
        }

        # 3. 主动补全关键信息 (根据任务类型判断)
        missing_info = self._check_missing_info(collected_info)
        if missing_info:
            return {
                "status": "need_more_info",
                "question": f"为了更精准地完成任务, 请补充以下信息: {', '.join(missing_info)}",
                "collected_info": collected_info,
            }
        return {"status": "complete", "data": collected_info}

    def _check_missing_info(self, collected_info: dict) -> List[str]:
        """检查是否缺失关键信息"""
        missing = []
        task_type = collected_info["task_type"]
        user_input = collected_info["user_input"]

        if task_type == "generate":
            # 代码生成任务: 检查是否有明确的功能描述, 输入输出
            if (
                "函数" not in user_input
                and "方法" not in user_input
                and "实现" not in user_input
            ):
                missing.append("具体功能 (比如 '写一个 xxx 函数')")
            if "输入" not in user_input and "参数" not in user_input:
                missing.append("输入参数或数据来源")
        elif task_type == "optimize":
            # 代码优化任务: 检查是否提供了完整代码
            if (
                "def" not in user_input
                and "class" not in user_input
                and "=" not in user_input
            ):
                missing.append("Python 代码不完整")

        return missing
