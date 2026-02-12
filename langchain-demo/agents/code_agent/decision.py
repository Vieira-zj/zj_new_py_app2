from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tools import get_api_key

load_dotenv()


class DecisionModule:

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", api_key=get_api_key, temperature=0.1
        )

    def judge_task_type(self, user_input: str) -> str:
        """判断任务类型: generate (代码生成) 或 optimize (代码优化)"""
        prompt = f"""请判断用户输入的任务类型, 只能返回 "generate" 或 "optimize", 不要其他内容:
- generate: 用户需要生成新的代码 (包含 "写", "生成", "实现", "函数", "方法" 等关键词).
- optimize: 用户需要优化现有代码 (包含 "优化", "修复", "改进", "重构" 等关键词, 或直接提供代码).
        
用户输入: {user_input}
"""
        completion = self.llm.invoke(prompt)
        return completion.model_dump_json()

    def make_decision(self, collected_info: dict) -> dict:
        """根据收集到的信息, 生成执行决策"""
        task_type = collected_info["task_type"]
        user_input = collected_info["user_input"]
        code_style = collected_info["code_style"]

        # 决策规则: 不同任务类型对应不同策略
        if task_type == "generate":
            decision = {
                "strategy": "generate_code",
                "params": {"demand": user_input, "code_style": code_style},
                "description": "调用代码生成工具, 根据需求和项目风格生成完整代码",
            }
        elif task_type == "optimize":
            decision = {
                "strategy": "optimize_code",
                "params": {"code": user_input, "code_style": code_style},
                "description": "调用代码优化工具, 修复bug, 提升性能, 规范格式",
            }
        else:
            decision = {}

        # 规则校验: 确保决策参数完整
        if not self._validate_decision(decision):
            return {"status": "fail", "message": "决策参数不完整, 请补充关键信息"}
        return {"status": "success", "decision": decision}

    def _validate_decision(self, decision: dict) -> bool:
        strategy = decision["strategy"]
        params = decision["params"]

        if strategy == "generate_code":
            return all(key in params for key in ["demand", "code_style"])
        if strategy == "optimize_code":
            return all(key in params for key in ["code", "code_style"])
        return False
