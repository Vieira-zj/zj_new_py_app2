from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tools.code_tools import CodeTools


class ToolCallingModule:

    def __init__(self):
        self.code_tools = CodeTools()

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type(Exception),
    )
    def call_tool(self, decision: dict) -> dict:
        """调用对应的工具, 执行决策"""
        strategy = decision["strategy"]
        params = decision["params"]

        try:
            if strategy == "generate_code":
                code = self.code_tools.generate_code(**params)
                return {"status": "success", "result": code, "message": "代码生成成功"}
            if strategy == "optimize_code":
                optimized_code = self.code_tools.optimize_code(**params)
                return {
                    "status": "success",
                    "result": optimized_code,
                    "message": "代码优化成功",
                }
            return {"status": "fail", "message": f"不支持的策略: {strategy}"}
        except Exception as e:
            print(f"工具调用失败, 原因: {str(e)}, 正在重试...")
            raise e  # 抛出异常, 触发重试机制
