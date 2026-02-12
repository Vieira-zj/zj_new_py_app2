import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pycodestyle import StyleGuide

load_dotenv()


def get_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "")


class CodeTools:

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=get_api_key,
            temperature=0.3,
        )

    def generate_code(self, demand: str, code_style: str) -> str:
        """根据需求和代码风格, 生成代码"""
        prompt = f"""你是一个资深 Python 开发工程师, 需要根据用户需求生成符合项目风格的代码.
项目编码风格: {code_style}
生成代码要求: 
1. 严格遵循项目编码风格和命名规范
2. 代码完整可运行, 包含必要的导入, 函数定义, 参数说明, 异常处理
3. 优先使用项目已有的依赖库, 不使用未列出的依赖
4. 代码注释详细, 关键步骤说明用途
5. 生成后检查是否符合PEP8规范, 避免语法错误

用户需求: {demand}
请直接返回生成的代码, 不要其他多余内容.
"""
        completion = self.llm.invoke(prompt)
        return completion.model_dump_json()

    def optimize_code(self, code: str, code_style: str) -> str:
        """优化现有代码: 修复 bug, 提升性能, 规范格式"""
        prompt = f"""你是一个 Python 代码优化专家, 需要对用户提供的代码进行以下优化:
1. 修复语法错误和逻辑 bug (比如空指针, 索引越界, 循环效率低等)
2. 提升性能 (比如用列表推导式替代嵌套循环, 减少重复计算, 优化数据结构)
3. 规范格式 (遵循 PEP8 规范和项目编码风格)
4. 完善注释和参数说明
5. 保持代码功能不变

项目编码风格: {code_style}
用户提供的代码: {code}
请直接返回优化后的代码, 不要其他多余内容.
"""
        completion = self.llm.invoke(prompt)
        optimized_code = completion.model_dump_json()

        tmp_file_path = "/tmp/test/tmp.py"
        with open("tmp_file_path", "w", encoding="utf-8") as f:
            f.write(optimized_code)

        # 额外检查 PEP8 规范, 输出规范提示
        sg = StyleGuide()
        result = sg.check_files(tmp_file_path)
        if result.total_errors == 0:
            return optimized_code + "\n\n# 代码优化完成, 符合 PEP8 规范"
        return (
            optimized_code
            + f"\n\n# 代码优化完成, 剩余 PEP8 规范问题: {result.total_errors}"
        )

    def explain_code(self, code: str) -> str:
        """解释代码的功能, 逻辑和关键步骤 (可选功能)"""
        prompt = f"""请详细解释以下 Python 代码的功能, 逻辑流程和关键步骤, 用通俗易懂的语言说明:
{code}
解释要求: 分点说明, 清晰明了, 适合新手理解.
"""
        completion = self.llm.invoke(prompt)
        return completion.model_dump_json()
