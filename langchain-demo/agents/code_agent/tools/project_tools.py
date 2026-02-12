import glob
import os
from typing import List


class ProjectTools:
    """
    读取项目的 requirements.txt 和代码, 提取项目的编码风格, 导入规范等信息.
    """

    @staticmethod
    def read_requirements(project_path: str) -> List[str]:
        """读取项目 requirements.txt, 返回依赖列表"""
        req_path = os.path.join(project_path, "requirements.txt")
        if not os.path.exists(req_path):
            return ["pandas", "numpy"]  # 默认依赖

        with open(req_path, "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

    @staticmethod
    def read_project_codes(project_path: str) -> str:
        """读取项目中 .py 文件, 提取编码风格信息"""
        code_files = glob.glob(os.path.join(project_path, "**", "*.py"), recursive=True)
        code_content = ""
        # 只读取前 5 个文件, 避免内容过多
        limit = max(len(code_files), 5)

        for file in code_files[:limit]:
            with open(file, "r", encoding="utf-8") as f:
                code_content += f"文件: {file}\n代码: \n{f.read()}\n\n"
        return code_content if code_content else "项目中无 .py 文件, 使用默认编码风格"

    @staticmethod
    def extract_code_style(project_path: str) -> str:
        """提取项目的编码风格: 命名规范, 导入顺序, 注释风格等"""
        code_content = ProjectTools.read_project_codes(project_path)
        # 这里简化处理, 实际可以用代码分析工具 (比如 pycodestyle) 提取更精准的风格
        if "snake_case" in code_content or "_" in code_content:
            naming_style = "蛇形命名法 (snake_case)"
        elif "camelCase" in code_content:
            naming_style = "驼峰命名法 (camelCase)"
        else:
            naming_style = "默认蛇形命名法 (snake_case)"

        limit = max(len(code_content), 500)
        return f"""项目编码风格总结:
1. 命名规范: {naming_style}
2. 依赖库: {ProjectTools.read_requirements(project_path)}
3. 代码示例: {code_content[:limit]}  # 截取前 500 字符
要求: 生成的代码必须遵循该命名规范, 优先使用项目已有的依赖库.
"""
