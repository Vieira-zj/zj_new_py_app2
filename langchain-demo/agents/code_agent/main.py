from decision import DecisionModule
from dotenv import load_dotenv
from memory import MemoryModule
from perception import PerceptionModule
from tool_calling import ToolCallingModule

load_dotenv()


class CodeAgent:

    def __init__(self, project_path: str):
        # 初始化各个模块
        self.project_path = project_path
        self.perception = PerceptionModule(project_path)
        self.memory = MemoryModule(project_path)
        self.decision = DecisionModule()
        self.tool_calling = ToolCallingModule()

        # 初始化时, 读取并存储项目编码风格到长期记忆
        code_style = self.perception.project_tools.extract_code_style(project_path)
        self.memory.store_project_style(code_style)

    def run(self):
        print("🎉 代码生成与优化 Agent 已启动! 输入 'exit' 即可结束对话")
        print("💡 支持功能: 1. 输入自然语言需求生成代码; 2. 输入现有代码进行优化")

        while True:
            # 1. 接收用户输入
            user_input = input("\n请输入你的需求或代码:")
            if user_input.strip() == "exit":
                print("👋 bye")
                self.memory.clear_short_term_memory()  # 退出时清空短期记忆
                break

            # 2. 决策模块: 判断任务类型
            task_type = self.decision.judge_task_type(user_input)
            print(
                f"🔍 识别任务类型: {'代码生成' if task_type == 'generate' else '代码优化'}"
            )

            # 3. 感知模块: 收集信息, 主动补全缺失信息
            perception_result = self.perception.collect_information(
                user_input, task_type
            )
            if perception_result["status"] == "need_more_info":
                print(f"❓ {perception_result['question']}")
                continue  # 等待用户补充信息后重新运行

            collected_info = perception_result["data"]
            # 加入对话历史到收集的信息中
            collected_info["conversation_history"] = self.memory.retrieve_conversation()

            # 4. 决策模块: 生成执行决策
            decision_result = self.decision.make_decision(collected_info)
            if decision_result["status"] == "fail":
                print(f"❌ {decision_result['message']}")
                continue

            decision = decision_result["decision"]
            print(f"📋 执行策略: {decision['description']}")

            # 5. 工具调用模块: 执行决策, 获取结果
            tool_result = self.tool_calling.call_tool(decision)
            if tool_result["status"] == "success":
                print(f"✅ {tool_result['message']}")
                print("📝 结果如下:")
                print("-" * 50)
                print(tool_result["result"])
                print("-" * 50)
                # 6. 记忆模块: 存储对话历史到短期记忆
                self.memory.store_conversation(user_input, tool_result["result"])
            else:
                print(f"❌ 工具调用失败: {tool_result['message']}")


if __name__ == "__main__":
    input_project_path = input("请输入你的项目路径:").strip()
    agent = CodeAgent(input_project_path)
    agent.run()
