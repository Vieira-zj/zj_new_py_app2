import os

from dotenv import load_dotenv
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def get_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "")


class MemoryModule:

    def __init__(self, project_path: str):
        # 项目路径, 用于关联记忆
        self.project_path = project_path
        # 初始化 Embedding 模型 (用于文本转向量)
        self.embeddings = OpenAIEmbeddings(api_key=get_api_key)
        # 长期记忆: 向量数据库, 存储项目编码风格 (核心记忆, 永久存储)
        self.long_term_memory = Chroma(
            persist_directory="./vector_db",
            embedding_function=self.embeddings,
            collection_name="project_style",
        )
        # 短期记忆: 对话历史, 存储用户需求和 Agent 回复 (临时记忆, 任务结束后清空)
        self.short_term_memory = ConversationBufferMemory(
            memory_key="conversation_history", return_messages=True
        )

    def store_project_style(self, code_style: str):
        """存储项目编码风格到长期记忆 (向量数据库)"""
        # 先检查是否已存储该项目的风格, 避免重复
        existing_docs = self.long_term_memory.similarity_search(self.project_path, k=1)
        if not existing_docs:
            self.long_term_memory.add_texts(
                texts=[code_style], metadatas=[{"project_path": self.project_path}]
            )
            self.long_term_memory.persist()  # 持久化存储

    def retrieve_project_style(self) -> str:
        """从长期记忆中检索项目编码风格 (语义检索)"""
        docs = self.long_term_memory.similarity_search(self.project_path, k=1)
        if docs:
            return docs[0].page_content
        return "默认编码风格: 蛇形命名法, 优先使用 pandas, numpy 依赖, 遵循 PEP8 规范"

    def store_conversation(self, user_message: str, agent_response: str):
        """存储对话历史到短期记忆"""
        self.short_term_memory.save_context(
            inputs={"human_input": user_message},
            outputs={"agent_output": agent_response},
        )

    def retrieve_conversation(self) -> str:
        """检索短期记忆中的对话历史"""
        return self.short_term_memory.load_memory_variables({})["conversation_history"]

    def clear_short_term_memory(self):
        """清空短期记忆（任务结束后调用）"""
        self.short_term_memory.clear()
