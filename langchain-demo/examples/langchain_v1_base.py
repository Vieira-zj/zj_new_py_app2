from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Demo: Sequence Chain


def test_langchain_v1_seq_01():
    prompt = ChatPromptTemplate.from_template("用一句话解释: {concept}")
    llm = OllamaLLM(model="qwen2.5-coder:7b", temperature=1)
    chain = prompt | llm | StrOutputParser()
    chain.invoke({"concept": "LangChain 1.0是什么"})


def test_langchain_v1_seq_02():
    prompt = ChatPromptTemplate.from_template(
        "请用非常简短的一句话总结下面的内容: {text}"
    )
    llm = OllamaLLM(model="qwen2.5-coder:7b", temperature=0.0)
    chain = RunnableParallel(
        original=RunnablePassthrough(),
        summary=prompt | llm | StrOutputParser(),
    )

    input_text = "Python是一种高级编程语言, 由Guido van Rossum于1991年首次发布。它是一种解释型语言, 具有简单易学、代码可读性高、功能强大等特点。Python广泛应用于Web开发、数据分析、人工智能、科学计算等领域。"
    result = chain.invoke({"text": input_text})
    print(result)  # {"original":{},"summary":{}}


# Demo: RAG


def build_retriever():
    loader = TextLoader(r"/tmp/test/output.md", encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(splits, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # topk=2
    return retriever  # Retriever 是 Runnable


def test_langchain_v1_rag_01():
    prompt = ChatPromptTemplate.from_template(
        """你是一个严谨的技术助理, 只能基于给定资料回答.

资料:
{context}

问题:
{question}
"""
    )

    llm = OllamaLLM(model="qwen2.5-coder:7b", temperature=0.0)
    retriever = build_retriever()
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    result = rag_chain.invoke("什么是逆向熵稳定区？")
    print(result)


# Demo: Branch Chain


def test_langchain_v1_cond_01():
    llm = OllamaLLM(model="qwen2.5-coder:7b", temperature=0.0)
    retriever = build_retriever()

    # 上下文充分
    high_answer_chain = (
        ChatPromptTemplate.from_template(
            """你是一个严谨的技术助理, 请基于资料回答问题。

                资料:
                {context}

                问题:
                {question}
                """
        )
        | llm
        | StrOutputParser()
    )

    # 回答 + 明确风险边界
    low_answer_chain = (
        ChatPromptTemplate.from_template(
            """你是一个谨慎的技术助理。

                以下资料可能不足以完整回答问题, 请:
                - 明确指出不确定性
                - 尽量引用资料原文
                - 不要做超出资料的推断

                资料:
                {context}

                问题:
                {question}
                """
        )
        | llm
        | StrOutputParser()
    )

    # 拒绝回答
    refuse_chain = RunnableLambda(lambda _: "检索资料不足, 无法对该问题给出可靠回答。")

    confidence_prompt = ChatPromptTemplate.from_template(
        """
        你是一个严格的检索结果评估器。

        问题:
        {question}

        检索到的内容:
        {context}

        请判断：这些内容是否足以回答该问题？

        只输出以下三个标签之一：
        - HIGH
        - LOW
        - NONE
        """
    )
    confidence_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | confidence_prompt
        | llm
        | StrOutputParser()
    )

    multi_strategy_rag = {
        "context": retriever,
        "question": RunnablePassthrough(),
        "confidence": confidence_chain,
    } | RunnableBranch(
        (
            lambda x: x["confidence"] == "HIGH",  # pyright: ignore[reportIndexIssue]
            high_answer_chain,
        ),
        (
            lambda x: x["confidence"] == "LOW",  # pyright: ignore[reportIndexIssue]
            low_answer_chain,
        ),
        refuse_chain,
    )

    result = multi_strategy_rag.invoke("什么是阿尔法层级漂移定律？")
    print(result)


if __name__ == "__main__":
    pass
