# src/graph/schema.py
from typing import TypedDict, Annotated

class RAGState(TypedDict):
    question: str
    retrieved_docs: Annotated[list[str], lambda a, b: (a + b)[:4]]
    answer: str


# src/graph/nodes.py
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain.chat_models import ChatOpenAI
from typing import List

# Retriever node function
def retrieve_docs(state):
    question = state["question"]
    docs = state["retriever"].retrieve(question)
    return {"retrieved_docs": [d["content"] for d in docs]}

# Generator node function
def generate_answer(state):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use only the provided context to answer the question."),
        ("human", "Context:\n{context}\n---\nQuestion: {question}")
    ])

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    chain = (
        RunnableMap({
            "context": lambda x: "\n".join(x["retrieved_docs"]),
            "question": lambda x: x["question"]
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(state)
    return {"answer": answer}
