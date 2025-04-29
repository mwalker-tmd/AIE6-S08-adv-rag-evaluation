from src.retrieval.naive_retriever import NaiveRetriever
from configs.rag_config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI

from src.graph.nodes import generate_answer, retrieve_docs
from src.graph.schema import RAGState
from pathlib import Path

def baseline_rag_app(documents: list[str]):
    retriever = NaiveRetriever(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )

    chunks = retriever.split_text(documents)
    retriever.build_vectorstore(chunks)

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("generate", generate_answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.set_finish_point("generate")

    app = graph.compile()
    return app, retriever
