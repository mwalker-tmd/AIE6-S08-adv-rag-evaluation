# src/rag/semantic_rag_app.py

#from src.chunking.semantic_chunker import SemanticChunker
from src.retrieval.semantic_chunker import SemanticChunker
from configs.rag_config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.graph.schema import RAGState
from src.graph.nodes import generate_answer, retrieve_docs
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI


def semantic_rag_app(documents: list[str]):
    chunker = SemanticChunker(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )

    chunks = chunker.chunk_documents(documents)
    chunker.build_vectorstore(chunks)

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("generate", generate_answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.set_finish_point("generate")

    app = graph.compile()
    return app, chunker

