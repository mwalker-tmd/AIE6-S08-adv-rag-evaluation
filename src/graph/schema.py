# src/graph/schema.py
from typing import Any, Annotated, TypedDict

class RAGState(TypedDict):
    question: str
    retriever: Any
    retrieved_docs: Annotated[list[str], lambda a, b: (a + b)[:4]]
    answer: str
