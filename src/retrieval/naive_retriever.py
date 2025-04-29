import json
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

class NaiveRetriever:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None

    def split_text(self, documents: List[str]) -> List[str]:
        chunks = []
        for doc in documents:
            chunks.extend(self.text_splitter.split_text(doc))
        return chunks

    def build_vectorstore(self, chunks: List[str]):
        self.vectorstore = FAISS.from_texts(chunks, self.embeddings)

    def save_vectorstore(self, save_path: str):
        if self.vectorstore:
            self.vectorstore.save_local(save_path)

    def load_vectorstore(self, load_path: str):
        self.vectorstore = FAISS.load_local(load_path, self.embeddings)

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Build or load first.")

        docs = self.vectorstore.similarity_search(query, k=k)
        return [{"content": doc.page_content} for doc in docs]
