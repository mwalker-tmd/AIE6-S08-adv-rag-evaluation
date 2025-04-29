import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np

class SemanticChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, similarity_threshold: float = 0.7, max_tokens: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
        self.embeddings = OpenAIEmbeddings()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vectorstore = None

    def split_into_sentences(self, text: str) -> List[str]:
        # Very basic splitting; can be improved later
        sentences = text.split(". ")
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        return self.model.encode(sentences)

    def greedy_merge(self, sentences: List[str], embeddings: np.ndarray) -> List[str]:
        merged_chunks = []
        used = [False] * len(sentences)

        for i in range(len(sentences)):
            if used[i]:
                continue
            current_chunk = [sentences[i]]
            current_chunk_len = len(sentences[i].split())
            used[i] = True

            for j in range(i + 1, len(sentences)):
                if used[j]:
                    continue
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]
                new_chunk_len = current_chunk_len + len(sentences[j].split())

                if sim >= self.similarity_threshold and new_chunk_len <= self.max_tokens:
                    current_chunk.append(sentences[j])
                    used[j] = True
                    current_chunk_len = new_chunk_len

            merged_chunks.append(" ".join(current_chunk))

        return merged_chunks

    def chunk_document(self, text: str) -> List[str]:
        sentences = self.split_into_sentences(text)
        embeddings = self.embed_sentences(sentences)
        chunks = self.greedy_merge(sentences, embeddings)
        return chunks

    def process_documents(self, documents: List[str]) -> List[Dict[str, str]]:
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            for chunk in chunks:
                all_chunks.append({"content": chunk})
        return all_chunks

    def save_chunks(self, chunks: List[Dict[str, str]], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    def chunk_documents(self, documents: list[str]) -> list[str]:
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk_document(doc))
        return chunks
    
    def build_vectorstore(self, chunks: list[str]):
        self.vectorstore = FAISS.from_texts(chunks, self.embeddings)

    def retrieve(self, query: str):
        """Simple retrieval using similarity search."""
        docs = self.vectorstore.similarity_search(query, k=4)
        return [{"content": doc.page_content} for doc in docs]
