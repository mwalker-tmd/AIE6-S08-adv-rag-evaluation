# configs/rag_config.py

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Embedding model instance
embedding_model = OpenAIEmbeddings()

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# Vectorstore factory method (optional, for reuse)
def build_vectorstore_from_texts(texts):
    return FAISS.from_texts(texts, embedding_model)
