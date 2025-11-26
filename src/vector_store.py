from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from typing import List

def create_vectorstore(documents: List[Document], embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore
