from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document

def load_documents(folder_path: str) -> List[Document]:
    folder = Path(folder_path)
    documents = []
    for file in folder.glob("*"):
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix.lower() == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
        else:
            continue
        documents.extend(loader.load())
    return documents

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = []
    for doc in documents:
        split_docs.extend(splitter.split_documents([doc]))
    return split_docs
