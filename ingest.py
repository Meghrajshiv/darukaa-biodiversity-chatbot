import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_DIR = "data/pdfs"
CHROMA_DIR = "data/chroma_store"

def load_pdfs():
    docs = []
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(PDF_DIR, filename)
            try:
                loader = PyPDFLoader(path)
                loaded = loader.load()
                docs.extend(loaded)
                print(f"Loaded: {filename} ({len(loaded)} pages)")
            except Exception as e:
                print(f"SKIPPED (error): {filename} — {type(e).__name__}: {e}")
    return docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    batch_size = 200
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        vectordb.add_documents(batch)
        print(f"Embedded {min(i + batch_size, total)} / {total} chunks")

    vectordb.persist()
    print(f"Stored {total} chunks in Chroma at {CHROMA_DIR}")
    return vectordb

if __name__ == "__main__":
    print("Loading PDFs...")
    raw_docs = load_pdfs()
    print(f"Loaded {len(raw_docs)} pages total.")

    print("Splitting into chunks...")
    chunks = chunk_docs(raw_docs)
    print(f"Created {len(chunks)} chunks.")

    print("Building vector store...")
    build_vector_store(chunks)

    print("Done. Your knowledge base is ready.")