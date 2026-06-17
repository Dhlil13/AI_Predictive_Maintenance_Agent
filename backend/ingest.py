import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "..", "vectorstore")

# 👉 embeddings fake (pas de torch)
embeddings = FakeEmbeddings(size=384)

def create_vectorstore():
    print("📂 Creating vector store...")

    all_chunks = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(DATA_DIR, filename)
            print(f"📄 Processing {filename}")

            loader = PyPDFLoader(path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            chunks = splitter.split_documents(documents)
            all_chunks.extend(chunks)

    if not all_chunks:
        print("⚠️ No documents found.")
        return

    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)

    print("✅ Vector store created successfully!")

if __name__ == "__main__":
    create_vectorstore()