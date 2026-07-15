import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

project = os.getenv("GOOGLE_CLOUD_PROJECT")
region = os.getenv("GOOGLE_CLOUD_REGION")


def build_vector_store(resume_folder: str = "./resumes", force_rebuild: bool = True):
    print("📄 Loading resumes...")
    docs = []

    from langchain_community.document_loaders import PyPDFLoader

    file_count = 0

    for file in Path(resume_folder).iterdir():
        if file.suffix.lower() == ".txt":
            print(f"  Loading TXT: {file.name}")
            loader = TextLoader(str(file), encoding="utf-8")
            loaded = loader.load()

            # Extract once from the full file text, then apply to every doc/chunk
            full_text = "\n".join(doc.page_content for doc in loaded)
            github_username = extract_github_username(full_text)

            for doc in loaded:
                doc.metadata["source"] = file.name
                doc.metadata["github_username"] = github_username

            docs.extend(loaded)
            file_count += 1

        elif file.suffix.lower() == ".pdf":
            print(f"  Loading PDF: {file.name}")
            loader = PyPDFLoader(str(file))
            loaded = loader.load()

            # Extract once from the full file text, then apply to every doc/chunk
            full_text = "\n".join(doc.page_content for doc in loaded)
            github_username = extract_github_username(full_text)

            for doc in loaded:
                doc.metadata["source"] = file.name
                doc.metadata["github_username"] = github_username

            docs.extend(loaded)
            file_count += 1

        else:
            print(f"  Skipping unsupported file: {file.name}")

    print(f"✅ Loaded {file_count} file(s) → {len(docs)} document chunk(s)")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks")

    # Delete old chroma_db if force_rebuild
    if force_rebuild and Path("./chroma_db").exists():
        print("🗑️  Deleting old vector store...")
        shutil.rmtree("./chroma_db")
        print("✅ Old vector store deleted")

    print("🔢 Generating embeddings with Vertex AI...")
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",  # type: ignore
        project=project,
        location=region,
    )

    print("💾 Storing in vector database...")
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory="./chroma_db"
    )

    print("✅ Vector store built successfully!")
    return vectorstore


def load_vector_store():
    """Load existing vector store without rebuilding."""
    print("📂 Loading existing vector store...")
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",  # type: ignore
        project=project,
        location=region,
    )
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    print("✅ Vector store loaded!")
    return vectorstore


def search_candidates(query: str, vectorstore, k: int = 6):
    print(f"\n🔍 Searching for: '{query}'")
    results = vectorstore.similarity_search(query, k=k)

    # Deduplicate by filename only (strip folder path)
    seen = set()
    unique_results = []
    for doc in results:
        source = Path(doc.metadata.get("source", "unknown")).name
        if source not in seen:
            seen.add(source)
            unique_results.append(doc)

    print(f"✅ Found {len(unique_results)} unique candidate(s)\n")
    for i, doc in enumerate(unique_results):
        filename = Path(doc.metadata.get("source", "unknown")).name
        print(f"--- Candidate {i + 1}: {filename} ---")
        print(doc.page_content)
        print()

    return unique_results


import re
from typing import Optional


def extract_github_username(resume_text: str) -> Optional[str]:
    match = re.search(r"github\.com/([\w-]+)", resume_text)
    return match.group(1) if match else None


if __name__ == "__main__":
    vs = build_vector_store(force_rebuild=True)
    search_candidates("Who has more than 10 years of experience?", vs)
    search_candidates("Who has GCP and AI experience?", vs)
