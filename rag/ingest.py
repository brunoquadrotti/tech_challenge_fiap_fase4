from dotenv import load_dotenv
load_dotenv()

import os
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Configurações
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", ".chroma")


def ingest_documents():
    """
    Carrega PDFs da pasta rag/documents/, divide em chunks,
    gera embeddings e persiste no ChromaDB local.
    """
    print("[RAG Ingest] Iniciando ingestão de documentos...")

    # Buscar todos os PDFs na pasta
    pdf_pattern = os.path.join(DOCUMENTS_DIR, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)

    if not pdf_files:
        print(f"[RAG Ingest] Nenhum PDF encontrado em: {DOCUMENTS_DIR}")
        return

    print(f"[RAG Ingest] {len(pdf_files)} PDF(s) encontrado(s).")

    # Carregar documentos
    all_documents = []
    for pdf_path in pdf_files:
        print(f"[RAG Ingest] Carregando: {os.path.basename(pdf_path)}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)

    print(f"[RAG Ingest] Total de páginas carregadas: {len(all_documents)}")

    # Dividir em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"[RAG Ingest] Total de chunks gerados: {len(chunks)}")

    # Gerar embeddings e persistir no ChromaDB
    print("[RAG Ingest] Gerando embeddings e salvando no ChromaDB...")
    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    print(f"[RAG Ingest] Banco vetorial salvo em: {CHROMA_PERSIST_DIR}")
    print("[RAG Ingest] Ingestão concluída com sucesso.")

    return vectorstore


if __name__ == "__main__":
    ingest_documents()
