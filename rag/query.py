from dotenv import load_dotenv
load_dotenv()

import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Configuração
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", ".chroma")


def search_documents(query: str, top_k: int = 3) -> list:
    """
    Busca semântica no banco vetorial ChromaDB.

    Args:
        query: Texto da consulta para busca.
        top_k: Número de documentos mais relevantes a retornar.

    Returns:
        Lista de dicionários com conteúdo e score de similaridade.
    """
    print(f"[RAG Query] Buscando: '{query}'")

    if not query or not query.strip():
        print("[RAG Query] ERRO: Query vazia recebida.")
        return []

    try:
        # Carregar banco vetorial existente
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )

        # Buscar documentos similares com score
        results = vectorstore.similarity_search_with_score(query, k=top_k)

        if not results:
            print("[RAG Query] Nenhum documento encontrado.")
            return []

        # Formatar resultado
        output = []
        for doc, score in results:
            output.append({
                "content": doc.page_content,
                "score": round(float(score), 4),
            })

        print(f"[RAG Query] {len(output)} documento(s) encontrado(s).")
        return output

    except Exception as e:
        print(f"[RAG Query] ERRO na busca: {e}")
        return []


if __name__ == "__main__":
    results = search_documents("sintomas de ansiedade em gestantes")
    for i, r in enumerate(results, 1):
        print(f"\n--- Resultado {i} (score: {r['score']}) ---")
        print(r["content"][:200])
