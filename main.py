# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 21:34:09 2025

@author: 000010478
"""

import os
import argparse
import numpy as np
import pandas as pd
import faiss
from openai import AzureOpenAI
from clases_textos import collect_files, chunk_text, extract_text_by_ext
import json

# ================== Azure OpenAI Configuration ==================
# API key is hardcoded as requested by the user
AZURE_API_KEY = "5FhrNYbHwABvRRYQ9kHzuJaAHtZRWJ0Ke1vpXGFfT0vFf7Wu6pqwJQQJ99BHACHYHv6XJ3w3AAABACOGvHwe"

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version='2024-12-01-preview',
    azure_endpoint='https://pnl-maestria.openai.azure.com/'
)

# ================== Model Configuration ==================
def load_config(config_path='config.json'):
    """Loads model configurations from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def get_embedding(text: str, model_deployment: str) -> np.ndarray:
    """Obtiene el embedding de un texto."""
    resp = client.embeddings.create(input=[text], model=model_deployment)
    emb = np.array(resp.data[0].embedding, dtype="float32")
    emb = np.ascontiguousarray(emb.reshape(1, -1))
    faiss.normalize_L2(emb)
    return emb

# ================== Indexing Logic ==================
def create_index(docs_folder: str, chunk_size: int, overlap: int, embedding_deployment: str):
    """Crea y guarda un índice FAISS y los chunks de texto."""
    chunks = []
    embeddings = []

    for file in collect_files(docs_folder):
        text = extract_text_by_ext(file)
        chunked_text = chunk_text(text, chunk_size, overlap)
        chunks.extend(chunked_text)

        response = client.embeddings.create(input=chunked_text, model=embedding_deployment)
        embs = [d.embedding for d in response.data]
        embeddings.extend(embs)

    data = pd.DataFrame(data=chunks, columns=["text"])

    d = len(embeddings[0])
    embeddings_np = np.ascontiguousarray(np.array(embeddings).astype('float32'))
    faiss.normalize_L2(embeddings_np)

    index = faiss.IndexFlatIP(d)
    index.add(embeddings_np)

    faiss.write_index(index, "faiss_index.faiss")
    data.to_parquet('chunks.parquet')
    print("Indexing complete. Index and chunks saved.")

# ================== Retrieval Logic ==================
def search(query: str, k: int, embedding_deployment: str, use_mmr: bool = False, lambda_val: float = 0.5):
    """
    Busca los k chunks más relevantes para una consulta.
    Incluye una opción para usar Maximal Marginal Relevance (MMR) para diversificar los resultados.
    """
    index = faiss.read_index("faiss_index.faiss")
    df = pd.read_parquet('chunks.parquet')

    query_embedding = get_embedding(query, embedding_deployment)

    # Si se usa MMR, se recuperan más documentos para tener un mejor pool de candidatos.
    search_k = k * 4 if use_mmr else k

    D, I = index.search(query_embedding, search_k)

    if not use_mmr:
        results = df.iloc[I[0]].copy()
        results["cosine_sim"] = D[0]
        results = results.sort_values("cosine_sim", ascending=False).reset_index(drop=True)
        return results
    else:
        # Lógica de MMR
        candidate_indices = I[0]
        candidate_embeddings = np.array([index.reconstruct(int(i)) for i in candidate_indices])
        faiss.normalize_L2(candidate_embeddings)

        query_similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()
        inter_candidate_similarities = np.dot(candidate_embeddings, candidate_embeddings.T)

        selected_indices = []
        remaining_indices = list(range(len(candidate_indices)))

        # Seleccionar el primer documento (el más relevante)
        best_initial_idx = np.argmax(query_similarities)
        selected_indices.append(remaining_indices.pop(best_initial_idx))

        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []
            for i in remaining_indices:
                relevance = query_similarities[i]
                max_similarity_with_selected = np.max(inter_candidate_similarities[i, selected_indices])
                mmr_score = lambda_val * relevance - (1 - lambda_val) * max_similarity_with_selected
                mmr_scores.append(mmr_score)

            best_mmr_idx = np.argmax(mmr_scores)
            selected_index_in_remaining = remaining_indices.pop(best_mmr_idx)
            selected_indices.append(selected_index_in_remaining)

        final_faiss_indices = [candidate_indices[i] for i in selected_indices]
        final_scores = [query_similarities[i] for i in selected_indices]

        results = df.iloc[final_faiss_indices].copy()
        results["cosine_sim"] = final_scores
        results = results.reset_index(drop=True)
        return results

# ================== RAG Query Logic ==================
def ask_question(query: str, model_config: dict):
    """Realiza una pregunta al modelo RAG."""
    use_mmr = model_config.get('use_mmr', False)
    lambda_val = model_config.get('lambda', 0.5)

    results_df = search(
        query,
        k=model_config['k'],
        embedding_deployment=model_config['embedding_deployment'],
        use_mmr=use_mmr,
        lambda_val=lambda_val
    )

    context = "\n\n".join(results_df['text'].values)
    prompt = f"Contexto:\n{context}\n\nPregunta: {query}\nRespuesta:"

    response = client.chat.completions.create(
        model=model_config['chat_deployment'],
        messages=[
            {"role": "system", "content": "You are an expert in aircraft accidents. Respond only in English."},
            {"role": "user", "content": prompt}
        ]
    )
    print(response.choices[0].message.content)

# ================== Main Execution ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema RAG para consulta de documentos.")
    parser.add_argument("action", choices=['index', 'query'], help="La acción a realizar: 'index' para crear el índice o 'query' para hacer una consulta.")
    parser.add_argument("--model", type=str, default="baseline", help="El nombre del modelo a utilizar (definido en config.json).")
    parser.add_argument("--query", type=str, help="La pregunta para el modo 'query'.")

    args = parser.parse_args()

    # Cargar configuraciones
    configs = load_config()
    if args.model not in configs:
        raise ValueError(f"Modelo '{args.model}' no encontrado en config.json.")

    model_config = configs[args.model]

    if args.action == 'index':
        create_index(
            docs_folder='./documentos/',
            chunk_size=model_config['chunk_size'],
            overlap=model_config['overlap'],
            embedding_deployment=model_config['embedding_deployment']
        )
    elif args.action == 'query':
        if not args.query:
            raise ValueError("El argumento --query es requerido para la acción 'query'.")
        ask_question(args.query, model_config)
