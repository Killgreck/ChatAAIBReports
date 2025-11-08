# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 13:00:00 2025

@author: Jules
"""

import argparse
import pandas as pd
from main import search, load_config

import json

# ================== Evaluation Dataset ==================
def load_ground_truth(path='ground_truth.json'):
    """Loads the ground truth dataset from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

# ================== Evaluation Logic ==================
def calculate_recall_at_k(retrieved_chunks: list, ground_truth_chunks: list) -> float:
    """Calcula el recall@k."""
    retrieved_set = set(retrieved_chunks)
    ground_truth_set = set(ground_truth_chunks)

    relevant_retrieved = retrieved_set.intersection(ground_truth_set)

    if not ground_truth_set:
        return 1.0 if not retrieved_set else 0.0

    recall = len(relevant_retrieved) / len(ground_truth_set)
    return recall

def generate_ground_truth(model_name: str):
    """Genera un archivo ground_truth.json a partir de un conjunto de preguntas."""
    questions = [
        "What caused the PW Orca aircraft to crash?",
        "What were the safety actions taken after the Uvify IFO incident?"
    ]

    ground_truth_data = []

    configs = load_config()
    if model_name not in configs:
        raise ValueError(f"Modelo '{model_name}' no encontrado en config.json.")

    model_config = configs[model_name]

    for question in questions:
        results_df = search(
            query=question,
            k=model_config['k'],
            embedding_deployment=model_config['embedding_deployment'],
            use_mmr=model_config.get('use_mmr', False),
            lambda_val=model_config.get('lambda', 0.5)
        )

        ground_truth_chunks = results_df['text'].tolist()

        ground_truth_data.append({
            "question": question,
            "ground_truth_chunks": ground_truth_chunks
        })

    with open('ground_truth.json', 'w') as f:
        json.dump(ground_truth_data, f, indent=4)

    print(f"ground_truth.json ha sido generado exitosamente usando el modelo '{model_name}'.")

def evaluate_model(model_name: str):
    """Evalúa un modelo RAG utilizando el dataset de ejemplo."""
    configs = load_config()
    if model_name not in configs:
        raise ValueError(f"Modelo '{model_name}' no encontrado en config.json.")

    model_config = configs[model_name]

    total_recall = 0

    print(f"--- Evaluating Model: {model_name} ---")

    evaluation_dataset = load_ground_truth()

    for item in evaluation_dataset:
        question = item["question"]
        ground_truth = item["ground_truth_chunks"]

        # Realizar la búsqueda con la configuración del modelo
        results_df = search(
            query=question,
            k=model_config['k'],
            embedding_deployment=model_config['embedding_deployment'],
            use_mmr=model_config.get('use_mmr', False),
            lambda_val=model_config.get('lambda', 0.5)
        )

        retrieved_chunks = results_df['text'].tolist()

        # Calcular recall@k
        recall = calculate_recall_at_k(retrieved_chunks, ground_truth)
        total_recall += recall

        print(f"Question: '{question}'")
        print(f"Recall@{model_config['k']}: {recall:.4f}\n")

    average_recall = total_recall / len(evaluation_dataset)
    print(f"--- Average Recall@{model_config['k']} for {model_name}: {average_recall:.4f} ---")

# ================== Main Execution ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación de modelos RAG.")
    parser.add_argument("--model", type=str, default="baseline", help="El nombre del modelo a utilizar (definido en config.json).")
    parser.add_argument("--generate-ground-truth", action="store_true", help="Genera un nuevo archivo ground_truth.json.")

    args = parser.parse_args()

    if args.generate_ground_truth:
        generate_ground_truth(args.model)
    else:
        evaluate_model(args.model)
