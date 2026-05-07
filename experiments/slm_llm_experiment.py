from models.embedding import EmbeddingModel
from models.generator import Generator
from evaluation.rouge_eval import compute_rouge

from retrieval.dense import DenseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import rrf

from utils.cache import save_json, load_json


def run_slm_llm_experiment(documents, queries, best_retrieval="hybrid"):
    CACHE_PATH = "results/llm_cache.json"

    cache = load_json(CACHE_PATH)
    if cache is None:
        cache = {}

    # Embeddings
    # Pre-trained
    embedder = EmbeddingModel()
    doc_embeddings_pre = embedder.encode(documents)
    
    # Retriever
    dense = DenseRetriever(doc_embeddings_pre)
    bm25 = BM25Retriever(documents)

    llm = Generator("Qwen/Qwen2.5-1.5B")
    slm = Generator("Qwen/Qwen2.5-0.5B")

    results = []

    for query in queries:
        query_id = query["_id"]
        query_text = query["text"]
        query_emb = embedder.encode([query_text])[0]

        # --- Retrieval ---
        dense_ids, _ = dense.search(query_emb, k=3)
        bm25_ids = [doc_id for doc_id, _ in bm25.search(query_text, k=3)]

        if best_retrieval == "hybrid":
            hybrid = rrf([dense_ids, bm25_ids])
            doc_ids = [doc_id for doc_id, _ in hybrid][:3]
        else:
            doc_ids = dense_ids

        context_docs = [documents[i] for i in doc_ids]
        context_text = " ".join(context_docs)

        # --- Ground Truth ---
        if query_id not in cache:
            cache[query_id] = {}

        if "ground_truth" not in cache[query_id]:
            cache[query_id]["ground_truth"] = Generator.generate_ground_truth(query_text)

        ground_truth = cache[query_id]["ground_truth"]

        # --- Antworten ---
        key = query_id + "||" + context_text
        if key not in cache:
            cache[key] = {}

        if "llm_answer" not in cache[key]:
            cache[key]["llm_answer"] = llm.generate(query_text, context_text)
        answer_llm = cache[key]["llm_answer"]
        
        if "slm_answer" not in cache[key]:
            cache[key]["slm_answer"] = slm.generate(query_text, context_text)
        answer_slm = cache[key]["slm_answer"]

        save_json(cache, CACHE_PATH)

        # --- ROUGE ---
        r1_llm, rL_llm = compute_rouge(answer_llm, ground_truth)
        r1_slm, rL_slm = compute_rouge(answer_slm, ground_truth)

        results.append({
            "query": query_text,
            "llm_rouge1": r1_llm,
            "llm_rougeL": rL_llm,
            "slm_rouge1": r1_slm,
            "slm_rougeL": rL_slm
        })

        print(f"\nQuery: {query_text}")
        print("Ground Truth:", ground_truth)

        print("\nLLM answer:", answer_llm)
        print(f"LLM ROUGE-1: {r1_llm:.3f}, ROUGE-L: {rL_llm:.3f}")

        print("\nSLM answer:", answer_slm)
        print(f"SLM ROUGE-1: {r1_slm:.3f}, ROUGE-L: {rL_slm:.3f}")

    return results