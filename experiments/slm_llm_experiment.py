from models.generator import Generator
from evaluation.rouge_eval import compute_rouge

from retrieval.dense import DenseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import rrf

from utils.cache import save_json, load_json


def run_slm_llm_experiment(fine_model, documents, queries, retrieval_method):
    CACHE_PATH = "results/llm_cache.json"

    cache_answers = load_json(CACHE_PATH)
    if cache_answers is None:
        cache_answers = {}

    doc_embeddings_fine = fine_model.encode(documents, convert_to_numpy=True)

    dense = DenseRetriever(doc_embeddings_fine)
    bm25 = BM25Retriever(documents)

    llm = Generator("Qwen/Qwen2.5-7B-Instruct")
    slm = Generator("Qwen/Qwen2.5-0.5B-Instruct")

    results = []

    for query in queries:
        query_id = query["_id"]
        query_text = query["text"]

        query_emb = fine_model.encode(
            [query_text],
            convert_to_numpy=True
        )[0]

        dense_ids, _ = dense.search(query_emb, k=3)
        
        if retrieval_method == "hybrid_rrf":
            bm25_ids = [doc_id for doc_id, _ in bm25.search(query_text, k=3)]

            hybrid = rrf([dense_ids, bm25_ids])

            doc_ids = [doc_id for doc_id, _ in hybrid][:3]

        elif retrieval_method == "dense":
            doc_ids = dense_ids[:3]

        else:
            raise ValueError(f"Unknown retrieval_method: {retrieval_method}") 

        context_docs = [documents[i] for i in doc_ids]
        context_text = " ".join(context_docs)

        # --- Cache-Struktur ---
        if query_id not in cache_answers:
            cache_answers[query_id] = {}

        if "ground_truth" not in cache_answers[query_id]:
            cache_answers[query_id]["ground_truth"] = {}

        if "with_context" not in cache_answers[query_id]:
            cache_answers[query_id]["with_context"] = {}

        if "without_context" not in cache_answers[query_id]:
            cache_answers[query_id]["without_context"] = {}

        context_key = context_text

        # --- Ground Truth für genau diesen Kontext ---
        if context_key not in cache_answers[query_id]["ground_truth"]:
            cache_answers[query_id]["ground_truth"][context_key] = Generator.generate_ground_truth(
                query_text,
                context_text
            )
            
        ground_truth = cache_answers[query_id]["ground_truth"][context_key]

        # --- Antworten MIT Kontext ---
        if context_key not in cache_answers[query_id]["with_context"]:
            cache_answers[query_id]["with_context"][context_key] = {}

        with_context_cache = cache_answers[query_id]["with_context"][context_key]

        if "llm_answer" not in with_context_cache:
            with_context_cache["llm_answer"] = llm.generate_with_context(
                query_text,
                context_text
            )

        if "slm_answer" not in with_context_cache:
            with_context_cache["slm_answer"] = slm.generate_with_context(
                query_text,
                context_text
            )

        answer_llm_with_context = with_context_cache["llm_answer"]
        answer_slm_with_context = with_context_cache["slm_answer"]

        # --- Antworten OHNE Kontext ---
        without_context_cache = cache_answers[query_id]["without_context"]

        if "llm_answer" not in without_context_cache:
            without_context_cache["llm_answer"] = llm.generate_without_context(
                query_text
            )

        if "slm_answer" not in without_context_cache:
            without_context_cache["slm_answer"] = slm.generate_without_context(
                query_text
            )

        answer_llm_without_context = without_context_cache["llm_answer"]
        answer_slm_without_context = without_context_cache["slm_answer"]

        save_json(cache_answers, CACHE_PATH)

        # --- ROUGE ---
        r1_llm_with, rL_llm_with = compute_rouge(
            answer_llm_with_context,
            ground_truth
        )
        r1_slm_with, rL_slm_with = compute_rouge(
            answer_slm_with_context,
            ground_truth
        )

        r1_llm_without, rL_llm_without = compute_rouge(
            answer_llm_without_context,
            ground_truth
        )
        r1_slm_without, rL_slm_without = compute_rouge(
            answer_slm_without_context,
            ground_truth
        )

        results.append({
            "query": query_text,

            "llm_with_context_rouge1": r1_llm_with,
            "llm_with_context_rougeL": rL_llm_with,
            "slm_with_context_rouge1": r1_slm_with,
            "slm_with_context_rougeL": rL_slm_with,

            "llm_without_context_rouge1": r1_llm_without,
            "llm_without_context_rougeL": rL_llm_without,
            "slm_without_context_rouge1": r1_slm_without,
            "slm_without_context_rougeL": rL_slm_without,
        })

    # --- Average Scores ---
    avg = {
        "llm_with_context_rouge1": sum(r["llm_with_context_rouge1"] for r in results) / len(results),
        "llm_with_context_rougeL": sum(r["llm_with_context_rougeL"] for r in results) / len(results),
        "slm_with_context_rouge1": sum(r["slm_with_context_rouge1"] for r in results) / len(results),
        "slm_with_context_rougeL": sum(r["slm_with_context_rougeL"] for r in results) / len(results),

        "llm_without_context_rouge1": sum(r["llm_without_context_rouge1"] for r in results) / len(results),
        "llm_without_context_rougeL": sum(r["llm_without_context_rougeL"] for r in results) / len(results),
        "slm_without_context_rouge1": sum(r["slm_without_context_rouge1"] for r in results) / len(results),
        "slm_without_context_rougeL": sum(r["slm_without_context_rougeL"] for r in results) / len(results),
    }

    print("\n==============================")
    print("SLM vs. LLM RESULTS")
    print("==============================")

    print(
        f"LLM without context: "
        f"ROUGE-1={avg['llm_without_context_rouge1']:.3f}, "
        f"ROUGE-L={avg['llm_without_context_rougeL']:.3f}"
    )

    print(
        f"SLM without context: "
        f"ROUGE-1={avg['slm_without_context_rouge1']:.3f}, "
        f"ROUGE-L={avg['slm_without_context_rougeL']:.3f}"
    )

    print(
        f"LLM with context: "
        f"ROUGE-1={avg['llm_with_context_rouge1']:.3f}, "
        f"ROUGE-L={avg['llm_with_context_rougeL']:.3f}"
    )

    print(
        f"SLM with context: "
        f"ROUGE-1={avg['slm_with_context_rouge1']:.3f}, "
        f"ROUGE-L={avg['slm_with_context_rougeL']:.3f}"
    )

    return {
        "results": results,
        "averages": avg
    }