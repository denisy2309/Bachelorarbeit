import os
os.environ["HF_HOME"] = "D:/huggingface_cache"
from models.embedding import EmbeddingModel
from retrieval.dense import DenseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import rrf
from evaluation.metrics import precision_at_k, recall_at_k, average_precision
# from models.fine_tune import FineTuner
from experiments.slm_llm_experiment import run_slm_llm_experiment
from data.trec_covid import load_trec_covid
from data.qrels import build_qrels_map

queries_ds, corpus, qrels = load_trec_covid()

queries_subset = queries_ds.select(range(10))

documents = [doc["text"] for doc in corpus]
doc_ids = [doc["_id"] for doc in corpus]

doc_id_map = {doc_id: i for i, doc_id in enumerate(doc_ids)}

qrels_map = build_qrels_map(qrels, doc_id_map)


""""
# Fine-Tuning
fine_tuner = FineTuner()
train_examples = fine_tuner.prepare_data(train_pairs)

fine_tuner.train(train_examples, epochs=1)
fine_tuner.save("fine_tuned_model")

# load fine-tuned model
fine_model = fine_tuner.get_model()
"""

# Embeddings
# Pre-trained
embedder = EmbeddingModel()
doc_embeddings_pre = embedder.encode(documents)

"""
# Fine-tuned
doc_embeddings_fine = fine_model.encode(documents)
"""

# Dense
dense_pre = DenseRetriever(doc_embeddings_pre)
# dense_fine = DenseRetriever(doc_embeddings_fine)

# BM25
bm25 = BM25Retriever(documents)

for query in queries_subset:
    query_id = query["_id"]
    query_text = query["text"]

    relevant = qrels_map.get(query_id, [])

    query_emb = embedder.encode([query_text])[0]

    dense_ids_pre, _ = dense_pre.search(query_emb)
    # dense_ids_fine, _ = dense_fine.search(fine_model.encode([query_text])[0])
    bm25_results = bm25.search(query_text)
    bm25_ids = [doc_id for doc_id, _ in bm25_results]

    hybrid_results_pre = rrf([dense_ids_pre, bm25_ids])
    hybrid_ids_pre = [doc_id for doc_id, _ in hybrid_results_pre]

    # hybrid_results_fine = rrf([dense_ids_fine, bm25_ids])
    # hybrid_ids_fine = [doc_id for doc_id, _ in hybrid_results_fine]

    # --- Evaluation ---
    print(f"\nQuery: {query_text}")

    print("Dense_Pre-trained:", dense_ids_pre)
    # print("Dense_Fine-tuned:", dense_ids_fine)
    print("BM25:", bm25_ids)
    print("Hybrid_Pre-trained:", hybrid_results_pre)
    # print("Hybrid_Fine-tuned:", hybrid_results_fine)

    for name, results in {
        "Dense_Pre-trained": dense_ids_pre,
        # "Dense_Fine-tuned": dense_ids_fine,
        "BM25": bm25_ids,
        "Hybrid_Pre-trained": hybrid_ids_pre,
        # "Hybrid_Fine-tuned": hybrid_ids_fine,
    }.items():
        p = precision_at_k(relevant, results, k=3)
        r = recall_at_k(relevant, results, k=3)
        ap = average_precision(relevant, results)

        print(f"{name}: Precision@3={p:.2f}, Recall@3={r:.2f}, AP={ap:.2f}")

# --- SLM vs LLM Experiment ---
print("\n--- SLM vs LLM Experiment ---")
run_slm_llm_experiment(documents, queries_subset)