from statistics import mean
from sklearn.model_selection import KFold

from models.embedding import EmbeddingModel
from models.fine_tune import FineTuner

from retrieval.dense import DenseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import (
    rrf,
    fixed_weighted_fusion,
    max_score_fusion,
    dynamic_weighted_fusion,
    DynamicAlphaTuner
)

from evaluation.metrics import precision_at_k, recall_at_k, average_precision

from experiments.slm_llm_experiment import run_slm_llm_experiment

from data.trec_covid import load_trec_covid
from data.nfcorpus import load_nfcorpus
from data.qrels import build_qrels_map
from data.train_pairs import build_contrastive_pairs, build_mnrl_pairs


def evaluate_dataset(dataset_name, queries, corpus, qrels_eval, min_score, fine_model, embedder, documents, dense_pre, bm25, dat_tuner, k):
    print(f"\n===== Evaluation: {dataset_name} =====")

    doc_ids = [doc["_id"] for doc in corpus]
    doc_id_map = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    qrels_map = build_qrels_map(qrels_eval, doc_id_map, min_score=min_score)

    doc_embeddings_fine = fine_model.encode(
        documents,
        convert_to_numpy=True
    )

    dense_fine = DenseRetriever(doc_embeddings_fine)

    results_all = {
        "Dense_Pre-trained": [],
        "Dense_Fine-tuned": [],
        "BM25": [],

        "RRF_Hybrid_Pre-trained": [],
        "RRF_Hybrid_Fine-tuned": [],

        "FixedWeighted_Hybrid_Pre-trained": [],
        "FixedWeighted_Hybrid_Fine-tuned": [],

        "MaxScore_Hybrid_Pre-trained": [],
        "MaxScore_Hybrid_Fine-tuned": [],

        "DAT_Hybrid_Pre-trained": [],
        "DAT_Hybrid_Fine-tuned": [],
    }

    for query in queries:
        query_id = query["_id"]
        query_text = query["text"]

        relevant = qrels_map.get(query_id, [])

        query_emb_pre = embedder.encode([query_text])[0]
        query_emb_fine = fine_model.encode(
            [query_text],
            convert_to_numpy=True
        )[0]

        dense_ids_pre, dense_scores_pre = dense_pre.search(
            query_emb_pre
        )

        dense_ids_fine, dense_scores_fine = dense_fine.search(
            query_emb_fine
        )

        bm25_results = bm25.search(query_text)
        bm25_ids = [
            doc_id
            for doc_id, _ in bm25_results
        ]

        # -----------------------------------
        # RRF
        # -----------------------------------
        rrf_results_pre = rrf([
            dense_ids_pre,
            bm25_ids
        ])

        rrf_ids_pre = [
            doc_id
            for doc_id, _ in rrf_results_pre
        ]

        rrf_results_fine = rrf([
            dense_ids_fine,
            bm25_ids
        ])

        rrf_ids_fine = [
            doc_id
            for doc_id, _ in rrf_results_fine
        ]

        # -----------------------------------
        # Fixed weighted fusion:
        # Dense = 0.7, BM25 = 0.3
        # -----------------------------------
        fixed_results_pre = fixed_weighted_fusion(
            dense_ids=dense_ids_pre,
            dense_scores=dense_scores_pre,
            bm25_results=bm25_results,
            dense_weight=0.7,
            bm25_weight=0.3
        )

        fixed_ids_pre = [
            doc_id
            for doc_id, _ in fixed_results_pre
        ]

        fixed_results_fine = fixed_weighted_fusion(
            dense_ids=dense_ids_fine,
            dense_scores=dense_scores_fine,
            bm25_results=bm25_results,
            dense_weight=0.7,
            bm25_weight=0.3
        )

        fixed_ids_fine = [
            doc_id
            for doc_id, _ in fixed_results_fine
        ]

        # -----------------------------------
        # Max-score fusion
        # -----------------------------------
        max_results_pre = max_score_fusion(
            dense_ids=dense_ids_pre,
            dense_scores=dense_scores_pre,
            bm25_results=bm25_results
        )

        max_ids_pre = [
            doc_id
            for doc_id, _ in max_results_pre
        ]

        max_results_fine = max_score_fusion(
            dense_ids=dense_ids_fine,
            dense_scores=dense_scores_fine,
            bm25_results=bm25_results
        )

        max_ids_fine = [
            doc_id
            for doc_id, _ in max_results_fine
        ]

        # -----------------------------------
        # DAT
        # Predict alpha once per query.
        # The same alpha is used for the
        # pre-trained and fine-tuned variants.
        # -----------------------------------
        dat_dense_weight = dat_tuner.predict_dense_weight(
            query_text
        )

        dat_results_pre = dynamic_weighted_fusion(
            dense_ids=dense_ids_pre,
            dense_scores=dense_scores_pre,
            bm25_results=bm25_results,
            dense_weight=dat_dense_weight
        )

        dat_ids_pre = [
            doc_id
            for doc_id, _ in dat_results_pre
        ]

        dat_results_fine = dynamic_weighted_fusion(
            dense_ids=dense_ids_fine,
            dense_scores=dense_scores_fine,
            bm25_results=bm25_results,
            dense_weight=dat_dense_weight
        )

        dat_ids_fine = [
            doc_id
            for doc_id, _ in dat_results_fine
        ]

        for name, retrieved in {
            "Dense_Pre-trained": dense_ids_pre,
            "Dense_Fine-tuned": dense_ids_fine,
            "BM25": bm25_ids,

            "RRF_Hybrid_Pre-trained": rrf_ids_pre,
            "RRF_Hybrid_Fine-tuned": rrf_ids_fine,

            "FixedWeighted_Hybrid_Pre-trained": fixed_ids_pre,
            "FixedWeighted_Hybrid_Fine-tuned": fixed_ids_fine,

            "MaxScore_Hybrid_Pre-trained": max_ids_pre,
            "MaxScore_Hybrid_Fine-tuned": max_ids_fine,

            "DAT_Hybrid_Pre-trained": dat_ids_pre,
            "DAT_Hybrid_Fine-tuned": dat_ids_fine,
        }.items():
            p = precision_at_k(relevant, retrieved, k=k)
            r = recall_at_k(relevant, retrieved, k=k)
            ap = average_precision(relevant, retrieved)

            results_all[name].append({
                "p": p,
                "r": r,
                "ap": ap
            })

    print(f"\n--- {dataset_name} Results ---")

    for method, metrics in results_all.items():
        avg_p = mean([m["p"] for m in metrics])
        avg_r = mean([m["r"] for m in metrics])
        avg_ap = mean([m["ap"] for m in metrics])

        print(
            f"{method}: "
            f"Precision@{k}={avg_p:.3f}, "
            f"Recall@{k}={avg_r:.3f}, "
            f"MAP={avg_ap:.3f}"
        )

    return results_all


def run_trec_covid():
    print("\n==============================")
    print("RUNNING TREC-COVID")
    print("==============================")

    queries_ds, corpus, qrels = load_trec_covid()

    documents = [doc["text"] for doc in corpus]

    embedder = EmbeddingModel()
    doc_embeddings_pre = embedder.encode(documents)

    dense_pre = DenseRetriever(doc_embeddings_pre)
    bm25 = BM25Retriever(documents)

    dat_tuner = DynamicAlphaTuner(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        default_dense_weight=0.7
    )

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    query_indices = list(range(len(queries_ds)))

    all_fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(query_indices)):
        print(f"\n===== TREC-COVID Fold {fold + 1} =====")

        train_queries = queries_ds.select(train_idx)
        test_queries = queries_ds.select(test_idx)

        train_pairs = build_contrastive_pairs(
            train_queries,
            corpus,
            qrels
        )

        fine_tuner = FineTuner()
        train_dataset = fine_tuner.prepare_contrastive_data(train_pairs)

        fine_tuner.train(
            train_dataset,
            run_name=f"trec_covid_fold_{fold + 1}",
            loss_type="contrastive",
            epochs=1,
            batch_size=32
        )

        fine_model = fine_tuner.get_model()

        fold_results = evaluate_dataset(
            dataset_name=f"TREC-COVID Fold {fold + 1}",
            queries=test_queries,
            corpus=corpus,
            qrels_eval=qrels,
            min_score=2,
            fine_model=fine_model,
            embedder=embedder,
            documents=documents,
            dense_pre=dense_pre,
            bm25=bm25,
            dat_tuner=dat_tuner,
            k=50
        )

        all_fold_results.append(fold_results)

    print("\n===== FINAL CROSS-VALIDATION RESULTS =====")

    methods = all_fold_results[0].keys()

    for method in methods:

        avg_p = mean(
            mean(m["p"] for m in fold[method])
            for fold in all_fold_results
        )

        avg_r = mean(
            mean(m["r"] for m in fold[method])
            for fold in all_fold_results
        )

        avg_ap = mean(
            mean(m["ap"] for m in fold[method])
            for fold in all_fold_results
        )

        print(
            f"{method}: "
            f"Precision@50={avg_p:.3f}, "
            f"Recall@50={avg_r:.3f}, "
            f"MAP={avg_ap:.3f}"
        )

    print("\n===== TREC-COVID SLM vs LLM =====")

    held_out_queries = queries_ds.select(range(40, 50))
    final_train_queries = queries_ds.select(range(40))

    final_train_pairs = build_contrastive_pairs(
        final_train_queries,
        corpus,
        qrels
    )

    final_tuner = FineTuner()
    final_dataset = final_tuner.prepare_contrastive_data(final_train_pairs)

    final_tuner.train(
        final_dataset,
        run_name="trec_covid_slm_llm_model",
        loss_type="contrastive",
        epochs=1,
        batch_size=32
    )

    final_model = final_tuner.get_model()

    run_slm_llm_experiment(
        final_model,
        documents,
        held_out_queries,
        retrieval_method="hybrid_rrf"
    )


def run_nfcorpus():
    print("\n==============================")
    print("RUNNING NFCORPUS")
    print("==============================")

    queries, corpus, qrels_train, qrels_dev, qrels_test = load_nfcorpus()

    train_query_ids = set(row["query-id"] for row in qrels_train)
    dev_query_ids = set(row["query-id"] for row in qrels_dev)
    test_query_ids = set(row["query-id"] for row in qrels_test)

    train_queries = queries.filter(lambda q: q["_id"] in train_query_ids)
    dev_queries = queries.filter(lambda q: q["_id"] in dev_query_ids)
    test_queries = queries.filter(lambda q: q["_id"] in test_query_ids)

    documents = [doc["text"] for doc in corpus]

    embedder = EmbeddingModel()
    doc_embeddings_pre = embedder.encode(documents)

    dense_pre = DenseRetriever(doc_embeddings_pre)
    bm25 = BM25Retriever(documents)

    dat_tuner = DynamicAlphaTuner(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        default_dense_weight=0.7
    )

    train_pairs = build_mnrl_pairs(
        train_queries,
        corpus,
        qrels_train
    )

    fine_tuner = FineTuner()
    train_dataset = fine_tuner.prepare_mnrl_data(train_pairs)

    fine_tuner.train(
        train_dataset,
        run_name="nfcorpus_mnrl",
        loss_type="mnrl",
        epochs=3,
        batch_size=8
    )

    fine_model = fine_tuner.get_model()

    evaluate_dataset(
        dataset_name="NFCorpus Test",
        queries=test_queries,
        corpus=corpus,
        qrels_eval=qrels_test,
        min_score=1,
        fine_model=fine_model,
        embedder=embedder,
        documents=documents,
        dense_pre=dense_pre,
        bm25=bm25,
        dat_tuner=dat_tuner,
        k=10
    )

    print("\n===== NFCorpus SLM vs LLM =====")

    run_slm_llm_experiment(
        fine_model,
        documents,
        test_queries,
        retrieval_method="dense"
    )


if __name__ == "__main__":
    run_trec_covid()
    run_nfcorpus()