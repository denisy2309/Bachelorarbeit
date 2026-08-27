import random

def build_contrastive_pairs(queries, corpus, qrels):
    query_map = {q["_id"]: q["text"] for q in queries}
    corpus_map = {doc["_id"]: doc["text"] for doc in corpus}

    pairs = []

    for query_id in query_map:

        positives = []
        negatives = []

        for row in qrels:

            if row["query-id"] != query_id:
                continue

            doc_id = row["corpus-id"]

            if doc_id not in corpus_map:
                continue

            if row["score"] == 2:
                positives.append(doc_id)

            elif row["score"] == 0:
                negatives.append(doc_id)

        sampled_negatives = random.sample(
            negatives,
            min(len(negatives), 5 * len(positives))
        )

        query_text = query_map[query_id]

        for doc_id in positives:
            pairs.append(
                (query_text, corpus_map[doc_id], 1.0)
            )

        for doc_id in sampled_negatives:
            pairs.append(
                (query_text, corpus_map[doc_id], 0.0)
            )

    return pairs

def build_mnrl_pairs(queries, corpus, qrels):
    query_map = {q["_id"]: q["text"] for q in queries}
    corpus_map = {doc["_id"]: doc["text"] for doc in corpus}

    pairs = []

    for row in qrels:
        query_id = row["query-id"]
        doc_id = row["corpus-id"]

        if query_id not in query_map:
            continue

        if doc_id not in corpus_map:
            continue

        pairs.append((
            query_map[query_id],
            corpus_map[doc_id]
        ))

    return pairs