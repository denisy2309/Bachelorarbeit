def build_qrels_map(qrels, doc_id_map, min_score):
    qrels_map = {}

    for row in qrels:
        score = int(row["score"])

        if score < min_score:
            continue

        qid = row["query-id"]
        doc_id = row["corpus-id"]

        if doc_id not in doc_id_map:
            continue

        doc_idx = doc_id_map[doc_id]

        if qid not in qrels_map:
            qrels_map[qid] = []

        qrels_map[qid].append(doc_idx)

    return qrels_map