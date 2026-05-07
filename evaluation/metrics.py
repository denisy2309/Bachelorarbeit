def precision_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / k

def recall_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / len(relevant)

def average_precision(relevant, retrieved):
    score = 0.0
    hits = 0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            score += hits / (i + 1)

    return score / len(relevant) if relevant else 0