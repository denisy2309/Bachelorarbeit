from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, corpus):
        tokenized = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.corpus = corpus

    def search(self, query, k=3):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]