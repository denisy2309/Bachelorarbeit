import faiss

class DenseRetriever:
    def __init__(self, embeddings):
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return indices[0], distances[0]