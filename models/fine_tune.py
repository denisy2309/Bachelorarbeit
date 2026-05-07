from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

class FineTuner:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def prepare_data(self, query_doc_pairs):
        train_examples = [
            InputExample(texts=[query, doc])
            for query, doc in query_doc_pairs
        ]
        return train_examples

    def train(self, train_examples, epochs=1, batch_size=8):
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            show_progress_bar=True
        )

    def save(self, path="fine_tuned_model"):
        self.model.save(path)

    def get_model(self):
        return self.model