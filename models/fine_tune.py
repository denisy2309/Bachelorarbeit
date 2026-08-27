import math

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from datasets import Dataset
import wandb

class FineTuner:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def prepare_contrastive_data(self, train_pairs):
        return Dataset.from_dict({
            "sentence1": [q for q, _, _ in train_pairs],
            "sentence2": [d for _, d, _ in train_pairs],
            "label": [float(s) for _, _, s in train_pairs]
        })
    
    def prepare_mnrl_data(self, train_pairs):
        return Dataset.from_dict({
            "anchor": [q for q, _ in train_pairs],
            "positive": [d for _, d in train_pairs]
        })

    def train(self, train_dataset, run_name, loss_type, epochs, batch_size):
        if wandb.run is not None:
            wandb.finish()

        wandb.init(
            project="Bachelorarbeit",
            name=run_name
        )

        if loss_type == "contrastive":
            loss = losses.ContrastiveLoss(self.model)
            batch_sampler = BatchSamplers.BATCH_SAMPLER

        elif loss_type == "mnrl":
            loss = losses.MultipleNegativesRankingLoss(self.model)
            batch_sampler = BatchSamplers.NO_DUPLICATES

        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        steps_per_epoch = math.ceil(len(train_dataset) / batch_size)

        args = SentenceTransformerTrainingArguments(
            output_dir="checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            report_to="wandb",
            run_name=run_name,
            disable_tqdm=True,
            logging_strategy="steps",
            logging_steps=max(1, int(steps_per_epoch * epochs / 20)),
            batch_sampler=batch_sampler
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            loss=loss
        )

        trainer.train()
        wandb.finish()

    def save(self, path="fine_tuned_model"):
        self.model.save(path)

    def get_model(self):
        return self.model