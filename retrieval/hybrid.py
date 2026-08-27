import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def rrf(rankings, k=60):
    scores = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


def _min_max_normalize(score_map):
    """
    Normalizes a dictionary of {document_id: score} to [0, 1].
    """
    if not score_map:
        return {}

    values = list(score_map.values())
    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return {
            doc_id: 1.0
            for doc_id in score_map
        }

    return {
        doc_id: (score - minimum) / (maximum - minimum)
        for doc_id, score in score_map.items()
    }


def _build_score_map(doc_ids, scores):
    """
    Converts separate ID and score lists into a score dictionary.
    """
    return {
        doc_id: float(score)
        for doc_id, score in zip(doc_ids, scores)
    }


def fixed_weighted_fusion(
    dense_ids,
    dense_scores,
    bm25_results,
    dense_weight=0.7,
    bm25_weight=0.3
):
    """
    Combines normalized dense and BM25 scores using fixed weights.
    """

    if not 0.0 <= dense_weight <= 1.0:
        raise ValueError("dense_weight must be between 0 and 1.")

    if not 0.0 <= bm25_weight <= 1.0:
        raise ValueError("bm25_weight must be between 0 and 1.")

    if abs((dense_weight + bm25_weight) - 1.0) > 1e-8:
        raise ValueError("dense_weight and bm25_weight must sum to 1.")

    dense_score_map = _build_score_map(
        dense_ids,
        dense_scores
    )

    bm25_score_map = {
        doc_id: float(score)
        for doc_id, score in bm25_results
    }

    dense_normalized = _min_max_normalize(dense_score_map)
    bm25_normalized = _min_max_normalize(bm25_score_map)

    candidate_ids = set(dense_normalized) | set(bm25_normalized)

    fused_scores = {}

    for doc_id in candidate_ids:
        dense_score = dense_normalized.get(doc_id, 0.0)
        bm25_score = bm25_normalized.get(doc_id, 0.0)

        fused_scores[doc_id] = (
            dense_weight * dense_score
            + bm25_weight * bm25_score
        )

    return sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


def max_score_fusion(
    dense_ids,
    dense_scores,
    bm25_results
):
    """
    Uses the maximum normalized retrieval score for each document.
    """

    dense_score_map = _build_score_map(
        dense_ids,
        dense_scores
    )

    bm25_score_map = {
        doc_id: float(score)
        for doc_id, score in bm25_results
    }

    dense_normalized = _min_max_normalize(dense_score_map)
    bm25_normalized = _min_max_normalize(bm25_score_map)

    candidate_ids = set(dense_normalized) | set(bm25_normalized)

    fused_scores = {}

    for doc_id in candidate_ids:
        fused_scores[doc_id] = max(
            dense_normalized.get(doc_id, 0.0),
            bm25_normalized.get(doc_id, 0.0)
        )

    return sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


def dynamic_weighted_fusion(
    dense_ids,
    dense_scores,
    bm25_results,
    dense_weight
):
    """
    Weighted fusion with a query-specific dense weight.

    dense_weight:
        1.0 -> only dense retrieval
        0.0 -> only BM25
    """

    dense_weight = max(0.0, min(1.0, float(dense_weight)))
    bm25_weight = 1.0 - dense_weight

    return fixed_weighted_fusion(
        dense_ids=dense_ids,
        dense_scores=dense_scores,
        bm25_results=bm25_results,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight
    )


class DynamicAlphaTuner:
    """
    Uses Qwen2.5-0.5B-Instruct to predict a query-specific dense weight.
    """

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        default_dense_weight=0.7
    ):
        self.default_dense_weight = default_dense_weight

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        self.model.eval()

    def predict_dense_weight(self, query):
        """
        Returns alpha in [0, 1].

        A higher alpha gives more weight to dense retrieval.
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "You determine the optimal hybrid retrieval weight "
                    "for a search query."
                )
            },
            {
                "role": "user",
                "content": (
                    "Choose the weight for dense semantic retrieval.\n"
                    "Return exactly one decimal number between 0 and 1 "
                    "and nothing else.\n\n"
                    "Interpretation:\n"
                    "0 means use only BM25 lexical retrieval.\n"
                    "1 means use only dense semantic retrieval.\n"
                    "Use a lower value for queries dominated by exact "
                    "technical terms, names, abbreviations, numbers, or "
                    "rare keywords.\n"
                    "Use a higher value for natural-language, conceptual, "
                    "or semantic queries.\n\n"
                    f"Query: {query}"
                )
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

        output_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        match = re.search(r"\d+(?:\.\d+)?", output_text)

        if match is None:
            print(
                "DAT warning: Could not parse alpha from "
                f"'{output_text}'. Using {self.default_dense_weight:.2f}."
            )
            return self.default_dense_weight

        alpha = float(match.group())

        # Allows an accidental percentage such as 70 instead of 0.7.
        if 1.0 < alpha <= 100.0:
            alpha = alpha / 100.0

        if not 0.0 <= alpha <= 1.0:
            print(
                f"DAT warning: Invalid alpha {alpha}. "
                f"Using {self.default_dense_weight:.2f}."
            )
            return self.default_dense_weight

        return alpha