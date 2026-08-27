from datasets import load_dataset


def load_nfcorpus():
    queries = load_dataset("mteb/nfcorpus", "queries")["queries"]
    corpus = load_dataset("mteb/nfcorpus", "corpus")["corpus"]
    qrels_ds = load_dataset("mteb/nfcorpus", "default")

    qrels_train = qrels_ds["train"]
    qrels_dev = qrels_ds["dev"]
    qrels_test = qrels_ds["test"]

    return queries, corpus, qrels_train, qrels_dev, qrels_test