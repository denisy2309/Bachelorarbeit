from datasets import load_dataset

def load_trec_covid():
    queries = load_dataset("mteb/trec-covid", "queries")["queries"]
    corpus = load_dataset("mteb/trec-covid", "corpus")["corpus"]
    qrels = load_dataset("mteb/trec-covid", "default")["test"]

    return queries, corpus, qrels