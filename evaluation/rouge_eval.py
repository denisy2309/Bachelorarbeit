from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def compute_rouge(pred, gt):
    scores = scorer.score(gt, pred)
    return scores["rouge1"].fmeasure, scores["rougeL"].fmeasure