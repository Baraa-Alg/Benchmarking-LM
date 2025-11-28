from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
from rouge import Rouge

class BLEUMetric:
    """Compute BLEU score between reference and generated text."""

    def __init__(self):
        self.smooth_fn = SmoothingFunction().method1
        self.name = "BLEU"

    def compute(self, reference: str, candidate: str) -> float:
        if not reference or not candidate:
            return 0.0
        score = sentence_bleu(
            [reference.split()],
            candidate.split(),
            smoothing_function=self.smooth_fn
        )
        return round(score, 4)

class RougeMetric:
    name = "ROUGE_L"

    def compute(self, ref, hyp):
        if not ref or not hyp:
            return 0.0
        rouge = Rouge()
        try:
            scores = rouge.get_scores(hyp, ref)
            return scores[0]["rouge-l"]["f"]
        except Exception:
            return 0.0


class BertScoreMetric:
    name = "BERTScore"

    def compute(self, ref, hyp):
        if not ref or not hyp:
            return 0.0
        try:
            P, R, F1 = score([hyp], [ref], lang="en", verbose=False)
            return F1.mean().item()
        except Exception:
            return 0.0
