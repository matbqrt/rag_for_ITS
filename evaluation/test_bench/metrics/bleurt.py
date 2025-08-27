from rag_folder.evaluation.test_bench.metrics.base_evaluator import BaseEvaluator
from bleurt import score

class BleurtScore(BaseEvaluator):
    def __init__(self, checkpoint:str):
        self.checkpoint = checkpoint
        self.scorer = score.BleurtScorer(checkpoint=self.checkpoint)

    def score(self, target_context:str, retrieved_context:str, question = None)->float:
        if not retrieved_context:
            return(0)
        scores = self.scorer.score(references=[target_context], candidates=[retrieved_context])
        return round(scores[0],2)
        