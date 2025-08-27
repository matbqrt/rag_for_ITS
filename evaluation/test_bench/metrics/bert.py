from rag_folder.evaluation.test_bench.metrics.base_evaluator import BaseEvaluator 

from bert_score import BERTScorer

class BertScore(BaseEvaluator):
    def __init__(self, model_type='bert-base-uncased'):
        self.model_type = model_type
        self.scorer = BERTScorer(model_type=self.model_type, lang='en', rescale_with_baseline=True)

    def score(self, target_context:str, retrieved_context:str, question = None)->float:
        p, r, f1 = self.scorer.score([retrieved_context], [target_context])
        p, r, f1 = max(0, round(p.mean().item(),2)) , max(0, round(r.mean().item(),2)), max(0, round(f1.mean().item(),2))
        return(p, r, f1)
