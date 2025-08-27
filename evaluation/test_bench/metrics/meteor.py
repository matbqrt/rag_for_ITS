from rag_folder.evaluation.test_bench.metrics.base_evaluator import BaseEvaluator 

from nltk.translate import meteor
from nltk import word_tokenize

class MeteorScore(BaseEvaluator):
    def __init__(self):
        pass

    def score(self, target_context:str, retrieved_context:str, question = None)->float:
        meteor_score = meteor([word_tokenize(target_context)], word_tokenize(retrieved_context)) 
        return round(meteor_score, 2)