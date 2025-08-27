from rag_folder.evaluation.test_bench.metrics.bert import BertScore
from rag_folder.evaluation.test_bench.metrics.bleurt import BleurtScore
from rag_folder.evaluation.test_bench.metrics.meteor import MeteorScore
from rag_folder.evaluation.test_bench.metrics.judge import JudgeScore

from langchain_core.language_models.llms import LLM
import json

class Scorer():
    def __init__(self, llm:LLM, bleurt_checkpoint=None, 
                 bert_model_type='bert-base-uncased'):
        self.bert_scorer = BertScore(bert_model_type)
        self.bleurt_scorer = BleurtScore(bleurt_checkpoint) if bleurt_checkpoint else None
        self.meteor_scorer = MeteorScore()
        self.judge_scorer = JudgeScore(llm)

    def load_questions(self, question_path:str, is_json=True)->list[str]:
        if not is_json:
            with open(question_path, "r", encoding="utf-8") as f:
                return(f.read().split('\n'))
        else:
            with open(question_path, "r", encoding="utf-8") as f:
                return(json.load(f))
        
    def load_references(self, reference_path:str, is_json=True)->list[str]:
        if not is_json:
            with open(reference_path, "r", encoding="utf-8") as f:
                return(f.read().split('\n-----\n'))
        else:
            with open(reference_path, "r", encoding="utf-8") as f:
                return(json.load(f))

    def evaluate(self, reference_context:str, retrieved_context:str, question:str):
        """ Semantic-based comparison (bleurt & bert) between target context 
        (most query-relevant documents) and retrieved context """
        p, r, f1 = self.bert_scorer.score(reference_context, retrieved_context)
        if self.bleurt_scorer:
            bleurt = self.bleurt_scorer.score(reference_context, retrieved_context)
        else:
            bleurt = "not initialize"
        meteor = self.meteor_scorer.score(reference_context, retrieved_context)
        judge = self.judge_scorer.score(reference_context, retrieved_context, question)

        return({'bleurt' : bleurt, 'bert' : [p, r, f1], 'meteor' : meteor, 'judge' : judge})

