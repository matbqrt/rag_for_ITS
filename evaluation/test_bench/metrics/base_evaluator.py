from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def score(self, target_context:str, retrieved_context:str, question = None)->float:
        pass