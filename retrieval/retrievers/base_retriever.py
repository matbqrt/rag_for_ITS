from abc import ABC, abstractmethod

RETRIEVER_REGISTRY = {} # Store implemented retrieval methods

def register_retriever(name:str):
    def decorator(cls):
        RETRIEVER_REGISTRY[name] = cls
        return cls
    return decorator

class BaseRetriever(ABC):
    def __init__(self, index, top_k):
        self.index = index
        self.top_k = top_k

    @abstractmethod
    def retrieve(self, question:str)->str:
        """Return query-relevant docs as a context string"""
        pass