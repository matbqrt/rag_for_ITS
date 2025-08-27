from rag_folder.retrieval.retrievers.base_retriever import RETRIEVER_REGISTRY

class Retriever:
    def __init__(self, method:str, **kwargs):
        if method not in RETRIEVER_REGISTRY:
            raise ValueError(f'Unknown method : {method}')
        RetrieverClass = RETRIEVER_REGISTRY[method]
        self.retriever = RetrieverClass(**kwargs)

    def retrieve(self, question:str):
        return self.retriever.retrieve(question)