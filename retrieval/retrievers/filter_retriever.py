from rag_folder.retrieval.retrievers.base_retriever import BaseRetriever, register_retriever
from rag_folder.retrieval.tools.keywords_extractor import KeywordsExtractor
from rag_folder.retrieval.tools.filter_retriever_head import FilterRetrieverHead

from langchain_core.language_models.llms import LLM

@register_retriever("filter")
class FilterRetriever(BaseRetriever):
    def __init__(self, llm:LLM, chunked_db, mongo_collection, top_k=5):
        super().__init__(chunked_db, top_k)
        self.method = 'filter'
        self.llm = llm
        self.collection = mongo_collection

    def retrieve(self, question:str)->str:
        """Identify and extract item and community from the query, align them, filter"""
        keywords = KeywordsExtractor(self.method, self.llm).extract_keywords(question)
        print(f'Extracted filter is : {keywords}')
        return FilterRetrieverHead(question, filter, self.collection, self.index, self.top_k).retriever_head()