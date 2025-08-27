from rag_folder.retrieval.retrievers.base_retriever import BaseRetriever, register_retriever
from rag_folder.retrieval.tools.keywords_extractor import KeywordsExtractor
from rag_folder.retrieval.tools.keywords_alignment import KeywordsAlignement
from rag_folder.retrieval.tools.filter_retriever_head import FilterRetrieverHead

from langchain_core.language_models.llms import LLM

@register_retriever("alignment")
class AlignmentRetriever(BaseRetriever):
    def __init__(self, llm:LLM, items_db, chunked_db, mongo_collection, top_k=5):
        super().__init__(chunked_db, top_k)
        self.method = 'alignment'
        self.llm = llm
        self.alignment_db = items_db
        self.collection = mongo_collection
    
    def retrieve(self, question:str)->str:
        """Identify and extract query's keywords, align them, filter"""
        keywords = KeywordsExtractor(self.method, self.llm).extract_keywords(question)
        filter = KeywordsAlignement(question, keywords, self.method, self.alignment_db).build_filter()
        return FilterRetrieverHead(question, filter, self.collection, self.index, self.top_k).retriever_head()