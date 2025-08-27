from rag_folder.retrieval.retrievers.base_retriever import BaseRetriever, register_retriever
from rag_folder.retrieval.tools.keywords_extractor import KeywordsExtractor
from rag_folder.retrieval.tools.keywords_alignment import KeywordsAlignement
from rag_folder.retrieval.tools.filter_retriever_head import FilterRetrieverHead

from langchain_core.language_models.llms import LLM

@register_retriever("adaptative")
class AdaptativeRetriever(BaseRetriever):
    def __init__(self, llm:LLM, items_db, community_db, chunked_db, mongo_collection, top_k=5, logger = None):
        super().__init__(chunked_db, top_k)
        self.method = 'adaptative'
        self.llm = llm
        self.items_db = items_db
        self.community_db = community_db
        self.collection = mongo_collection
        self.log = logger or print # for interface purposes

    def retrieve(self, question:str)->str:
        """Identify and extract item and community from the query, align them, filter"""
        keywords = KeywordsExtractor(self.method, self.llm).extract_keywords(question)
        filter = KeywordsAlignement(question, keywords, self.method, self.items_db, self.community_db).build_filter()
        if filter:
            self.log(f'Extracted filter is : {filter}')
        else:
            self.log(f'No filter extracted from the query.')
        return FilterRetrieverHead(question, filter, self.collection, self.index, self.top_k, self.log).retriever_head()
        

            