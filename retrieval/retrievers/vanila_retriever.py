from rag_folder.retrieval.retrievers.base_retriever import BaseRetriever, register_retriever

import re

@register_retriever("vanila")
class VanilaRetriever(BaseRetriever):
    def __init__(self, chunked_db, top_k):
        super().__init__(chunked_db, top_k)

    def retrieve(self, question:str)->str:
        """Embed question and retrieve"""
        retrieved_docs = self.index.similarity_search(question, self.top_k)

        content = [doc.page_content for doc in retrieved_docs]
        context = re.sub(r'^\s*---\s*$\n?', '', '\n-----\n'.join(content), flags=re.MULTILINE)

        ids = [doc.metadata['id'] for doc in retrieved_docs]
        is_diagram = False

        return(is_diagram, context, ids)
        