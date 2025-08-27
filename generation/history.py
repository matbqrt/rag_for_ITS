from langchain_core.language_models.llms import LLM
import re, json
from collections import defaultdict
from langchain.schema import Document

from rag_folder.templates.templates_loader import PromptBuilder

class History():
    def __init__(self, llm:LLM, collection, logger=None):
        self.llm = llm
        self.collection = collection
        self.name = 'history'
        self.prompt_builder = PromptBuilder(prompt_dir="../templates/prompts")
        self.log = logger or print

    def load_sources(self, ids:list)->list[dict]:
        """ Get retrieved documents using their ids """
        sources = []
        if ids:
            docs = self.collection.find({'metadata.id': {"$in":ids}})
            for doc in docs:
                sources.append({doc['metadata']['name'] : doc['metadata']['id']})
        return sources

    def retrieve_ids(self, question:str, history_ids:list)->list:
        """ Get useful documents from history """
        ids, names_short_list = [], []
        prompt = self.prompt_builder.get_prompt_template(self.name)

        def get_names(sources:list[dict]):
            return [next(iter(d)) for d in sources]
        
        sources = self.load_sources(history_ids)

        if sources:
            sources_as_dict = {k: v for d in sources for k, v in d.items()}
            names = get_names(sources)
            names_short_list = self.parse_response(self.llm.invoke(prompt.format(question=question, sources=names)))

        if names_short_list:
            self.log(f"The following documents are useful to answer the question : {names_short_list}")
            ids = [sources_as_dict[name] for name in names if name in names_short_list]

        return ids
    
    def get_chunked_docs(self, ids:list, question:str, chunked_db, top_k=5)->list: # Use ids and similarity search to find most relevant documents
        """ Using the retrieved-documents' ids, perform similarity search (semantic comparison using cosine distance).
            This module allow the retriever to retrieve relevant documents among a short list of relevant docs """
        chunks, results = [], []
        if ids:
            chunks = chunked_db.similarity_search(query=question, k=top_k, filter={ 'id':{"$in": ids}})

            grouped_docs = defaultdict(list)    
            if chunks:
                for doc in chunks:
                    id = doc.metadata.get('id')
                    grouped_docs[id].append(doc)

                for id, docs in grouped_docs.items():
                    new_content = "\n".join(doc.page_content for doc in docs)
                    new_metadata = docs[0].metadata
                    results.append(Document(page_content=new_content, metadata=new_metadata))

        return(results)
    
    def get_context(self, ids:list, question:str, chunked_db)->str:
        chunks = self.get_chunked_docs(ids, question, chunked_db)
        if not chunks:
            chunks = chunked_db.similarity_search(query=question, k=5)

        content = [doc.page_content for doc in chunks]
        context = re.sub(r'^\s*---\s*$\n?', '', '\n-----\n'.join(content), flags=re.MULTILINE)

        return(context, ids)

    def parse_response(self, response:str):
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
            if not match:
                return []   
            json_block = match.group(1)
            json_block = json_block.replace("'", '"')
            data = json.loads(json_block)
            return data.get("relevant docs", [])
        except (json.JSONDecodeError, TypeError):
            return []
        
    def merge_ids(self, past_ids:list, new_ids:list)->str:
        """ Merge past and new ids and get context from unique documents """
        seen = set()
        merged = []

        if not past_ids and not new_ids:
            return merged
        
        past_ids = past_ids or []
        new_ids = new_ids or []
        ids = past_ids + new_ids

        for id in ids:
            if id not in seen:
                seen.add(id)
                merged.append(id)

        return merged