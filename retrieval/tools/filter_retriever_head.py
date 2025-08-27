from collections import defaultdict
from langchain.schema import Document
from io import BytesIO
import base64, re
from PIL import Image

class FilterRetrieverHead():
    def __init__(self, question:str, filter:dict, collection, chunked_db, top_k=5, logger=None):
        self.question = question
        self.filter = filter
        self.collection = collection
        self.chunked_db = chunked_db
        self.top_k = top_k
        self.log = logger or print
    
    def build_collection_filter(self)->dict:
        """ Adapt the extracted filter to the collection's terminlogy """
        filter = {}
        if not self.filter:
            return(filter)

        if 'name' in self.filter:
            filter['metadata.name'] = self.filter['name']
        elif 'community' in self.filter:
            filter = {'metadata.item type' : {"$regex": self.filter['item type'], "$options": "i"}, # Deal with maj
                    'metadata.community' : {"$regex": self.filter['community'], "$options": "i"}}
        elif 'class' in self.filter:
            filter['metadata.class'] = self.filter['class']
        elif 'item type' in self.filter:
            if 'physical' in self.filter:
                filter = {'metadata.item type' : self.filter['item type'], 'metadata.physical' : self.filter['physical']}
            else:
                filter = {'metadata.item type' : {"$regex": self.filter['item type'], "$options": "i"}}
        elif 'type' in self.filter:
            filter['metadata.type'] = self.filter['type'].lower()
        elif 'nature' in self.filter:
            filter['metadata.nature'] = self.filter['nature'].lower()
        elif 'domain name' in self.filter:
            filter['metadata.domain name'] = self.filter['domain name'].lower()
        elif 'service name' in self.filter:
            filter['metadata.service name'] = self.filter['service name'].lower()
        elif 'object name' in self.filter:
            filter['metadata.object name'] = self.filter['object name'].lower()

        return(filter)
    
    def get_ids(self, collection_filter:dict):
        """ Get the collection's retrieved documents ids to link collection and vector db.
            The latter is necessary to perform similarity search """
        ids = []
        if filter:
            docs = self.collection.find(collection_filter)
            if docs:
                for doc in docs:
                    if 'diagram' in doc['metadata']:
                        return(True, doc['metadata']['diagram'])
                    ids.append(doc['metadata']['id'])
                
        return(False, ids)
    
    def get_chunked_docs(self, ids:list)->list: # Use ids and similarity search to find most relevant documents
        """ Using the retrieved-documents' ids, perform similarity search (semantic comparison using cosine distance).
            This module allow the retriever to retrieve relevant documents among a short list of relevant docs """
        chunks, results = [], []
        if ids:
            chunks = self.chunked_db.similarity_search(query=self.question, k=self.top_k, filter={ 'id':{"$in": ids}})

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
    
    def decode_img(self, encoded_diagram):
        """ If a diagram is asked, decode its image (stored in its document's metadatas) """
        base64_img = encoded_diagram
        decoded_img = base64.b64decode(base64_img)
        img = Image.open(BytesIO(decoded_img))
        return(True, img, encoded_diagram)

    def retriever_head(self):
        """ Build filter, retrieve ids by filtering, chunks by similarity search and return 
            context as string """
        
        if not self.filter:
            self.log('Retrieving using similarity search.')
            chunks = self.chunked_db.similarity_search(query=self.question, k=5)
            ids = [doc.metadata['id'] for doc in chunks]
            is_diagram = False

        else:
            collection_filter = self.build_collection_filter()
            is_diagram, ids = self.get_ids(collection_filter)
            if is_diagram:
                return self.decode_img(ids)
            
            chunks = self.get_chunked_docs(ids)
            if not chunks: # In case filter is not valid, perform retrieval using similarity search (based on semantic equivalence)
                self.log('Retrieving using similarity search.')
                chunks = self.chunked_db.similarity_search(query=self.question, k=5)
                ids = [doc.metadata['id'] for doc in chunks]
                is_diagram = False
        
        content = [doc.page_content for doc in chunks]
        context = re.sub(r'^\s*---\s*$\n?', '', '\n-----\n'.join(content), flags=re.MULTILINE)
        return(is_diagram, context, ids)
