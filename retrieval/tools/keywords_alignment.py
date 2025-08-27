from rag_folder.retrieval.tools.keywords_preprocessing import KeywordPreprocessing
from rag_folder.retrieval.tools.keywords_postprocessing import KeywordPostprocessing

class KeywordsAlignement():
    def __init__(self, question:str, keywords, method:str, 
                 items_db, community_db=None, similarity_threshold = 0.4, 
                 classes = ['field', 'vehicle', 'personal', 'center', 'support']):
        
        self.question = question
        self.keywords = keywords
        self.method = method
        self.items = items_db
        self.communities = community_db
        self.similarity_threshold = similarity_threshold
        self.classes = classes # To clean final filter

    def align_keyword(self, keyword:str, field = None):
        if self.method == 'adaptative':
            if field == 'item':
                items_with_score = self.items.similarity_search_with_score(keyword, k=1) # List of tuples (Doc, Score)
                if items_with_score[0][1] < self.similarity_threshold:
                    return(items_with_score[0][0].metadata) # return {'item type' : , 'name' : }
            if field == 'community':
                community_with_score = self.communities.similarity_search_with_score(keyword, k=1)
                if community_with_score[0][1] < self.similarity_threshold:
                    return(community_with_score[0][0].metadata) # return {'name' : }
            return({})
        else: # alignment method
            return self.items.similarity_search_with_score(keyword, k=1) # return [(Document, score)]
        
    def build_filter(self):
        """ From extracted keywords (either a dict or a list) return a cleaned (and aligned) filter """

        if self.method == 'adaptative':
            item, community = self.keywords['item'], self.keywords['community'] if self.keywords['community'] is not None else ''
            if community in item:
                community = ''
                
            aligned_keywords = {}

            # Align item
            cleaned_item = KeywordPreprocessing(item, self.question).preprocess_keyword() if item else '' # Get rid of context words, noisy words, goal is to improve the similarity score
            aligned_item = self.align_keyword(cleaned_item, 'item') if cleaned_item else '' # Get best match from items db
            if aligned_item: # Make sure that the cleaned item wasn't a 'context word'
                if 'name' in aligned_item:
                    aligned_keywords['name'] = aligned_item['name']
                else:
                    aligned_keywords['item type'] = aligned_item['item type']

            # Align community
            cleaned_community = KeywordPreprocessing(community, self.question).preprocess_keyword() if community else ''
            aligned_community = self.align_keyword(cleaned_community, 'community') if cleaned_community else ''
            if aligned_community:
                if 'name' in aligned_community and 'item type' in aligned_keywords:
                    aligned_keywords['community'] = aligned_community['name']
                if not aligned_item:
                    if 'name' in aligned_community:
                        aligned_keywords['name'] = aligned_community['name']
                    else:
                        aligned_keywords['item type'] = aligned_community['item type']

            # Process aligned keywords
            if 'item type' in aligned_keywords:
                if aligned_keywords['item type'] == 'process':
                    if 'community' in aligned_keywords:
                        aligned_keywords['physical'] = aligned_keywords.pop('community') # A process community is the physical object linked
                elif aligned_keywords['item type'] == 'service':
                    if 'community' in aligned_keywords:
                        if KeywordPreprocessing(aligned_keywords['community'], self.question).is_service(): # If service type is service and community is a service name then community field becomes name field
                            aligned_keywords['name'] = aligned_keywords.pop('community')
                            aligned_keywords.pop('item type')
                elif aligned_keywords['item type'] == 'domain':
                    if 'community' in aligned_keywords:
                        if KeywordPreprocessing(aligned_keywords['community'], self.question).is_domain(): # If service type is service and community is a service name then community field becomes name field
                            aligned_keywords['name'] = aligned_keywords.pop('community')
                            aligned_keywords.pop('item type')
                            
            if 'community' in aligned_keywords and aligned_keywords['community'] in self.classes:
                aligned_keywords['class'] = aligned_keywords.pop('community')
                
            return(aligned_keywords)

        if self.method == 'alignment':

            cleaned_keywords, aligned_keywords = [], []
            for keyword in self.keywords:
                cleaned_keyword = KeywordPreprocessing(keyword, self.question).preprocess_keyword()
                cleaned_keywords.append(cleaned_keyword), aligned_keywords.append(self.align_keyword(cleaned_keyword))

            filter = KeywordPostprocessing(cleaned_keywords, aligned_keywords, self.items).post_processing()
            return(filter)

