import re

class KeywordPreprocessing():
    def __init__(self, keyword:str, question:str, 
                 context_words = ["domain", "area", "service", "object", "item",  "physical", "functional", "information flow", "diagram", "process", "field"], 
                 noisy_words = ["definition", "description", "its", "reference architecture", "elements", "package"], 
                 domains=["commercial vehicle operations", "data management", "maintenance and construction", "parking management", "public safety", "public transportation", "sustainable travel", "support", "traveler information and personal mobility", "traffic management", "vehicle safety", "weather"]
    ):
        self.keyword = keyword
        self.question = question
        self.context_words = context_words
        self.noisy_words = noisy_words
        self.domains = domains
    
    def keyword_in_query(self):
        return(self.keyword.lower() in self.question.lower())
    
    def word_to_singular(self, word:str, context_words:list[str])->str:
        if not word:
            return word
        
        keyword_lower = word.lower()
        if keyword_lower.endswith('s') and keyword_lower[:-1] in context_words: # Field word in plural form, convert to singular form to ensure perfect match
            return(keyword_lower[:-1])
        return(self.keyword)
    
    def context_word_removing(self, context_words:list[str])->str: # Get rid of the context words (like service in the query : what is the definition of the X service), it will improve the alignement step
        split_keyword = self.keyword.split()
        nb_words = len(split_keyword)
        new_keyword = self.keyword
        
        if self.keyword == 'physical diagram':
            return 'diagram'
        
        if nb_words > 1: # Remove context words only if they are not the main subject of the keyword
            last_word = split_keyword[-1]
            last_word_sing = self.word_to_singular(last_word, context_words)
            if last_word in context_words or last_word_sing in context_words:
                if split_keyword[0].lower() == "its":
                    new_keyword = ' '.join(split_keyword[1:]) # deal with the case 'ITS domains', etc
                else:  
                    new_keyword = ' '.join(split_keyword[:-1])
        
        return new_keyword

    def is_service(self)->bool:
        service_pattern = r'^([a-zA-Z]{2,3})(0[1-9]|[12][0-9]|50):'
        return bool(re.match(service_pattern, self.keyword))

    def is_domain(self)->bool:
        return(self.keyword in self.domains)
    
    def preprocess_keyword(self)->str:
        clean_keyword = ''
        context_words = self.context_words + self.noisy_words
        if self.keyword:
            if self.keyword_in_query() and self.keyword.lower() not in self.noisy_words: # ignore keywords that are not relevant for the filtering
                clean_keyword = self.context_word_removing(context_words)
                clean_keyword = self.word_to_singular(clean_keyword, context_words) # convert to singular to have a cleaned filter
                if clean_keyword == 'information flow':
                    clean_keyword = 'flow' # Adapt to the lattest db terminology (flow instead of information flow)  
        return(clean_keyword)