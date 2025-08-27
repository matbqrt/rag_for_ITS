""" The goal here is to improve the similarity scores after alignment (exclusively for the alignement method) """
from itertools import combinations
import os

class KeywordPostprocessing():
    def __init__(self, cleaned_keywords: list[str], aligned_keywords:list[tuple], items_db):
        
        self.cleaned_keywords = cleaned_keywords
        self.aligned_keywords = aligned_keywords
        self.item_names = self.load_item_names()
        self.items_db = items_db

    def load_item_names(self, file_name = "item_names.txt"):
        base_dir = os.path.dirname(os.path.abspath(__file__)) # Absolute path
        txt_path = os.path.join(base_dir, "../data", file_name)

        with open(txt_path, "r", encoding='utf-8') as f:
            return [line.strip() for line in f]

    def is_perfect_match(self, keyword_score, similarity_threshold=0.1)->bool: 
        return(keyword_score < similarity_threshold)
    
    def sorted_filters(self): # Separate perfect match and uncertain match into two list
        top_filters, unmatched_keywords = {}, []

        for i, match in enumerate(self.aligned_keywords):
            best_match = match[0] # The doc with highest score
            if self.is_perfect_match(best_match[1]): # Check if the best match is also a perfect match
                top_filters.update(best_match[0].metadata)
            else:
                unmatched_keywords.append((self.cleaned_keywords[i], best_match[1])) # Add keyword and its score to the keywords-to-match list
        return(top_filters, unmatched_keywords)
        
    def get_matches(self, keywords: list[str]) -> dict[tuple, list]: # For each pair of keywords, get the list of matches
        best_match = {}
        
        def item_matching(keyword:str, item_names:list[str]): # List all the items similar to the extracted keywords
            return [item_name for item_name in item_names if keyword.lower() in item_name.lower()]
    
        # Generate all unique pairs of keywords
        for kw1, kw2 in combinations(keywords, 2):
            matches_kw1 = set(item_matching(kw1), self.item_names)
            matches_kw2 = set(item_matching(kw2), self.item_names)
            
            common_matches = list(matches_kw1 & matches_kw2)  # Intersection
            if common_matches:
                best_match[(kw1, kw2)] = common_matches
        
        return best_match
    
    def post_processing(self)->dict:
        perfect_matches, partial_matches = self.sorted_filters() # Separate perfect and partial matches

        def try_match(best_match, items_db = self.items_db): # Check if the new match improve the similarity score
            if not best_match:
                return None
            
            first_pair = next(iter(best_match.items()))
            _, common_service_name_list = first_pair
            if common_service_name_list:
                res = items_db.similarity_search_with_score(common_service_name_list[0], k=1) # res is a list : for each keyword the top k nearest neighbours
                new_match = res[0] # best match is a tuple (doc, score)
                if self.is_perfect_match(new_match[1]):
                    return(new_match)
                
        def keep_new_match(new_match, partial_match_list:list): # Keep the new combination if the similarity score is improved
            partial_scores = [score for (_,score) in partial_match_list]
            return(new_match[1] < min(partial_scores) if partial_scores else True)

        def set_final_type_value(filters:dict)->dict:
            filters_copy = filters.copy()
            if 'nature' in filters_copy:
                filters_copy['type'] = 'object'
            return(filters_copy)
    
        if partial_matches:
            keywords_to_match = []
            for tuple in partial_matches:
                keywords_to_match.append(tuple[0])
            
            if len(keywords_to_match) > 1:
                new_matches = self.get_matches(keywords_to_match)
                if new_matches:
                    new_filter = try_match(new_matches)
                    if keep_new_match(new_filter, partial_matches):
                        perfect_matches.update(new_filter[0].metadata)
            else: 
                partial_res = self.items_db.similarity_search(keywords_to_match[0], k=1)
                perfect_matches.update(partial_res[0].metadata)

        final_filter = set_final_type_value(perfect_matches)

        return(final_filter) 