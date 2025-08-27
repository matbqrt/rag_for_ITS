from rag_folder.evaluation.test_bench.metrics.base_evaluator import BaseEvaluator 
from rag_folder.templates.templates_loader import PromptBuilder

import re, json
from langchain_core.language_models.llms import LLM

class JudgeScore(BaseEvaluator):
    def __init__(self, llm:LLM):
        self.llm = llm
        self.prompt_builder = PromptBuilder(prompt_dir="prompts")
        self.name = 'judge'

    def parse_response(self, response:str)->dict:
        match_with_null = {}
        try:
            dict_as_string = re.sub(r"```json\s*|\s*```", "", response).strip() if response else ""
            match = re.findall(r'\{.*?\}', dict_as_string, re.DOTALL) if dict_as_string else ""
            match_with_null = re.sub(r'\bNone\b', 'null', match[0]) # None value cannot be loaded (invalid JSON format)
            return json.loads(match_with_null)
        except:
            return(match_with_null)
        
    def score(self, target_context:str, retrieved_context:str, question = None)->dict:
        prompt = self.prompt_builder.get_prompt_template(self.name)
        response = self.llm.invoke(prompt.format(question = question, reference = target_context, prediction = retrieved_context))
        
        parsed_response = self.parse_response(response)
        parsed_response['grade'] = round(parsed_response['grade']/4,2) if 'grade' in parsed_response else 0
        return parsed_response