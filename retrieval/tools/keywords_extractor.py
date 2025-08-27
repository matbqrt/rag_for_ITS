# from langchain_ollama import OllamaLLM
from langchain_core.language_models.llms import LLM
import re, json

from rag_folder.templates.templates_loader import PromptBuilder


class KeywordsExtractor():
    def __init__(self, method:str, llm:LLM):
        self.llm = llm
        self.prompt_builder = PromptBuilder(prompt_dir="../templates/prompts")
        self.method = method

    def extract_keywords(self, question:str):
        prompt = self.prompt_builder.get_prompt_template(name=self.method)
        response = self.llm.invoke(prompt.format(question=question))
        return self.parse_response(response, self.method)

    def parse_response(self, response:str, method):
        if method == 'alignment': # parse a JSON list
            keywords = []
            pattern = r"(?:```json|'''json)\s*(\[.*?\])\s*(?:```|''')"
            match = re.search(pattern, response, re.DOTALL) if response else ""
            if match:
                keywords = match.group(1)
                try:
                    return json.loads(keywords)
                except json.JSONDecodeError:
                    return keywords 
            pattern_array = r'(\[.*?\])'
            matches = re.finditer(pattern_array, response, re.DOTALL)
            for match in matches:
                keywords = match.group(1)
                try:
                    return json.loads(keywords)
                except:
                    continue
            return keywords

        else: # Filter and adaptative methods, parse a JSON dict
            keywords = {}
            pattern = r"```json\s*|\s*```"
            raw_keywords = re.sub(pattern, "", response).strip() if response else ""
            match = re.findall(r'\{.*?\}', raw_keywords, re.DOTALL) if raw_keywords else ""
            keywords = re.sub(r'\bNone\b', 'null', match[0])
            try:
                return json.loads(keywords)
            except:
                return keywords
