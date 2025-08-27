# from langchain_ollama import OllamaLLM
from langchain_core.language_models.llms import LLM

from rag_folder.templates.templates_loader import PromptBuilder

class Summarization():
    def __init__(self, llm:LLM):
        self.name = 'summarization'
        self.llm = llm
        self.prompt_builder = PromptBuilder(prompt_dir="../templates/prompts")

    def summarize(self, question:str, context:str)->str:
        prompt = self.prompt_builder.get_prompt_template(self.name)
        return self.llm.invoke(prompt.format(question=question, docs=context))