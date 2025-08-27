from generation.summarization import Summarization
from generation.generator import Generation

from langchain_core.language_models.llms import LLM

class RAG():
    def __init__(self, retriever:str, llm:LLM):
        self.llm = llm
        self.retriever = retriever
        self.generator = Generation(llm)
        self.summarizer = Summarization(llm)

    def invoke(self, question:str, summarize = True):
        is_diagram, context, ids = self.retriever.retrieve(question)
        if is_diagram:
            context.show()
        else:
            if summarize:
                summary = self.summarizer.summarize(question, context)
            else:
                summary = context
            return self.generator.generate(question, summary)

