from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser

from rag_folder.templates.schemas.output_schemas import schemas

class PromptBuilder:
    def __init__(self, prompt_dir="prompts"):
        base_path = Path(__file__).parent.resolve() # Use absolute path from current file
        self.prompt_dir = base_path / prompt_dir
        self.env = Environment(loader=FileSystemLoader(prompt_dir))

    def load_instruction_format(self, name):
        if name in schemas:
            response_schemas = schemas[name]
            parser = StructuredOutputParser.from_response_schemas(response_schemas)
            return parser.get_format_instructions()
        return None

    def load_prompt(self, name):
        prompt_path = self.prompt_dir / f"{name}.txt"
        with open(prompt_path, "r", encoding='utf-8') as f:
            return(f.read())
    
    def load_input_variables(self, name):
        if name in ['summarization', 'generation']:
            return ['question', 'docs']
        elif name == 'judge':
            return ['question', 'reference', 'prediction']
        elif name == 'history':
            return ['question', 'sources']
        return ['question']

    def get_prompt_template(self, name):
        prompt = self.load_prompt(name)
        instruction_format = self.load_instruction_format(name)
        input_variables = self.load_input_variables(name)

        partial_var = {}
        if instruction_format:
            partial_var['instruction_format'] = instruction_format

        return PromptTemplate(
            template=prompt,
            input_variables=input_variables,
            partial_variables=partial_var,
        )
