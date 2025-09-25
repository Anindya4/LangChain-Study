from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm1)

parser = JsonOutputParser()

temp = PromptTemplate(
    template='Give me the name, city and agr of this fictional character. \n {formate_instruction}',
    input_variables=[],
    partial_variables= {'formate_instruction': parser.get_format_instructions()}
)

