from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define the model:
llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm1)


class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="Name of the city the person resides")
    
parser = PydanticOutputParser(pydantic_object=Person)

temp = PromptTemplate(
    template= "Write me the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables= ["place"],
    partial_variables= {"format_instruction": parser.get_format_instructions()}
)

# prompt = temp.invoke({'place': 'Indian'})
# print(prompt)

# result = model.invoke(prompt)

# final_res = parser.parse(result.content)

# print(final_res)


# VIA CHAINS
chain = temp | model | parser

result = chain.invoke({'place': 'Indian'})
print(result)