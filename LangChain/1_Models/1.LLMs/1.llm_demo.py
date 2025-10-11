#to run this code we need OpenAI API/CLAUDE API keys which i don't have so the output will be an error.

from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct")
result = llm.invoke("Write me a Hello World program in Python")

print(result)


#CLAUDE CODE:

from langchain_anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

llm= Anthropic(model_name="claude-3-5-sonnet-latest")
result = llm.invoke("Write me a Hello World program in C++")

print(result)
