from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

promt = PromptTemplate(
    template='Generate me 5 facts about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI(model="gpt-5-nano")

parser = StrOutputParser()

chain = promt | model | parser

result = chain.invoke({'topic': "GPUs"})
print(result)

#Visualize chains:
# chain.get_graph().print_ascii()