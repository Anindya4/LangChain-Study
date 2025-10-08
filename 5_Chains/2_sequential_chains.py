from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano")

promt1 = PromptTemplate(
    template="Give me a detailed report on {topic}",
    input_variables=['topic']
)

promt2 = PromptTemplate(
    template="Generate me 5 pointers summary from the given text. \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = promt1 | model | parser | promt2 | model | parser

result = chain.invoke({"topic": "LangChain"})
print(result)