from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm1)

# 1 promt -> report generation
temp1 = PromptTemplate(
    template= "Generate a detailed report of the following topic. Topic: {topic}",
    input_variables=['topic']
)


# 2 promt -> summary generate
temp2 = PromptTemplate(
    template= "Generate a 5 line summary of the following text. \n Text: {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = temp1 | model | parser | temp2 | model | parser

result = chain.invoke({"topic": "Black Hole"})
print(result)
