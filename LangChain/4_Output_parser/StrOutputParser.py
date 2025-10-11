from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

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


promt1 = temp1.invoke({"topic" : "Black Hole"})
result1 = model.invoke(promt1)

promt2 = temp2.invoke({"text": result1.content})
result2 = model.invoke(promt2)

print(result2.content)