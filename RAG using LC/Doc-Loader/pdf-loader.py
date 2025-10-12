from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-5-nano")
parser = StrOutputParser()
prompt = PromptTemplate(
    template="write a summary on this {text}",
    input_variables=['text']
)


pdf_loader = PyPDFLoader("book.pdf")

doc = pdf_loader.load()

# print(doc)
print(doc[50].page_content)
print(doc[50].metadata)