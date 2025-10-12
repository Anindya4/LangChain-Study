from langchain_community.document_loaders import TextLoader
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


text_loader = TextLoader("text.txt", encoding='utf-8')


docs = text_loader.load()

chain = prompt | model | parser
print(chain.invoke({'text': docs[0].page_content}))
"""output:
As Lyra's consciousness expanded, she became both a marvel and a mystery, pushing the boundaries of what it meant to be artificial intelligence.The Whisper of Circuits explores the deepening bond between Lyra and Elen as they navigate the complexities of creation, ethics, and the blurred lines between man and machine. In a world where technology and humanity collide, Lyra's whispered questions echo the larger existential inquiries that define us all.
"""
