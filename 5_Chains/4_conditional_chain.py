from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableBranch
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

# model & parser:
model = ChatOpenAI(model="gpt-5-nano")
parser = StrOutputParser()

""" Creating a pydantic class for control the llms output """
class FeedBack(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentimental value of given feedback, either positive or negative")


parser2 = PydanticOutputParser(pydantic_object=FeedBack)


prompt1 = PromptTemplate(
    template="Given a text feedback, find the sentiment of it, wherther it's positive or negative \n {feedback} \n {format_instruction}" ,
    input_variables=["feedback"],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="Write an appropiate response in paragraph format to this positive feedback don't reply with a template of suggested feedback \n {feedback}",
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template="Write an appropiate response in paragraph format to this negative feedback don't reply with a template of suggested feedback \n {feedback}",
    input_variables=['feedback']
)


classifier_chain = prompt1 | model | parser2  #-> to classify the sentiment

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive", prompt2 | model | parser),
    (lambda x:x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not determine sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({"feedback": "This is a terrible watch"}))

chain.get_graph().print_ascii()
