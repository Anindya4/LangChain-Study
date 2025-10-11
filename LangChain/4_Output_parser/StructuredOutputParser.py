from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser , ResponseSchema

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm1)



# Schema:
schema = [
    ResponseSchema(name="Fact-1", description="Fact 1 about the topic"),
    ResponseSchema(name="Fact-2", description="Fact 2 about the topic"),
    ResponseSchema(name="Fact-3", description="Fact 3 about the topic"),
    ResponseSchema(name="Fact-4", description="Fact 4 about the topic"),
    ResponseSchema(name="Fact-5", description="Fact 5 about the topic")
]

# Parser:
parser = StructuredOutputParser.from_response_schemas(schema)

# TEMPLATE :
temp = PromptTemplate(
    template= "Give me 5 facts about {topic} \n {format_instructions}",
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# prompt = temp.invoke({'topic':'black hole'})

# result = model.invoke(prompt)

# final_res = parser.parse(result.content)

# print(final_res)

# BY CHAINS:
chain = temp | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)