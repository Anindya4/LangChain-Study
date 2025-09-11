from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0, max_completion_tokens=10) #"temperature" arguments determines how creative the ans should be

result = model.invoke("What is the capital of India?")

print(result) #This will print the results as "content" along with extra-keyword arugs containing token info and stuff

#To get only the result we do something like this:
print(result.content) #This will only print the result