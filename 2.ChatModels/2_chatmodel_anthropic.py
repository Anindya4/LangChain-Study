from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-opus-4-1-20250805", temperature=0) #"temperature" arguments determines how creative the ans should be

result = model.invoke("What is the capital of India?")

print(result) #This will print the results as "content" along with extra-keyword arugs containing token info and stuff.1

#To get only the result we do something like this:
print(result.content) #This will only print the result