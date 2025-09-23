# Creating dynamic promts for chatbots:
from langchain_core.prompts import ChatPromptTemplate

chat_temp = ChatPromptTemplate([
    ('system', 'You are an {domain} expert'),
    ('human', 'Explain what is {topic} ')
])

promt = chat_temp.invoke({'domain': 'computer science', 'topic':'programming'})

print(promt)