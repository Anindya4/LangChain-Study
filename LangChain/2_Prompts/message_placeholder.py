from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template:
chat_temp = ChatPromptTemplate([
    ('system','You are a customer service agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# load chat history:
chat_history =[]

with open("2_Prompts/chat_his.txt") as f:
    chat_history.extend(f.readlines())

print(chat_history)

# load promt
promt = chat_temp.invoke({'chat_history':chat_history, 'query':'What is the status of my refund?'})
print(promt)