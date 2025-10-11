# A simple question answer bot runs on terminal via openai api key
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano")

# Adding memory:
chat_history = [
    SystemMessage(content='You are a helpful AI assistant')  # setting a system message
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input)) # adding to memory and converting to human massages
    if user_input == "exit":
        break

    result = model.invoke(chat_history) # sending the entire chat history to the LLM for context
    chat_history.append(AIMessage(content=result.content))  # adding the results in memory and converting it to ai messages
    print("AI: ", result.content)
    
print(chat_history)

