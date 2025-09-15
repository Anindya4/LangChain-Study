#AGAIN WE NEED API KEY WHICH I DON'T HAVE😭😭

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

document = [
    "Delhi is the capital of India",
    "Tokyo is the capital of Japan",
    "Today is Wednesday",
    "The sky is blue"
]

result = embedding.aembed_documents(document)
 
print(str(result))

