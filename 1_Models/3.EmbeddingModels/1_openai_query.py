#AGAIN WE NEED API KEY WHICH I DON'T HAVE😭😭

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

result = embedding.aembed_query("Delhi is the capital of India")
 
print(str(result))

