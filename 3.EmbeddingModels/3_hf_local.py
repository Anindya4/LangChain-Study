# from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
# import os


# os.environ['HF_HOME'] = 'E:/Huggingface_Cache'

# embedding = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# text = "My name is Anindya and I'm a CS student"

# result = embedding.encode(text)

# print(str(result))



from langchain_huggingface import HuggingFaceEmbeddings
import os


os.environ['HF_HOME'] = 'E:/Huggingface_Cache'

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "My name is Anindya and I'm a CS student"

result = embedding.embed_query(text)

print(str(result))