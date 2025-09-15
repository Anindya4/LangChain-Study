from langchain_huggingface import HuggingFaceEmbeddings
import os


os.environ['HF_HOME'] = 'E:/Huggingface_Cache'

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "My name is Anindya and I'm a CS student"

document = [
    "Delhi is the capital of India",
    "Tokyo is the capital of Japan",
    "The sky is blue"
]

result = embedding.embed_documents(document)

print(str(result))



# from sentence_transformers import SentenceTransformer
# import os


# os.environ['HF_HOME'] = 'E:/Huggingface_Cache'

# embedding = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# text = "My name is Anindya and I'm a CS student"

# document = [
#     "Delhi is the capital of India",
#     "Tokyo is the capital of Japan",
#     "The sky is blue"
# ]

# result = embedding.encode(document)

# print(str(result))
