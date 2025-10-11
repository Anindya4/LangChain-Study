from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

document = [
    "Albert Einstein developed the theory of relativity, revolutionizing physics, and won the Nobel Prize for the photoelectric effect in 1921.",
    "Marie Curie was the first woman to win Nobel Prizes in Physics and Chemistry, pioneering research on radioactivity and radium.",
    "Martin Luther King Jr. led nonviolent protests for civil rights in America, inspiring the famous 'I Have a Dream' speech.",
    "Leonardo da Vinci was a Renaissance genius, famous for the Mona Lisa and innovative studies in anatomy and flight.",
    "Malala Yousafzai fights for girls' education worldwide, surviving a Taliban attack and becoming the youngest Nobel Peace Prize laureate."
]

user_query1 = "Who was the first woman to win Nobel Prize in physics?"
user_query2 = "Who is famous for Mona Lisa?"

doc_embedding = embedding.encode(document)
query_embedding = embedding.encode(user_query2)

scores = cosine_similarity([query_embedding],doc_embedding)[0]  #The output needs to a 1D list that is why we are doing this

"""
Now we need the cosine similarity with the largest value ie the value which is biggest is out ans.
so we will first enumerate that list and then sort them on the basis of highest similarity value.
"""

index, score = sorted(list(enumerate(scores)),key= lambda x: x[1])[-1]

print(user_query2)
print(document[index])
print(f"The similarity score is: {score}")

