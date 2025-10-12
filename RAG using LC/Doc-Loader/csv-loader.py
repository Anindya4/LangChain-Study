from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("Social_Network_Ads.csv")

docs = loader.load()

print(docs[15].page_content)