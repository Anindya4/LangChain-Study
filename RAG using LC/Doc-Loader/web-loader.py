from langchain_community.document_loaders import WebBaseLoader

url = "https://www.amazon.com"


loader = WebBaseLoader(web_path=url)

docs = loader.load()

print(docs[0].page_content)