from langchain_community.document_loaders import TextLoader, PyPDFLoader


text_loader = TextLoader("text.txt", encoding='utf-8')
# pdf_loader = PyPDFLoader("book.pdf")



docs = text_loader.load()
# print(list(pdf_loader.load()))

print(type(docs))

# print(docs[0])
print(docs[0].metadata)