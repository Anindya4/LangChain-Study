from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

loader = DirectoryLoader(
    path= "folder",
    glob="*.pdf",
    loader_cls= PyPDFLoader,
    show_progress=True
    
)

docs = loader.lazy_load()
# print(len(docs))

for d in docs:
    print(d.metadata)