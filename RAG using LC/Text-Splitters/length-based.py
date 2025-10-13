from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('book.pdf')
docs = loader.load()
splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10,
    separator=''
)

# text = """
# The cyber landscape has changed dramatically with the rapid adoption of artificial intelligence. In the frenzied race to harness the potential of AI, organizations often find themselves up against the clock, eager to deploy AI without first assessing their foundational cybersecurity measures. This creates a dangerous parallel: while businesses scramble to adopt AI for competitive advantage, cybercriminals are just as rapidly incorporating these technologies into their attack arsenals.
# It's not all bad news. For the first time in five years, global data breach costs have declined. IBM's newly released 2025 Cost of a Data Breach Report found that average global costs dropped to USD 4.44 million—down from USD 4.88 million, or 9%, in the year prior. The catalyst? Faster breach containment driven by AI-powered defenses. According to the report, organizations were able to identify and contain a breach within a mean time of 241 days, the lowest it's been in nine years.
# """

result = splitter.split_documents(docs)

print(result[:3])
# print(type(result[250]))