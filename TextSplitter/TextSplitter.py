from langchain_community.document_loaders import PyPDFLoader



loader=PyPDFLoader("AI_Healthcare_Guide.pdf")

docs=loader.load()

print(docs[0].page_content)

