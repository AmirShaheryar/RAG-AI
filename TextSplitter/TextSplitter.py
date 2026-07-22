from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import CharacterTextSplitter

loader=PyPDFLoader("AI_Healthcare_Guide.pdf")

docs=loader.load()

print(docs[0].page_content)

text_split=CharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=0,
    separator="\n"
    )

docs_split=text_split.split_documents(docs)

print(docs_split[0].page_content)

print(len(docs_split))

print(docs_split)