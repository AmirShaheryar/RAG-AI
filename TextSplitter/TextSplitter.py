from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import CharacterTextSplitter

from langchain_text_splitters import RecursiveCharacterTextSplitter

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


textSplit=RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=0,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
    )

docs_split=textSplit.split_documents(docs)
print("Split Text\n")
print(docs_split[0].page_content)