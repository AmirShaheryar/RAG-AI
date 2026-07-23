import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


print("Loading PDF document... ")

loader=PyPDFLoader("AI_Healthcare_Guide.pdf")

document=loader.load()

print(document[0].page_content)


print("Splitting Document into chunks...")

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20,
    separators=["\n\n","\n"," ",""]
    )

chunks=text_splitter.split_documents(document)

print("len(chunks): ", len(chunks))
print("First chunk: ", chunks[0].page_content)