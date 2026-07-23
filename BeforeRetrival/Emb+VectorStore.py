import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


print("Loading PDF document... ")

loader=PyPDFLoader("AI_Healthcare_Guide.pdf")

document=loader.load()

print(document[0].page_content)


