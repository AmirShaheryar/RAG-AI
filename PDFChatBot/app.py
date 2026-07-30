from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

docs=PyPDFLoader("Health_RAG_Test_Document.pdf").load()

#print(docs[0].page_content)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20
    ) 

splitted_text=text_splitter.split_documents(docs)

print(splitted_text[0].page_content)