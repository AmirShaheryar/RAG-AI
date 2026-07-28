from langchain_community.document_loaders import TextLoader, PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

docs=PyPDFLoader("RAG_MQR_Contextual_Retrieval_Test_Document.pdf").load()

print(docs[0].page_content)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20,
    separators=["\n\n","\n"," ",""]
    )

split_Doc=splitter.split_documents(docs)


print(split_Doc[0].page_content)