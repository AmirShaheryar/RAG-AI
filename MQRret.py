from langchain_community.document_loaders import TextLoader, PyPDFLoader

docs=PyPDFLoader("RAG_MQR_Contextual_Retrieval_Test_Document.pdf").load()

print(docs[0].page_content)


