from langchain_community.document_loaders import TextLoader, PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings

from langchain_chroma import Chroma


docs=PyPDFLoader("RAG_MQR_Contextual_Retrieval_Test_Document.pdf").load()

print(docs[0].page_content)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=100,
    separators=["\n\n","\n",". "]
    )

split_Doc=splitter.split_documents(docs)


print(split_Doc[0].page_content)

embedings = OllamaEmbeddings(
    model="nomic-embed-text" 
    )

vector_store=Chroma.from_documents(
    documents=split_Doc,
    embedding=embedings
)

base_retriver=vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
    )

query = "What is Core Workflow?"

retrieved_chunks = base_retriver.invoke(query)

for i in range(len(retrieved_chunks)):
    print(f"Retrieved chunk {i+1}:")
    print(retrieved_chunks[i].page_content)
    print("\n")