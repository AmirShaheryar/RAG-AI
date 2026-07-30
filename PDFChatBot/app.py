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
    chunk_size=500, 
    chunk_overlap=100
    ) 

splitted_text=text_splitter.split_documents(docs)

#print(splitted_text[0].page_content)


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
    )

vectorstore = FAISS.from_documents(splitted_text, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatOllama(model="llama3", temperature=0)

prompt_template = """You are an AI assistant answering questions about a PDF document.
Use ONLY the provided context to answer the question. If you don't know or if it's not mentioned in the context, say "Information not found in document."

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(prompt_template)



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

test_questions = [
    "What daily lifestyle choices contribute to good mental health according to the text?",
    "What are the key elements included in preventive care?",
    "What role does nutrition and exercise play in overall physical wellness according to the document?",
    "How many hours of sleep should a person get each night?",
    "What type of medicine should I take for a fever?"
    "Tell me about the importance of regular health check-ups and screenings.",
    "What are the recommended vaccinations for adults?",
    "What are the common symptoms of stress and how can they be managed?",
    "Document is about health and wellness. Can you summarize the main points discussed in the document?",
]

for q in test_questions:
    print(f"\n Question: {q}")
    response = rag_chain.invoke(q)
    print(f" Response: {response}")
    print("-" * 50)