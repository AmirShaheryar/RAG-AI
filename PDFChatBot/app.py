import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import langchain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import MultiQueryRetriever

docs=PyPDFLoader("Health_RAG_Test_Document.pdf").load()

#print(docs[0].page_content)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
    )
splitted_text=text_splitter.split_documents(docs)

#print(splitted_text[0].page_content)


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
    )

llm = ChatOllama(model="llama3", temperature=0)

vectorstore = FAISS.from_documents(splitted_text, embeddings)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever, 
    llm=llm
)   

prompt_template = """You are an AI assistant answering questions about a document.
Answer the question using strictly the provided context. If the answer cannot be determined from the context, say "Information not found in document."

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
    "Why is working out good for your heart and body?",
    
    "How can someone avoid getting sick according to the document?",
    
    "What should I eat and drink to keep my body healthy?",
    
    "What daily habits help me stay emotionally strong?",

    "What are the best outdoor sports mentioned in the PDF?"

]

for q in test_questions:
    print(f"\n Question: {q}")
    response = rag_chain.invoke(q)
    print(f" Response: {response}")
    print("-" * 50)