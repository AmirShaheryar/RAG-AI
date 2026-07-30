import os
import tempfile
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

import streamlit as st

st.set_page_config(page_title="Health RAG Test", page_icon=" 🏥 ")
st.title("Health RAG Test")

st.caption("This is a test of the Health RAG system using a PDF document.")

@st.cache_resource(show_spinner="Processing and Indexing the PDF document...")
def process_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # ORIGINAL LOGIC: Text loading, splitting, and vectorstore creation
        docs = PyPDFLoader(tmp_path).load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splitted_text = text_splitter.split_documents(docs)
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.from_documents(splitted_text, embeddings)
        return vectorstore
    finally:
        os.remove(tmp_path)

st.sidebar.header("Document Upload")

uploaded_file = st.sidebar.file_uploader("Upload a Health PDF", type=["pdf"])

if uploaded_file is not None:
    vectorstore = process_pdf(uploaded_file)
    
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(
        model="llama3", 
        temperature=0
        )
    retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    
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
    RunnablePassthrough.assign(
        source_docs=lambda x: retriever.invoke(x["question"])
    )
    | RunnablePassthrough.assign(
        context=lambda x: format_docs(x["source_docs"])
    )
    | RunnablePassthrough.assign(
        answer=prompt | llm | StrOutputParser()
    )
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sources" in message and message["sources"]:
                with st.expander(" View Retrieved Source Chunks"):
                    for idx, doc in enumerate(message["sources"], 1):
                        page_num = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**Chunk {idx} (Page {page_num}):**")
                        st.info(doc.page_content)

    if user_query := st.chat_input("Ask a question about the document..."):
        
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Generating rephrased queries & retrieving answer..."):
                result = rag_chain.invoke({"question": user_query})
                
                answer_text = result["answer"]
                sources = result["source_docs"]
                st.markdown(answer_text)

                if sources:
                    with st.expander("📚 View Retrieved Source Chunks"):
                        for idx, doc in enumerate(sources, 1):
                            page_num = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**Chunk {idx} (Page {page_num}):**")
                            st.info(doc.page_content)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer_text,
            "sources": sources
        })

else:

    st.info(" Please upload a PDF in the sidebar to get started.")