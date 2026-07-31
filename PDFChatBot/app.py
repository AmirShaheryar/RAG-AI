import html
import os
import re
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


def format_docs_with_ids(docs):
    return "\n\n".join(
        f"[{i}] (Page {doc.metadata.get('page', 0) + 1})\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )


def extract_cited_indices(answer_text, max_index):
    indices = {int(match) for match in re.findall(r"\[(\d+)\]", answer_text)}
    return sorted(index for index in indices if 1 <= index <= max_index)


def find_matching_spans(chunk_text, answer_text, min_words=4):
    chunk_lower = chunk_text.lower()
    answer_words = re.sub(r"\[\d+\]", "", answer_text.lower()).split()
    spans = []

    for window_size in range(min(20, len(answer_words)), min_words - 1, -1):
        for start_idx in range(len(answer_words) - window_size + 1):
            phrase = " ".join(answer_words[start_idx : start_idx + window_size])
            if len(phrase) < 20:
                continue

            search_from = 0
            while True:
                match_idx = chunk_lower.find(phrase, search_from)
                if match_idx == -1:
                    break
                spans.append((match_idx, match_idx + len(phrase)))
                search_from = match_idx + 1

    if not spans:
        return []

    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def highlight_text_in_chunk(chunk_text, answer_text):
    spans = find_matching_spans(chunk_text, answer_text)
    if not spans:
        return html.escape(chunk_text)

    highlighted_parts = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            highlighted_parts.append(html.escape(chunk_text[cursor:start]))
        match_text = html.escape(chunk_text[start:end])
        highlighted_parts.append(
            f'<mark style="background-color:#ffeb3b;padding:1px 3px;border-radius:2px;">'
            f"{match_text}</mark>"
        )
        cursor = end

    if cursor < len(chunk_text):
        highlighted_parts.append(html.escape(chunk_text[cursor:]))

    return "".join(highlighted_parts)


def get_relevant_chunk_indices(source_docs, answer_text, max_chunks=2):
    cited = extract_cited_indices(answer_text, len(source_docs))
    if cited:
        return cited

    scored_chunks = []
    for index, doc in enumerate(source_docs, 1):
        overlap = sum(end - start for start, end in find_matching_spans(doc.page_content, answer_text))
        if overlap:
            scored_chunks.append((overlap, index))

    if scored_chunks:
        scored_chunks.sort(reverse=True)
        return [index for _, index in scored_chunks[:max_chunks]]

    return [1]


def render_citations(answer_text, source_docs):
    cited_indices = get_relevant_chunk_indices(source_docs, answer_text)

    with st.expander("📚 View Citations"):
        for index in cited_indices:
            doc = source_docs[index - 1]
            page_num = doc.metadata.get("page", 0) + 1
            st.markdown(f"**Source [{index}] — Page {page_num}**")
            highlighted = highlight_text_in_chunk(doc.page_content, answer_text)
            st.markdown(
                (
                    '<div style="background:#f0f2f6;padding:12px;border-radius:8px;'
                    'border-left:4px solid #4CAF50;line-height:1.6;">'
                    f"{highlighted}</div>"
                ),
                unsafe_allow_html=True,
            )


@st.cache_resource(show_spinner="Processing and Indexing the PDF document...")
def process_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
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

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

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

When you use information from a source, cite it inline using the source number in square brackets, e.g. [1], [2].

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate.from_template(prompt_template)

    rag_chain = (
    RunnablePassthrough.assign(
        source_docs=lambda x: retriever.invoke(x["question"])
    )
    | RunnablePassthrough.assign(
        context=lambda x: format_docs_with_ids(x["source_docs"])
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
                render_citations(message["content"], message["sources"])

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
                    render_citations(answer_text, sources)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer_text,
            "sources": sources
        })

else:

    st.info(" Please upload a PDF in the sidebar to get started.")