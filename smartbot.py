import streamlit as st

from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


# -----------------------------
# SETUP LLM (OpenAI, API key only)
# -----------------------------
# Before running: set environment variable OPENAI_API_KEY
#   PowerShell:  $env:OPENAI_API_KEY="YOUR_OPENAI_KEY"
from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ["GOOGLE_API_KEY"] = ""
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7
)


# -----------------------------
# 1. SEARCH TOOL
# -----------------------------
# Needs SERPER_API_KEY env var:
#   PowerShell:  $env:SERPER_API_KEY="YOUR_SERPER_KEY"
search = GoogleSerperAPIWrapper()

@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    return search.run(query)


# -----------------------------
# 2. PDF SUMMARIZATION
# -----------------------------
def summarize_pdf(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text = "".join(p.page_content for p in pages)
    resp = llm.invoke(
        f"Summarize the following PDF content in simple bullet points for a non-technical user:\n{text}"
    )
    return getattr(resp, "content", getattr(resp, "text", str(resp)))


# -----------------------------
# 3. RAG TOOL (PDF Q&A)
# -----------------------------
def rag_query(question: str, pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

    retrieved_docs = db.similarity_search(question, k=3)
    context = "\n\n".join(d.page_content for d in retrieved_docs)

    resp = llm.invoke(
        f"Use the following PDF context to answer the question.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    return getattr(resp, "content", getattr(resp, "text", str(resp)))


# -----------------------------
# 4. SERVICE LOOKUP
# -----------------------------
def service_lookup(query: str) -> str:
    database = {
        "catering": "Top catering services near you: A1 Catering, FoodZone, ChefMate.",
        "decoration": "Decoration services: EventDecor Pro, FlowerArt, Shine Events.",
        "photography": "Photographers: LensCraft, PixelShot, WeddingFrames.",
    }
    q = query.lower()
    for key, value in database.items():
        if key in q:
            return value
    return "Service not found, try querying catering, decoration, or photography."


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🤖 SmartBot – Multi Tool AI Assistant (LangChain + LLM)")

mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Chat", "Web Search", "PDF Summary", "RAG Q&A", "Service Lookup"],
)

# CHAT MODE (pure LLM)
if mode == "Chat":
    query = st.text_input("Ask SmartBot anything:")
    if query:
        resp = llm.invoke(query)
        st.write(getattr(resp, "content", getattr(resp, "text", str(resp))))

# WEB SEARCH
elif mode == "Web Search":
    query = st.text_input("Enter topic to search:")
    if query:
        st.write(web_search(query))

# PDF SUMMARY
elif mode == "PDF Summary":
    file = st.file_uploader("Upload PDF", type=["pdf"])
    if file:
        path = "uploaded.pdf"
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        st.write(summarize_pdf(path))

# RAG Q&A
elif mode == "RAG Q&A":
    file = st.file_uploader("Upload PDF", type=["pdf"])
    question = st.text_input("Ask a question from the PDF")
    if file:
        path = "rag.pdf"
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        if question:
            st.write(rag_query(question, path))

# SERVICE LOOKUP
elif mode == "Service Lookup":
    query = st.text_input("Enter service needed:")
    if query:
        st.write(service_lookup(query))
