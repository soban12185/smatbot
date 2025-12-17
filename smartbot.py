import os
import streamlit as st

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import numpy as np


# -----------------------------
# LOAD .env VARIABLES
# -----------------------------
load_dotenv()  # loads variables from .env into environment

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")


# -----------------------------
# SETUP EMBEDDINGS (Gemini)
# -----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
)


# -----------------------------
# SETUP LLM (Gemini)
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY,
)


# -----------------------------
# 1. SEARCH TOOL (Serper)
# -----------------------------
search = GoogleSerperAPIWrapper(
    serper_api_key=SERPER_API_KEY
)


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

    db = FAISS.from_documents(chunks, embeddings)

    retrieved_docs = db.similarity_search(question, k=3)
    context = "\n\n".join(d.page_content for d in retrieved_docs)

    resp = llm.invoke(
        f"Use the following PDF context to answer the question.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    return getattr(resp, "content", getattr(resp, "text", str(resp)))


# -----------------------------
# 4. SERVICE LOOKUP (with location, contact, Google Maps)
# -----------------------------
def service_lookup(query: str) -> str:
    services = {
        "catering": [
            {
                "name": "A1 Catering",
                "location": "Chennai",
                "contact": "+91-98765-11111",
                "maps_url": "https://www.google.com/maps?q=A1+Catering+Chennai",
            },
            {
                "name": "FoodZone",
                "location": "Chennai",
                "contact": "+91-98765-22222",
                "maps_url": "https://www.google.com/maps?q=FoodZone+Chennai",
            },
        ],
        "decoration": [
            {
                "name": "EventDecor Pro",
                "location": "Chennai",
                "contact": "+91-98765-33333",
                "maps_url": "https://www.google.com/maps?q=EventDecor+Pro+Chennai",
            },
            {
                "name": "FlowerArt",
                "location": "Chennai",
                "contact": "+91-98765-44444",
                "maps_url": "https://www.google.com/maps?q=FlowerArt+Chennai",
            },
        ],
        "photography": [
            {
                "name": "LensCraft",
                "location": "Chennai",
                "contact": "+91-98765-55555",
                "maps_url": "https://www.google.com/maps?q=LensCraft+Chennai",
            },
            {
                "name": "WeddingFrames",
                "location": "Chennai",
                "contact": "+91-98765-66666",
                "maps_url": "https://www.google.com/maps?q=WeddingFrames+Chennai",
            },
        ],
    }

    q = query.lower()

    # detect service type
    matched_type = None
    for key in services.keys():
        if key in q:
            matched_type = key
            break

    if not matched_type:
        return "Service not found, try using words like catering, decoration, or photography."

    # optional location filter, e.g. "catering in chennai"
    location = None
    if " in " in q:
        location = q.split(" in ", 1)[1].strip()

    results = services[matched_type]

    if location:
        filtered = [
            s for s in results
            if location.lower() in s["location"].lower()
        ]
        if filtered:
            results = filtered

    lines = [f"**Available {matched_type} services:**"]
    for s in results:
        lines.append(
            f"- **{s['name']}**  \n"
            f"  Location: {s['location']}  \n"
            f"  Contact: {s['contact']}  \n"
            f"  [View directions on Google Maps]({s['maps_url']})"
        )

    return "\n".join(lines)


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🤖 SmartBot – Multi Tool AI Assistant ")

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
        st.write(web_search.invoke(query))

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
            
# -----------------------------
# 5. DUMMY BOOKING API
# -----------------------------
DUMMY_API_KEY = "DUMMY_BOOKING_123"

def dummy_book_service(service_name: str, user_name: str, api_key: str):
    if api_key != DUMMY_API_KEY:
        return {
            "status": "failed",
            "message": "Invalid API key"
        }

    booking_id = f"BOOK-{np.random.randint(1000, 9999)}"

    return {
        "status": "success",
        "booking_id": booking_id,
        "service": service_name,
        "user": user_name,
        "message": "Service booked successfully (Dummy Booking)"
    }

# SERVICE LOOKUP + DUMMY BOOKING
if mode == "Service Lookup":
    query = st.text_input("Enter service needed (e.g., catering in chennai):")

    if query:
        st.markdown(service_lookup(query))

        st.divider()
        st.subheader("📅 Book a Service (Dummy Booking)")

        user_name = st.text_input("Your Name")
        service_name = st.text_input("Service Name to Book")
        api_key = st.text_input("API Key", value="DUMMY_BOOKING_123")

        if st.button("✅ Book Service"):
            if not user_name or not service_name:
                st.warning("Please enter all details.")
            else:
                result = dummy_book_service(
                    service_name=service_name,
                    user_name=user_name,
                    api_key=api_key
                )

                if result["status"] == "success":
                    st.success("🎉 Booking Confirmed!")
                    st.write(f"**Booking ID:** {result['booking_id']}")
                    st.write(f"**Service:** {result['service']}")
                    st.write(f"**Booked By:** {result['user']}")
                else:
                    st.error(result["message"])

