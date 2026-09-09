"""
Enhanced SmartBot with Short-Term + Long-Term Memory using PostgreSQL.

Differences from main.py:
  - Short-Term Memory: PostgresSaver (persistent conversation checkpoints)
  - Long-Term Memory: PostgresStore (user facts, preferences, entities)
  - Neo4j is optional and can coexist
  - MemoryManager provides a clean unified interface
  - JWT user authentication (register/login)

Install:
  pip install psycopg[binary] langgraph-checkpoint-postgres langgraph-store-postgres pyjwt bcrypt

Environment Variables (add to .env):
  DATABASE_URI=postgresql://postgres:postgres@localhost:5432/langgraph_memory
  JWT_SECRET=your-random-secret-key
  NEO4J_URI=bolt://localhost:7687       (optional)
  NEO4J_USERNAME=neo4j                  (optional)
  NEO4J_PASSWORD=password               (optional)
"""

import os
import logging
import asyncio
from flask import Flask, render_template, request, jsonify, g
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.graphs import Neo4jGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import random

# ── Import the new memory system ──────────────────────────────────
from memory_system import MemoryManager, sync_connect

# ── Import auth ───────────────────────────────────────────────────
from auth import require_auth, init_auth, decode_token

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

# ── Optional Neo4j Graph ─────────────────────────────────────────
try:
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    logger.info("Neo4j Knowledge Graph connected")
except Exception as e:
    logger.warning(f"Neo4j connection failed (optional): {str(e)}")
    graph = None

# ── LangGraph State ───────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[List[BaseMessage], "The list of messages in the conversation"]

# ── AI Components ─────────────────────────────────────────────────
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY,
    )
    search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
    logger.info("AI components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing AI components: {str(e)}")
    raise

# ── Initialise Memory Manager (Postgres STM + LTM) ───────────────
memory_manager = MemoryManager()

checkpointer = None
try:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(memory_manager.connect())
    loop.close()
    checkpointer = memory_manager.get_checkpointer()
    logger.info("Postgres memory system active (STM=PostgresSaver, LTM=PostgresStore)")
except Exception as e:
    logger.warning(f"Postgres memory unavailable, falling back to MemorySaver: {e}")
    checkpointer = MemorySaver()

# ── Build LangGraph ───────────────────────────────────────────────
def call_model(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

chat_graph = workflow.compile(checkpointer=checkpointer)

# ── Flask App ─────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Initialise Auth (creates users table + registers blueprint) ───
init_auth(app)

# ── Helper: resolve session_id from request ───────────────────────
def resolve_session_id() -> str:
    """Return user-specific session_id if authenticated, else fallback."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = decode_token(token)
        if payload:
            return f"user_{payload['user_id']}"
    # Fallback: use provided session_id or default
    data = request.get_json(silent=True) or {}
    return data.get("session_id", "default_user")


# ── Services Database ─────────────────────────────────────────────
SERVICES_DB = {
    "catering": [
        {"name": "A1 Catering", "location": "Chennai", "contact": "+91-98765-11111",
         "maps_url": "https://www.google.com/maps?q=A1+Catering+Chennai"},
        {"name": "FoodZone", "location": "Chennai", "contact": "+91-98765-22222",
         "maps_url": "https://www.google.com/maps?q=FoodZone+Chennai"},
    ],
    "decoration": [
        {"name": "EventDecor Pro", "location": "Chennai", "contact": "+91-98765-33333",
         "maps_url": "https://www.google.com/maps?q=EventDecor+Pro+Chennai"},
        {"name": "FlowerArt", "location": "Chennai", "contact": "+91-98765-44444",
         "maps_url": "https://www.google.com/maps?q=FlowerArt+Chennai"},
    ],
    "photography": [
        {"name": "LensCraft", "location": "Chennai", "contact": "+91-98765-55555",
         "maps_url": "https://www.google.com/maps?q=LensCraft+Chennai"},
        {"name": "WeddingFrames", "location": "Chennai", "contact": "+91-98765-66666",
         "maps_url": "https://www.google.com/maps?q=WeddingFrames+Chennai"},
    ],
}

DUMMY_API_KEY = "DUMMY_BOOKING_123"


# ═══════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
@require_auth
def chat():
    """
    Chat endpoint with Postgres-backed memory.

    Protected by JWT auth. The authenticated user's ID is used as the
    session_id automatically (user_{user_id}).
    """
    try:
        data = request.get_json()
        query = data.get("query", "")
        session_id = f"user_{g.user_id}"

        if not query:
            return jsonify({"error": "Query is required"}), 400

        logger.debug(f"Chat query: {query}  (session={session_id})")

        # ── 1. Retrieve LTM context ────────────────────────────────
        ltm_context = ""
        if memory_manager.ltm.store is not None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                ltm_context = loop.run_until_complete(
                    memory_manager.ltm.build_context_prompt(session_id, query)
                )
                loop.close()
            except Exception as e:
                logger.error(f"LTM context error: {e}")

        # ── 2. Inject LTM context into the user message ────────────
        augmented_query = query
        if ltm_context:
            augmented_query = f"{ltm_context}\n\nUser Question: {query}"

        # ── 3. Invoke LangGraph (STM handled by PostgresSaver) ─────
        inputs = {"messages": [HumanMessage(content=augmented_query)]}
        config = {"configurable": {"thread_id": session_id}}
        output = chat_graph.invoke(inputs, config=config)
        response_text = output["messages"][-1].content

        # ── 4. Extract & persist LTM facts ─────────────────────────
        if memory_manager.ltm.store is not None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    memory_manager.ltm.extract_and_store_facts(
                        session_id, query, response_text
                    )
                )
                loop.close()
            except Exception as e:
                logger.error(f"LTM fact extraction error: {e}")

        # ── 5. Optional Neo4j sync ─────────────────────────────────
        if graph:
            try:
                graph.query(
                    "MERGE (u:User {id: $session_id}) "
                    "CREATE (m:Message {content: $query, response: $response, timestamp: timestamp()}) "
                    "MERGE (u)-[:SENT]->(m)",
                    {"session_id": session_id, "query": query, "response": response_text},
                )
            except Exception as e:
                logger.error(f"Neo4j save error: {e}")

        logger.debug(f"Chat response: {response_text[:100]}...")
        return jsonify({"response": response_text})

    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory/facts", methods=["GET"])
@require_auth
def get_user_facts():
    """Return stored LTM facts for the authenticated user."""
    session_id = f"user_{g.user_id}"
    if memory_manager.ltm.store is None:
        return jsonify({"facts": {}, "note": "LTM not available"})

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        keys = loop.run_until_complete(
            memory_manager.ltm.list_keys(session_id)
        )
        info = {}
        for k in keys:
            val = loop.run_until_complete(
                memory_manager.ltm.get(session_id, k)
            )
            info[k] = val
        loop.close()
        return jsonify({"facts": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory/clear", methods=["POST"])
@require_auth
def clear_user_memory():
    """Clear all LTM data for the authenticated user."""
    session_id = f"user_{g.user_id}"
    if memory_manager.ltm.store is None:
        return jsonify({"status": "LTM not available"})

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        keys = loop.run_until_complete(
            memory_manager.ltm.list_keys(session_id)
        )
        for k in keys:
            loop.run_until_complete(
                memory_manager.ltm.delete(session_id, k)
            )
        loop.close()
        return jsonify({"status": "cleared", "keys_removed": len(keys)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Existing endpoints (unchanged from main.py) ──────────────────

@app.route("/api/search", methods=["POST"])
def web_search():
    try:
        data = request.get_json()
        query = data.get("query", "")
        if not query:
            return jsonify({"error": "Query is required"}), 400

        results_dict = search.results(query)
        organic_results = results_dict.get("organic", [])
        return jsonify({"results": organic_results})
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/pdf/summary", methods=["POST"])
def pdf_summary():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not file.filename.endswith(".pdf"):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        loader = PyPDFLoader(filepath)
        pages = loader.load()
        text = "".join(p.page_content for p in pages)

        resp = llm.invoke(
            f"Summarize the following PDF content in simple bullet points "
            f"for a non-technical user:\n{text[:10000]}"
        )
        response_text = getattr(resp, "content", getattr(resp, "text", str(resp)))
        os.remove(filepath)
        return jsonify({"response": response_text})
    except Exception as e:
        logger.error(f"PDF summary error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/query", methods=["POST"])
def rag_query():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        question = request.form.get("question", "")
        if not question:
            return jsonify({"error": "Question is required"}), 400
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not file.filename.endswith(".pdf"):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        loader = PyPDFLoader(filepath)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        db = FAISS.from_documents(chunks, embeddings)
        retrieved_docs = db.similarity_search(question, k=3)
        context = "\n\n".join(d.page_content for d in retrieved_docs)

        resp = llm.invoke(
            f"Use the following PDF context to answer the question.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        )
        response_text = getattr(resp, "content", getattr(resp, "text", str(resp)))
        os.remove(filepath)
        return jsonify({"response": response_text})
    except Exception as e:
        logger.error(f"RAG error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/services/lookup", methods=["POST"])
def service_lookup():
    try:
        data = request.get_json()
        query = data.get("query", "").lower()
        if not query:
            return jsonify({"error": "Query is required"}), 400

        matched_type = None
        for key in SERVICES_DB:
            if key in query:
                matched_type = key
                break

        if not matched_type:
            return jsonify({
                "error": "Service not found. Try using words like catering, decoration, or photography."
            }), 404

        location = None
        if " in " in query:
            location = query.split(" in ", 1)[1].strip()

        results = SERVICES_DB[matched_type]
        if location:
            filtered = [s for s in results if location.lower() in s["location"].lower()]
            if filtered:
                results = filtered

        return jsonify({"service_type": matched_type, "services": results})
    except Exception as e:
        logger.error(f"Service lookup error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/services/book", methods=["POST"])
def book_service():
    try:
        data = request.get_json()
        service_name = data.get("service_name", "")
        user_name = data.get("user_name", "")
        api_key = data.get("api_key", "")

        if not all([service_name, user_name, api_key]):
            return jsonify({"error": "All fields are required"}), 400

        if api_key != DUMMY_API_KEY:
            return jsonify({"error": "Invalid API key"}), 401

        booking_id = f"BOOK-{random.randint(1000, 9999)}"
        return jsonify({
            "status": "success",
            "booking_id": booking_id,
            "service": service_name,
            "user": user_name,
            "message": "Service booked successfully (Dummy Booking)",
        })
    except Exception as e:
        logger.error(f"Booking error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting SmartBot v2 with PG STM+LTM memory + JWT auth...")
    app.run(debug=True, host="0.0.0.0", port=5000)
