import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
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
from langchain_community.graphs import Neo4jGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import numpy as np
import json

# Configure logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

if GOOGLE_API_KEY:
    logger.info(f"Loaded GOOGLE_API_KEY: {GOOGLE_API_KEY[:5]}...{GOOGLE_API_KEY[-3:]}")
else:
    logger.error("GOOGLE_API_KEY is missing!")

# Neo4j Setup
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

try:
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    logger.info("Neo4j Knowledge Graph connected")
except Exception as e:
    logger.warning(f"Neo4j connection failed (optional): {str(e)}")
    graph = None

# LangGraph State Definition
class State(TypedDict):
    messages: Annotated[List[BaseMessage], "The list of messages in the conversation"]

# Define the node that calls the model
def call_model(state: State):
    # Construct the prompt with Long-Term Context (if available)
    # Note: We can pass ltm_context through the state if needed
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Initialize LangGraph with MemorySaver for Short-Term Memory
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# MemorySaver acts as the short-term memory checkpointer
checkpointer = MemorySaver()
chat_graph = workflow.compile(checkpointer=checkpointer)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize AI components
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

# Service database
SERVICES_DB = {
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

DUMMY_API_KEY = "DUMMY_BOOKING_123"


# Routes
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat with the LLM with LangGraph (Short-Term) and Neo4j (Long-Term) Memory"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        session_id = data.get('session_id', 'default_user')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        logger.debug(f"Chat query: {query} (Session: {session_id})")
        
        # 1. Retrieve Long-Term Context (Neo4j Search)
        ltm_context = ""
        if graph:
            try:
                ltm_results = graph.query(
                    "MATCH (u:User {id: $session_id})-[:SENT]->(m:Message) "
                    "WHERE m.content CONTAINS $query OR m.response CONTAINS $query "
                    "RETURN m.content + ' ' + m.response AS content "
                    "ORDER BY m.timestamp DESC LIMIT 2",
                    {"session_id": session_id, "query": query[:15]}
                )
                if ltm_results:
                    ltm_context = "\nRelevant Past Knowledge: " + " ".join([r['content'] for r in ltm_results if r.get('content')])
            except Exception as e:
                logger.error(f"LTM Retrieval Error: {str(e)}")

        # 2. Prepare LangGraph Input
        # We append the user message and any LTM context to the state
        user_message_text = query
        if ltm_context:
            user_message_text = f"{ltm_context}\n\nUser Question: {query}"
        
        inputs = {"messages": [HumanMessage(content=user_message_text)]}
        config = {"configurable": {"thread_id": session_id}}
        
        # 3. Invoke LangGraph (Handles Short-Term Memory automatically via checkpointer)
        output = chat_graph.invoke(inputs, config=config)
        response_text = output["messages"][-1].content
        
        # 4. Save to Long-Term Memory (Neo4j)
        if graph:
            try:
                graph.query(
                    "MERGE (u:User {id: $session_id}) "
                    "CREATE (m:Message {content: $query, response: $response, timestamp: timestamp()}) "
                    "MERGE (u)-[:SENT]->(m)",
                    {"session_id": session_id, "query": query, "response": response_text}
                )
            except Exception as e:
                logger.error(f"Failed to save to Neo4j: {str(e)}")
        
        logger.debug(f"Chat response: {response_text[:100]}...")
        
        return jsonify({'response': response_text})
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['POST'])
def web_search():
    """Perform web search"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        logger.debug(f"Search query: {query}")
        
        # Get structured results instead of raw text
        results_dict = search.results(query)
        organic_results = results_dict.get('organic', [])
        
        logger.debug(f"Search found {len(organic_results)} organic results")
        
        return jsonify({'results': organic_results})
    
    except Exception as e:
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/pdf/summary', methods=['POST'])
def pdf_summary():
    """Summarize uploaded PDF"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.debug(f"Processing PDF: {filename}")
        
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        text = "".join(p.page_content for p in pages)
        
        resp = llm.invoke(
            f"Summarize the following PDF content in simple bullet points for a non-technical user:\n{text[:10000]}"
        )
        response_text = getattr(resp, "content", getattr(resp, "text", str(resp)))
        
        # Clean up uploaded file
        os.remove(filepath)
        
        logger.debug(f"PDF summary generated: {response_text[:100]}...")
        
        return jsonify({'response': response_text})
    
    except Exception as e:
        logger.error(f"Error in PDF summary: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/rag/query', methods=['POST'])
def rag_query():
    """Answer questions from uploaded PDF using RAG"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        question = request.form.get('question', '')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.debug(f"RAG query on PDF: {filename}, Question: {question}")
        
        loader = PyPDFLoader(filepath)
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
        response_text = getattr(resp, "content", getattr(resp, "text", str(resp)))
        
        # Clean up uploaded file
        os.remove(filepath)
        
        logger.debug(f"RAG response: {response_text[:100]}...")
        
        return jsonify({'response': response_text})
    
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/services/lookup', methods=['POST'])
def service_lookup():
    """Look up available services"""
    try:
        data = request.get_json()
        query = data.get('query', '').lower()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        logger.debug(f"Service lookup query: {query}")
        
        # Detect service type
        matched_type = None
        for key in SERVICES_DB.keys():
            if key in query:
                matched_type = key
                break
        
        if not matched_type:
            return jsonify({
                'error': 'Service not found. Try using words like catering, decoration, or photography.'
            }), 404
        
        # Optional location filter
        location = None
        if " in " in query:
            location = query.split(" in ", 1)[1].strip()
        
        results = SERVICES_DB[matched_type]
        
        if location:
            filtered = [
                s for s in results
                if location.lower() in s["location"].lower()
            ]
            if filtered:
                results = filtered
        
        logger.debug(f"Found {len(results)} services for {matched_type}")
        
        return jsonify({
            'service_type': matched_type,
            'services': results
        })
    
    except Exception as e:
        logger.error(f"Error in service lookup: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/services/book', methods=['POST'])
def book_service():
    """Book a service (dummy booking)"""
    try:
        data = request.get_json()
        service_name = data.get('service_name', '')
        user_name = data.get('user_name', '')
        api_key = data.get('api_key', '')
        
        if not all([service_name, user_name, api_key]):
            return jsonify({'error': 'All fields are required'}), 400
        
        logger.debug(f"Booking request: {service_name} by {user_name}")
        
        if api_key != DUMMY_API_KEY:
            return jsonify({'error': 'Invalid API key'}), 401
        
        booking_id = f"BOOK-{np.random.randint(1000, 9999)}"
        
        logger.info(f"Booking successful: {booking_id}")
        
        return jsonify({
            'status': 'success',
            'booking_id': booking_id,
            'service': service_name,
            'user': user_name,
            'message': 'Service booked successfully (Dummy Booking)'
        })
    
    except Exception as e:
        logger.error(f"Error in booking: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting SmartBot Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
