import os
import time
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from memory_system import ltm
from pdf_analyzer import analyze_pdf, search as pdf_search

RATE_LIMIT_PER_MINUTE = int(os.environ.get('RATE_LIMIT_PER_MINUTE', '10'))
RATE_LIMIT_PER_DAY = int(os.environ.get('RATE_LIMIT_PER_DAY', '1000'))

_request_log = []

def _check_rate_limit():
    now = time.time()
    _request_log.append(now)
    _request_log[:] = [t for t in _request_log if now - t < 86400]
    if len(_request_log) > RATE_LIMIT_PER_DAY:
        return False, f"Daily limit of {RATE_LIMIT_PER_DAY} requests reached. Try again tomorrow."
    recent = [t for t in _request_log if now - t < 60]
    if len(recent) > RATE_LIMIT_PER_MINUTE:
        return False, f"Rate limit of {RATE_LIMIT_PER_MINUTE} requests/minute reached. Wait a moment."
    return True, None

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.graphs import Neo4jGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import random
import json

# Configure logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

if GROQ_API_KEY:
    logger.info(f"Loaded GROQ_API_KEY: {GROQ_API_KEY[:5]}...{GROQ_API_KEY[-3:]}")
else:
    logger.error("GROQ_API_KEY is missing!")

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
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', '/tmp/uploads')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize AI components
try:
    llm = ChatOpenAI(
        model="openai/gpt-oss-120b",
        temperature=0.7,
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
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
    """Chat with the LLM with Postgres-backed Long-Term Memory"""
    try:
        ok, msg = _check_rate_limit()
        if not ok:
            return jsonify({'error': msg}), 429

        data = request.get_json()
        query = data.get('query', '')
        session_id = data.get('session_id', 'default_user')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # 1. Retrieve LTM context from Postgres
        ltm_context = ""
        try:
            ltm_context = ltm.build_context(session_id, query)
        except Exception as e:
            logger.error(f"LTM retrieval error: {e}")

        # 2. Build prompt with LTM
        user_message_text = query
        if ltm_context:
            user_message_text = f"{ltm_context}\n\nUser Question: {query}"
        
        inputs = {"messages": [HumanMessage(content=user_message_text)]}
        config = {"configurable": {"thread_id": session_id}}
        
        # 3. Invoke LangGraph
        output = chat_graph.invoke(inputs, config=config)
        response_text = output["messages"][-1].content
        
        # 4. Save to LTM
        try:
            ltm.extract_facts(session_id, query, response_text)
        except Exception as e:
            logger.error(f"LTM save error: {e}")
        
        return jsonify({'response': response_text})
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/memory/facts', methods=['GET'])
def get_memory_facts():
    """Return stored LTM facts for a session."""
    try:
        session_id = request.args.get('session_id', 'default_user')
        keys = ltm.list_keys(session_id)
        info = {}
        for k in keys:
            info[k] = ltm.get(session_id, k)
        return jsonify({'session_id': session_id, 'facts': info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/memory/history', methods=['GET'])
def get_memory_history():
    """Return recent conversation history for a session."""
    try:
        session_id = request.args.get('session_id', 'default_user')
        limit = int(request.args.get('limit', '10'))
        conversations = ltm.get_recent_conversations(session_id, limit)
        return jsonify({'session_id': session_id, 'history': conversations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/memory/clear', methods=['POST'])
def clear_memory():
    """Clear all LTM data for a session."""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', 'default_user')
        ltm.clear_all(session_id)
        return jsonify({'status': 'cleared', 'session_id': session_id})
    except Exception as e:
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
    """Analyze uploaded PDF — rule-based summary + FAISS index"""
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

        result = analyze_pdf(filepath, filename)

        os.remove(filepath)

        if "error" in result:
            return jsonify({'error': result['error']}), 400

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in PDF analysis: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to analyze PDF. Please try again.'}), 500


@app.route('/api/pdf/ask', methods=['POST'])
def pdf_ask():
    """Ask a question about the uploaded PDF"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        doc_id = data.get('doc_id', '')

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        if not doc_id:
            return jsonify({'error': 'Please upload a PDF first.'}), 400

        result = pdf_search(question, doc_id)

        if "error" in result:
            return jsonify({'error': result['error']}), 400

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in PDF search: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to search document. Please try again.'}), 500


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
        
        booking_id = f"BOOK-{random.randint(1000, 9999)}"
        
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


@app.route('/api/event/plan', methods=['POST'])
def event_plan():
    """Generate AI-powered event plan recommendations"""
    try:
        data = request.get_json()
        event_type = data.get('event_type', '')
        date = data.get('date', '')
        location = data.get('location', '')
        exactlocation = data.get('exactlocation', '')
        guest_count = int(data.get('guest_count', 0))
        total_budget = float(data.get('total_budget', 0))
        special_requirements = data.get('special_requirements', '')
        use_dummy = data.get('use_dummy', False)

        if not all([event_type, date, location, guest_count, total_budget]):
            return jsonify({'error': 'All fields are required'}), 400

        per_person_budget = total_budget / guest_count

        if use_dummy:
            food_cost = total_budget * 0.5
            decor_cost = total_budget * 0.2
            entertainment_cost = total_budget * 0.1
            buffer_budget = total_budget - (food_cost + decor_cost + entertainment_cost)

            response_text = (
                f"{event_type} Event Plan\n\n"
                f"Date: {date}\n"
                f"City: {location}\n"
                f"Venue Area: {exactlocation}\n"
                f"Total Guests: {guest_count}\n\n"
                f"--- Budget Overview ---\n"
                f"Total Budget: {total_budget:,.0f}\n"
                f"Estimated Cost Per Person: {per_person_budget:.2f}\n\n"
                f"--- Catering ---\n"
                f"South & North Indian buffet\n"
                f"Cost per plate: {per_person_budget * 0.5:.0f}\n"
                f"Total Catering Cost: {food_cost:,.0f}\n\n"
                f"--- Decoration ---\n"
                f"Floral stage decoration\n"
                f"Theme-based entrance\n"
                f"Decoration Cost: {decor_cost:,.0f}\n\n"
                f"--- Entertainment ---\n"
                f"DJ & traditional music\n"
                f"Cost: {entertainment_cost:,.0f}\n\n"
                f"--- Logistics ---\n"
                f"Guest transport, parking & coordination\n\n"
                f"--- Budget Summary ---\n"
                f"Food: {food_cost:,.0f}\n"
                f"Decoration: {decor_cost:,.0f}\n"
                f"Entertainment: {entertainment_cost:,.0f}\n"
                f"Buffer: {buffer_budget:,.0f}\n\n"
                f"--- Nearby Services ---\n"
                f"Catering: Royal Caterers, Annapoorna Foods\n"
                f"Decoration: Dream Decors, Floral Art Studio"
            )
            return jsonify({'response': response_text, 'per_person': per_person_budget})

        ok, msg = _check_rate_limit()
        if not ok:
            return jsonify({'error': msg}), 429

        prompt = f"""
You are an expert event planner. Provide a detailed event plan with venue, catering,
decoration, entertainment, logistics, budget breakdown, and nearby services.

Event Details:
- Event Type: {event_type}
- Date: {date}
- City: {location}
- Exact Location: {exactlocation}
- Total Guests: {guest_count}
- Total Budget: {total_budget}
- Per-Person Budget: {per_person_budget:.2f}
- Special Requirements: {special_requirements}
"""

        resp = llm.invoke(prompt)
        response_text = getattr(resp, "content", getattr(resp, "text", str(resp)))

        return jsonify({'response': response_text, 'per_person': per_person_budget})

    except Exception as e:
        logger.error(f"Error in event plan: {str(e)}", exc_info=True)
        err = str(e)
        if "RESOURCE_EXHAUSTED" in err or "429" in err:
            return jsonify({'error': 'AI service quota exhausted. Please try again later.'}), 429
        if "API_KEY_INVALID" in err or "403" in err:
            return jsonify({'error': 'AI service configuration error. Please contact support.'}), 403
        return jsonify({'error': 'Something went wrong while planning your event. Please try again.'}), 500


if __name__ == '__main__':
    logger.info("Starting SmartBot Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
