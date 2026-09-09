"""
JWT-based User Authentication System.

Provides:
  - User registration & login
  - JWT token generation & verification
  - @require_auth decorator for protected routes
  - User table auto-creation in PostgreSQL

Requires:
  pip install pyjwt bcrypt
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional

import bcrypt
import jwt
from flask import request, jsonify, g

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────
JWT_SECRET = os.environ.get("JWT_SECRET", "change-this-to-a-random-secret-key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 72  # 3 days

# ── DB helpers (sync, using psycopg) ──────────────────────────────
_conn_pool = None


def get_db_connection():
    """Return a psycopg connection using DATABASE_URI."""
    import psycopg

    global _conn_pool
    if _conn_pool is None:
        db_url = os.environ.get(
            "DATABASE_URI",
            "postgresql://postgres:postgres@localhost:5432/langgraph_memory",
        )
        _conn_pool = psycopg.connect(db_url)
    return _conn_pool


def init_users_table():
    """Create the users table if it doesn't exist."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                last_login TIMESTAMP
            );
        """)
        conn.commit()
        cur.close()
        logger.info("Users table ready.")
    except Exception as e:
        logger.warning(f"Could not init users table: {e}")


# ── Password utilities ────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


# ── JWT utilities ─────────────────────────────────────────────────
def create_token(user_id: int, username: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ── Decorator ─────────────────────────────────────────────────────
def require_auth(f):
    """Decorator: protects a Flask route with JWT auth.

    Sets g.user_id and g.username on success.
    Returns 401 JSON if token missing or invalid.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = None
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

        if not token:
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        payload = decode_token(token)
        if payload is None:
            return jsonify({"error": "Token expired or invalid"}), 401

        g.user_id = payload["user_id"]
        g.username = payload["username"]
        return f(*args, **kwargs)

    return wrapper


# ── Flask Blueprint ───────────────────────────────────────────────
from flask import Blueprint

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/api/auth/register", methods=["POST"])
def register():
    """Register a new user.
    Body: { "username": "...", "email": "...", "password": "..." }
    """
    try:
        data = request.get_json()
        username = data.get("username", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        if not username or not email or not password:
            return jsonify({"error": "username, email, and password are required"}), 400

        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400

        conn = get_db_connection()
        cur = conn.cursor()

        # Check duplicates
        cur.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
        if cur.fetchone():
            cur.close()
            return jsonify({"error": "Username or email already taken"}), 409

        pwd_hash = hash_password(password)
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s) RETURNING id",
            (username, email, pwd_hash),
        )
        user_id = cur.fetchone()[0]
        conn.commit()
        cur.close()

        token = create_token(user_id, username)
        logger.info(f"User registered: {username} (id={user_id})")
        return jsonify({"message": "Registration successful", "token": token, "user_id": user_id, "username": username}), 201

    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    """Login with username/email and password.
    Body: { "username": "...", "password": "..." }
    """
    try:
        data = request.get_json()
        login_id = data.get("username", "").strip()
        password = data.get("password", "")

        if not login_id or not password:
            return jsonify({"error": "username and password are required"}), 400

        conn = get_db_connection()
        cur = conn.cursor()

        # Allow login by username or email
        cur.execute(
            "SELECT id, username, password_hash FROM users WHERE username = %s OR email = %s",
            (login_id, login_id.lower()),
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            return jsonify({"error": "Invalid credentials"}), 401

        user_id, username, pwd_hash = row

        if not check_password(password, pwd_hash):
            cur.close()
            return jsonify({"error": "Invalid credentials"}), 401

        # Update last_login
        cur.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (user_id,))
        conn.commit()
        cur.close()

        token = create_token(user_id, username)
        logger.info(f"User logged in: {username} (id={user_id})")
        return jsonify({"message": "Login successful", "token": token, "user_id": user_id, "username": username})

    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@auth_bp.route("/api/auth/me", methods=["GET"])
@require_auth
def me():
    """Return current user info. Requires Authorization header."""
    return jsonify({"user_id": g.user_id, "username": g.username})


def init_auth(app_obj):
    """Call this from the main app to register the blueprint and create the table."""
    init_users_table()
    app_obj.register_blueprint(auth_bp)
    logger.info("Auth blueprint registered.")
