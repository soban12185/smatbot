"""
Memory System for Vercel Serverless:
  - Short-Term Memory: PostgresSaver (conversation checkpoints)
  - Long-Term Memory:  Raw psycopg table (user facts, preferences)

Designed for ephemeral Lambda: lazy connect per request, connection pool.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

DATABASE_URI = os.environ.get(
    "DATABASE_URI",
    "postgresql://postgres:postgres@localhost:5432/langgraph_memory",
)

_pool: Optional[ConnectionPool] = None


def _get_pool() -> ConnectionPool:
    """Lazy singleton connection pool — safe for Vercel Lambda."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            DATABASE_URI,
            min_size=1,
            max_size=5,
            kwargs={"row_factory": dict_row, "sslmode": "require"},
        )
        _init_tables()
    return _pool


def _init_tables():
    """Create tables if they don't exist."""
    pool = _pool
    with pool.connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ltm_facts (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(session_id, key)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ltm_conversations (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ltm_facts_session ON ltm_facts(session_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ltm_conv_session ON ltm_conversations(session_id)
        """)
        conn.commit()
    logger.info("Memory tables ready.")


# ===================================================================
# Short-Term Memory (PostgresSaver)
# ===================================================================
def get_checkpointer():
    """Return a PostgresSaver checkpointer for LangGraph."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        return PostgresSaver.from_conn_string(DATABASE_URI)
    except Exception as e:
        logger.warning(f"PostgresSaver unavailable: {e}")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


# ===================================================================
# Long-Term Memory (raw psycopg)
# ===================================================================
class LongTermMemory:
    """Simple Postgres-backed key-value store for user facts."""

    def put(self, session_id: str, key: str, value: dict[str, Any]):
        pool = _get_pool()
        with pool.connection() as conn:
            conn.execute("""
                INSERT INTO ltm_facts (session_id, key, value, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (session_id, key)
                DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
            """, (session_id, key, json.dumps(value)))
            conn.commit()

    def get(self, session_id: str, key: str) -> Optional[dict[str, Any]]:
        pool = _get_pool()
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT value FROM ltm_facts WHERE session_id = %s AND key = %s",
                (session_id, key),
            ).fetchone()
        return row["value"] if row else None

    def delete(self, session_id: str, key: str):
        pool = _get_pool()
        with pool.connection() as conn:
            conn.execute(
                "DELETE FROM ltm_facts WHERE session_id = %s AND key = %s",
                (session_id, key),
            )
            conn.commit()

    def list_keys(self, session_id: str) -> list[str]:
        pool = _get_pool()
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT key FROM ltm_facts WHERE session_id = %s ORDER BY updated_at DESC",
                (session_id,),
            ).fetchall()
        return [r["key"] for r in rows]

    def search(self, session_id: str, text: str) -> list[dict]:
        pool = _get_pool()
        with pool.connection() as conn:
            rows = conn.execute(
                """SELECT key, value FROM ltm_facts
                   WHERE session_id = %s AND value::text ILIKE %s
                   LIMIT 5""",
                (session_id, f"%{text}%"),
            ).fetchall()
        return [{"key": r["key"], "value": r["value"]} for r in rows]

    def save_conversation(self, session_id: str, query: str, response: str):
        pool = _get_pool()
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO ltm_conversations (session_id, query, response) VALUES (%s, %s, %s)",
                (session_id, query, response),
            )
            conn.commit()

    def get_recent_conversations(self, session_id: str, limit: int = 10) -> list[dict]:
        pool = _get_pool()
        with pool.connection() as conn:
            rows = conn.execute(
                """SELECT query, response, created_at FROM ltm_conversations
                   WHERE session_id = %s ORDER BY created_at DESC LIMIT %s""",
                (session_id, limit),
            ).fetchall()
        return rows

    def build_context(self, session_id: str, query: str) -> str:
        """Build LTM context string for LLM prompts."""
        parts = []

        prefs = self.get(session_id, "preferences")
        if prefs:
            parts.append(f"User Preferences: {json.dumps(prefs)}")

        facts = self.get(session_id, "facts")
        if facts and isinstance(facts, list):
            parts.append("Known Facts:\n" + "\n".join(f"- {f}" for f in facts[-10:]))

        hits = self.search(session_id, query)
        if hits:
            parts.append(f"Relevant Past: {json.dumps(hits[:3], default=str)}")

        recent = self.get_recent_conversations(session_id, 3)
        if recent:
            convos = "\n".join(
                f"User: {r['query']}\nBot: {r['response'][:200]}" for r in reversed(recent)
            )
            parts.append(f"Recent Conversation:\n{convos}")

        if not parts:
            return ""
        return "\n\n--- Long-Term Memory ---\n" + "\n\n".join(parts)

    def extract_facts(self, session_id: str, query: str, response: str):
        """Store conversation and extract facts."""
        self.save_conversation(session_id, query, response)

        existing = self.get(session_id, "facts") or []
        interaction = f"{query[:100]} -> {response[:100]}"
        existing.append(interaction)
        existing = existing[-50:]
        self.put(session_id, "facts", existing)

    def clear_all(self, session_id: str):
        """Clear all data for a session."""
        pool = _get_pool()
        with pool.connection() as conn:
            conn.execute("DELETE FROM ltm_facts WHERE session_id = %s", (session_id,))
            conn.execute("DELETE FROM ltm_conversations WHERE session_id = %s", (session_id,))
            conn.commit()


ltm = LongTermMemory()
