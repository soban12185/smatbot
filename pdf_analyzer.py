"""
PDF Analyzer — Hybrid RAG Pipeline with Jina Reranker

Pipeline
--------
1. PDF extraction (page-aware via PyMuPDF)
2. Text preprocessing
3. Structure-aware / sentence-aware chunking (page-by-page)
4. Jina embeddings (jina-embeddings-v3)
5. In-memory vector store
6. BM25 keyword retrieval
7. Hybrid retrieval (70% semantic + 30% BM25)
8. Jina hosted reranker API (top 20 → top 5)
9. Context building with metadata/citations
10. LLM generation (Groq primary, Gemini fallback)

Environment
-----------
Required:
    JINA_API_KEY

One LLM key:
    GROQ_API_KEY
    or GOOGLE_API_KEY
"""

import logging
import math
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE = 350
OVERLAP_SIZE = 60
RETRIEVAL_K = 20
FINAL_K = 5
MAX_PAGES = 20

JINA_EMBED_MODEL = "jina-embeddings-v3"
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-2.0-flash"

SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30

vector_store: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+(?:[-']\w+)*\b", text.lower())


def sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    protected = {
        "e.g.": "e\u00a7g\u00a7",
        "i.e.": "i\u00a7e\u00a7",
        "mr.": "mr\u00a7",
        "mrs.": "mrs\u00a7",
        "dr.": "dr\u00a7",
        "etc.": "etc\u00a7",
    }
    for old, new in protected.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)

    for old, new in protected.items():
        parts = [p.replace(new, old) for p in parts]

    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Step 1: Page-aware PDF loading
# ---------------------------------------------------------------------------

def load_document(filepath: str) -> List[Dict[str, Any]]:
    import fitz

    doc = fitz.open(filepath)
    pages = []
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "page": i + 1,
                    "text": normalize_text(text),
                })
    finally:
        doc.close()
    return pages


# ---------------------------------------------------------------------------
# Step 2: Structure-aware chunking (page-by-page)
# ---------------------------------------------------------------------------

def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 140:
        return False
    if re.match(r"^(chapter|section|part)\s+\w+", line, re.I):
        return True
    if re.match(r"^\d+(?:\.\d+)*[\s.)-]+[A-Z]", line):
        return True
    words = line.split()
    if 1 <= len(words) <= 10 and not line.endswith((".", "?", "!")):
        alpha = [w for w in words if any(c.isalpha() for c in w)]
        if alpha:
            upper_ratio = sum(1 for w in alpha if w.upper() == w) / len(alpha)
            return upper_ratio >= 0.65
    return False


def split_into_sections(page_text: str) -> List[Tuple[str, str]]:
    lines = [line.strip() for line in page_text.split("\n") if line.strip()]
    sections: List[Tuple[str, str]] = []
    current_heading = "Page content"
    current_lines: List[str] = []

    for line in lines:
        if is_heading(line):
            if current_lines:
                sections.append((current_heading, " ".join(current_lines)))
                current_lines = []
            current_heading = line
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, " ".join(current_lines)))

    return sections


def make_chunk(
    text: str,
    heading: str,
    page: int,
    chunk_id: int,
) -> Dict[str, Any]:
    return {
        "text": text.strip(),
        "page": page,
        "section": heading,
        "chunk_id": chunk_id,
    }


def chunk_page(
    page_text: str,
    page: int,
    start_chunk_id: int,
) -> List[Dict[str, Any]]:
    sections = split_into_sections(page_text)
    chunks: List[Dict[str, Any]] = []
    chunk_id = start_chunk_id

    for heading, body in sections:
        sentences = sentence_split(body)
        current: List[str] = []
        current_words = 0

        for sentence in sentences:
            words = sentence.split()
            n = len(words)

            if current and current_words + n > CHUNK_SIZE:
                chunks.append(make_chunk(" ".join(current), heading, page, chunk_id))
                chunk_id += 1

                overlap: List[str] = []
                overlap_words = 0
                for old_sentence in reversed(current):
                    old_n = len(old_sentence.split())
                    if overlap_words + old_n > OVERLAP_SIZE:
                        break
                    overlap.insert(0, old_sentence)
                    overlap_words += old_n

                current = overlap
                current_words = overlap_words

            current.append(sentence)
            current_words += n

        if current:
            chunks.append(make_chunk(" ".join(current), heading, page, chunk_id))
            chunk_id += 1

    return chunks


def chunk_document(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    next_id = 0
    for page in pages:
        page_chunks = chunk_page(page["text"], page["page"], next_id)
        chunks.extend(page_chunks)
        next_id += len(page_chunks)
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Jina embeddings
# ---------------------------------------------------------------------------

def _jina_headers() -> Dict[str, str]:
    api_key = os.environ.get("JINA_API_KEY", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def embed_texts(
    texts: List[str],
    task: str,
    batch_size: int = 50,
) -> Optional[List[List[float]]]:
    api_key = os.environ.get("JINA_API_KEY", "")
    if not api_key:
        logger.error("JINA_API_KEY is not configured.")
        return None

    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = requests.post(
                JINA_EMBED_URL,
                headers=_jina_headers(),
                json={
                    "model": JINA_EMBED_MODEL,
                    "input": [{"text": t} for t in batch],
                    "task": task,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            data.sort(key=lambda item: item.get("index", 0))

            if len(data) != len(batch):
                logger.error(
                    "Jina returned %d embeddings for %d inputs.",
                    len(data),
                    len(batch),
                )
                return None

            all_embeddings.extend(item["embedding"] for item in data)

        except Exception as exc:
            logger.exception("Jina embedding request failed: %s", exc)
            return None

    return all_embeddings


def embed_chunks(chunks: List[Dict[str, Any]]) -> Optional[List[List[float]]]:
    texts = [chunk["text"] for chunk in chunks]
    return embed_texts(texts, task="retrieval.passage")


def embed_query(text: str) -> Optional[List[float]]:
    result = embed_texts([text], task="retrieval.query")
    return result[0] if result else None


# ---------------------------------------------------------------------------
# Step 4: Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Step 5: BM25 keyword retrieval
# ---------------------------------------------------------------------------

class SimpleBM25:
    def __init__(
        self,
        documents: List[str],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(d) for d in documents]
        self.doc_count = len(self.documents)
        self.doc_lengths = [len(d) for d in self.documents]
        self.avgdl = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 1.0
        )
        self.df: Dict[str, int] = {}
        for doc in self.documents:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1

    def score(self, query: str) -> List[float]:
        query_tokens = tokenize(query)
        scores = [0.0] * self.doc_count
        if not query_tokens:
            return scores

        for i, doc in enumerate(self.documents):
            frequencies: Dict[str, int] = {}
            for token in doc:
                frequencies[token] = frequencies.get(token, 0) + 1
            dl = self.doc_lengths[i]
            for token in query_tokens:
                if token not in frequencies:
                    continue
                df = self.df.get(token, 0)
                idf = max(
                    0.0,
                    math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5)),
                )
                tf = frequencies[token]
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * dl / max(self.avgdl, 1e-9)
                )
                scores[i] += idf * (tf * (self.k1 + 1)) / max(denominator, 1e-9)

        return scores


# ---------------------------------------------------------------------------
# Step 6: Store vectors + BM25 index
# ---------------------------------------------------------------------------

def store_vectors(
    doc_id: str,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> None:
    bm25 = SimpleBM25([chunk["text"] for chunk in chunks])
    vector_store[doc_id] = {
        "chunks": chunks,
        "embeddings": embeddings,
        "bm25": bm25,
    }


# ---------------------------------------------------------------------------
# Step 7: Hybrid retrieval
# ---------------------------------------------------------------------------

def min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high - low < 1e-9:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(v - low) / (high - low) for v in values]


def hybrid_retrieve(
    query: str,
    doc_id: str,
    k: int = RETRIEVAL_K,
) -> List[Dict[str, Any]]:
    store = vector_store.get(doc_id)
    if not store:
        return []

    query_embedding = embed_query(query)
    if not query_embedding:
        return []

    embeddings = store["embeddings"]
    bm25 = store["bm25"]
    chunks = store["chunks"]

    semantic_scores = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    lexical_scores = bm25.score(query)

    semantic_norm = min_max_normalize(semantic_scores)
    lexical_norm = min_max_normalize(lexical_scores)

    results: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        hybrid_score = (
            SEMANTIC_WEIGHT * semantic_norm[i] + BM25_WEIGHT * lexical_norm[i]
        )
        results.append({
            "index": i,
            "chunk": chunk,
            "semantic_score": semantic_scores[i],
            "bm25_score": lexical_scores[i],
            "hybrid_score": hybrid_score,
        })

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:k]


# ---------------------------------------------------------------------------
# Step 8: Jina hosted reranker API
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    final_k: int = FINAL_K,
) -> List[Dict[str, Any]]:
    """
    Rerank candidates using the Jina Reranker API.

    Falls back to hybrid score ordering if the API call fails.
    """
    if not candidates:
        return []

    api_key = os.environ.get("JINA_API_KEY", "")
    if not api_key:
        logger.warning("JINA_API_KEY not set; skipping reranker.")
        return candidates[:final_k]

    documents = [item["chunk"]["text"] for item in candidates]

    try:
        response = requests.post(
            JINA_RERANK_URL,
            headers=_jina_headers(),
            json={
                "model": JINA_RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": final_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])

        ranked: List[Dict[str, Any]] = []
        for result in results:
            idx = result.get("index", 0)
            reranker_score = result.get("relevance_score", 0.0)
            item = candidates[idx]
            item["reranker_score"] = reranker_score
            ranked.append(item)

        if not ranked:
            logger.warning("Jina reranker returned empty results; using hybrid fallback.")
            return candidates[:final_k]

        return ranked[:final_k]

    except Exception as exc:
        logger.warning("Jina reranker failed; using hybrid fallback: %s", exc)
        return candidates[:final_k]


# ---------------------------------------------------------------------------
# Step 9: Context construction
# ---------------------------------------------------------------------------

def build_context(retrieved: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for i, item in enumerate(retrieved, 1):
        chunk = item["chunk"]
        parts.append(
            f"[Source {i}]\n"
            f"Page: {chunk['page']}\n"
            f"Section: {chunk['section']}\n"
            f"Chunk ID: {chunk['chunk_id']}\n"
            f"Text:\n{chunk['text']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step 10: LLM generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a high-accuracy document question-answering assistant.

Rules:
1. Answer ONLY from the supplied document context.
2. Do not use outside knowledge.
3. If the context does not contain enough evidence, say:
   "The information was not found in the document."
4. For names, dates, IDs, phone numbers, emails, prices, percentages,
   quantities and other exact values, copy the value exactly from the context.
5. Do not invent or estimate missing values.
6. Prefer the most directly relevant source.
7. When useful, cite the page number in the answer.
8. If multiple sources disagree, explicitly say that the document contains
   conflicting information and identify the relevant pages.
9. Keep the answer concise unless the question requires explanation."""


def generate_answer(question: str, context: str) -> Optional[str]:
    user_prompt = (
        f"Document context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer using only the document context."
    )

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            client = OpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
            )
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content
            if answer:
                return answer.strip()
        except Exception as exc:
            logger.warning("Groq generation failed: %s", exc)

    google_key = os.environ.get("GOOGLE_API_KEY", "")
    if google_key:
        try:
            client = OpenAI(
                api_key=google_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            response = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content
            if answer:
                return answer.strip()
        except Exception as exc:
            logger.warning("Gemini generation failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Step 11: Confidence / evidence score
# ---------------------------------------------------------------------------

def calculate_confidence(retrieved: List[Dict[str, Any]]) -> float:
    """
    Heuristic confidence indicator, NOT a probability.

    Uses reranker score when available, combined with hybrid/semantic evidence.
    """
    if not retrieved:
        return 0.0

    top = retrieved[0]
    semantic = max(0.0, min(1.0, top.get("semantic_score", 0.0)))
    hybrid = max(0.0, min(1.0, top.get("hybrid_score", 0.0)))
    reranker_score = top.get("reranker_score")

    if reranker_score is not None:
        reranker_norm = 1.0 / (1.0 + math.exp(-reranker_score))
        confidence = 0.45 * reranker_norm + 0.30 * hybrid + 0.25 * semantic
    else:
        confidence = 0.65 * hybrid + 0.35 * semantic

    return round(max(0.0, min(0.99, confidence)) * 100, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_pdf(filepath: str, filename: str) -> Dict[str, Any]:
    import fitz

    try:
        doc = fitz.open(filepath)
        page_count = doc.page_count
        doc.close()
    except Exception as exc:
        return {"error": f"Could not open PDF: {exc}"}

    if page_count > MAX_PAGES:
        return {
            "error": (
                f"This application supports PDF files with up to "
                f"{MAX_PAGES} pages only."
            )
        }

    pages = load_document(filepath)
    if not pages:
        return {
            "error": "No text found in the PDF. The document may be image-based."
        }

    chunks = chunk_document(pages)
    if not chunks:
        return {"error": "Could not create chunks from the document."}

    embeddings = embed_chunks(chunks)
    if not embeddings:
        return {"error": "Failed to embed document. Please try again."}

    doc_id = str(uuid.uuid4())[:8]
    store_vectors(doc_id, chunks, embeddings)

    full_text = "\n\n".join(page["text"] for page in pages)
    words = re.findall(r"\b\w+\b", full_text)
    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()
    ]

    return {
        "doc_id": doc_id,
        "filename": filename,
        "pages": page_count,
        "chunks": len(chunks),
        "summary": {
            "total_pages": page_count,
            "total_words": len(words),
            "total_paragraphs": len(paragraphs),
            "total_characters": len(full_text),
            "overview": paragraphs[0][:500] if paragraphs else "",
        },
        "search_mode": "hybrid+reranker",
    }


def search(query: str, doc_id: str) -> Dict[str, Any]:
    if doc_id not in vector_store:
        return {"error": "No PDF uploaded. Please upload a PDF first."}

    if not query or not query.strip():
        return {"error": "Please provide a question."}

    candidates = hybrid_retrieve(query=query, doc_id=doc_id, k=RETRIEVAL_K)

    if not candidates:
        return {
            "answer": "No relevant information found in the document.",
            "sources": [],
        }

    retrieved = rerank(query=query, candidates=candidates, final_k=FINAL_K)
    context = build_context(retrieved)
    answer = generate_answer(question=query, context=context)
    confidence = calculate_confidence(retrieved)

    sources: List[Dict[str, Any]] = []
    for item in retrieved:
        chunk = item["chunk"]
        sources.append({
            "page": chunk["page"],
            "section": chunk["section"],
            "chunk_id": chunk["chunk_id"],
            "semantic_score": round(item.get("semantic_score", 0.0) * 100, 1),
            "bm25_score": round(item.get("bm25_score", 0.0), 3),
            "hybrid_score": round(item.get("hybrid_score", 0.0) * 100, 1),
            "reranker_score": (
                round(item["reranker_score"], 3)
                if "reranker_score" in item
                else None
            ),
        })

    if answer:
        return {
            "answer": answer,
            "best_page": retrieved[0]["chunk"]["page"],
            "confidence": confidence,
            "method": "hybrid+reranker",
            "sources": sources,
        }

    return {
        "answer": retrieved[0]["chunk"]["text"],
        "best_page": retrieved[0]["chunk"]["page"],
        "confidence": confidence,
        "method": "retrieval-fallback",
        "sources": sources,
    }
