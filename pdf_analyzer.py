"""
PDF Analyzer — Improved Hybrid RAG Pipeline

Pipeline
--------
1. PDF extraction (page-aware)
2. Text preprocessing
3. Structure-aware / sentence-aware chunking
4. Jina embeddings
5. In-memory vector store
6. BM25 keyword retrieval
7. Hybrid retrieval
8. Cross-encoder reranking (optional, with safe fallback)
9. Context building with metadata/citations
10. LLM generation with grounded-answer instructions

Environment
-----------
Required:
    JINA_API_KEY

One LLM key:
    GROQ_API_KEY
    or GOOGLE_API_KEY

Optional:
    RERANKER_MODEL
        Default: cross-encoder/ms-marco-MiniLM-L-6-v2

Install:
    pip install pymupdf requests openai rank-bm25 sentence-transformers
"""

import logging
import os
import re
import uuid
from typing import Dict, List, Tuple, Optional, Any

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE = 350              # approximate words
OVERLAP_SIZE = 60             # approximate words
RETRIEVAL_K = 20              # candidates before reranking
FINAL_K = 5                   # context chunks sent to the LLM
MAX_PAGES = 20

JINA_MODEL = "jina-embeddings-v3"
JINA_API_URL = "https://api.jina.ai/v1/embeddings"

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.0-flash"

# Optional reranker. If unavailable, hybrid scores are used.
RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

vector_store: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize whitespace without destroying paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Simple tokenization suitable for BM25."""
    return re.findall(r"\b\w+(?:[-']\w+)*\b", text.lower())


def sentence_split(text: str) -> List[str]:
    """Reasonably robust sentence splitter without external NLP dependency."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Protect common abbreviations from naïve splitting.
    protected = {
        "e.g.": "e§g§",
        "i.e.": "i§e§",
        "mr.": "mr§",
        "mrs.": "mrs§",
        "dr.": "dr§",
        "etc.": "etc§",
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
    """
    Load each page independently.

    Improvement:
    - We preserve exact page ownership.
    - We do NOT guess a chunk's page by word overlap.
    """
    import fitz

    doc = fitz.open(filepath)
    pages = []

    try:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    {
                        "page": i + 1,
                        "text": normalize_text(text),
                    }
                )
    finally:
        doc.close()

    return pages


# ---------------------------------------------------------------------------
# Step 2: Structure-aware chunking
# ---------------------------------------------------------------------------

def is_heading(line: str) -> bool:
    """Heuristic heading detector for ordinary text PDFs."""
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
            upper_ratio = sum(
                1 for w in alpha if w.upper() == w
            ) / len(alpha)
            return upper_ratio >= 0.65

    return False


def split_into_sections(page_text: str) -> List[Tuple[str, str]]:
    """
    Return (heading, body) sections.

    If no obvious heading exists, the heading is 'Page content'.
    """
    lines = [line.strip() for line in page_text.split("\n") if line.strip()]

    sections = []
    current_heading = "Page content"
    current_lines = []

    for line in lines:
        if is_heading(line):
            if current_lines:
                sections.append(
                    (current_heading, " ".join(current_lines))
                )
                current_lines = []
            current_heading = line
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, " ".join(current_lines)))

    return sections


def make_chunk(text: str, heading: str, page: int, chunk_id: int) -> Dict[str, Any]:
    return {
        "text": text.strip(),
        "page": page,
        "section": heading,
        "chunk_id": chunk_id,
    }


def chunk_page(page_text: str, page: int, start_chunk_id: int) -> List[Dict[str, Any]]:
    """
    Create chunks while trying to preserve section and sentence boundaries.
    """
    sections = split_into_sections(page_text)
    chunks = []
    chunk_id = start_chunk_id

    for heading, body in sections:
        sentences = sentence_split(body)

        current: List[str] = []
        current_words = 0

        for sentence in sentences:
            words = sentence.split()
            n = len(words)

            if current and current_words + n > CHUNK_SIZE:
                chunks.append(
                    make_chunk(
                        " ".join(current),
                        heading,
                        page,
                        chunk_id,
                    )
                )
                chunk_id += 1

                # Preserve a short tail as overlap.
                overlap = []
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
            chunks.append(
                make_chunk(
                    " ".join(current),
                    heading,
                    page,
                    chunk_id,
                )
            )
            chunk_id += 1

    return chunks


def chunk_document(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    next_id = 0

    for page in pages:
        page_chunks = chunk_page(
            page["text"],
            page["page"],
            next_id,
        )
        chunks.extend(page_chunks)
        next_id += len(page_chunks)

    return chunks


# ---------------------------------------------------------------------------
# Step 3: Jina embeddings
# ---------------------------------------------------------------------------

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
        batch = texts[i:i + batch_size]

        try:
            response = requests.post(
                JINA_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": JINA_MODEL,
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

            all_embeddings.extend(
                item["embedding"] for item in data
            )

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
    """
    Lightweight BM25 implementation.

    This avoids requiring a database and is enough for the small
    <=20-page PDF use case.
    """

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
            sum(self.doc_lengths) / self.doc_count
            if self.doc_count
            else 1.0
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

                # Standard BM25 IDF with a safe floor.
                idf = max(
                    0.0,
                    __import__("math").log(
                        1.0 +
                        (self.doc_count - df + 0.5) /
                        (df + 0.5)
                    )
                )

                tf = frequencies[token]

                denominator = (
                    tf +
                    self.k1 *
                    (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                )

                scores[i] += (
                    idf *
                    (tf * (self.k1 + 1)) /
                    max(denominator, 1e-9)
                )

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

    return [
        (v - low) / (high - low)
        for v in values
    ]


def hybrid_retrieve(
    query: str,
    doc_id: str,
    k: int = RETRIEVAL_K,
) -> List[Dict[str, Any]]:
    """
    Retrieve using both semantic and lexical signals.

    Hybrid score:
        70% vector similarity
        30% BM25

    These weights should ultimately be tuned on an evaluation set.
    """
    store = vector_store.get(doc_id)
    if not store:
        return []

    query_embedding = embed_query(query)
    if not query_embedding:
        return []

    embeddings = store["embeddings"]
    bm25 = store["bm25"]
    chunks = store["chunks"]

    semantic_scores = [
        cosine_similarity(query_embedding, emb)
        for emb in embeddings
    ]

    lexical_scores = bm25.score(query)

    semantic_norm = min_max_normalize(semantic_scores)
    lexical_norm = min_max_normalize(lexical_scores)

    results = []

    for i, chunk in enumerate(chunks):
        hybrid_score = (
            0.70 * semantic_norm[i] +
            0.30 * lexical_norm[i]
        )

        results.append(
            {
                "index": i,
                "chunk": chunk,
                "semantic_score": semantic_scores[i],
                "bm25_score": lexical_scores[i],
                "hybrid_score": hybrid_score,
            }
        )

    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )

    return results[:k]


# ---------------------------------------------------------------------------
# Step 8: Optional cross-encoder reranking
# ---------------------------------------------------------------------------

_reranker = None


def get_reranker():
    global _reranker

    if _reranker is not None:
        return _reranker

    try:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker: %s", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
        return _reranker

    except Exception as exc:
        logger.warning(
            "Reranker unavailable; using hybrid retrieval only: %s",
            exc,
        )
        _reranker = False
        return None


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    final_k: int = FINAL_K,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    reranker = get_reranker()

    if reranker is None:
        return candidates[:final_k]

    pairs = [
        [query, item["chunk"]["text"]]
        for item in candidates
    ]

    try:
        raw_scores = reranker.predict(pairs)

        for item, score in zip(candidates, raw_scores):
            item["reranker_score"] = float(score)

        candidates.sort(
            key=lambda x: x["reranker_score"],
            reverse=True,
        )

    except Exception as exc:
        logger.warning(
            "Reranker failed; falling back to hybrid score: %s",
            exc,
        )

    return candidates[:final_k]


# ---------------------------------------------------------------------------
# Step 9: Context construction
# ---------------------------------------------------------------------------

def build_context(
    retrieved: List[Dict[str, Any]]
) -> str:
    parts = []

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

SYSTEM_PROMPT = """
You are a high-accuracy document question-answering assistant.

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
9. Keep the answer concise unless the question requires explanation.
"""


def generate_answer(
    question: str,
    context: str,
) -> Optional[str]:

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
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
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
                base_url=(
                    "https://generativelanguage.googleapis.com/"
                    "v1beta/openai/"
                ),
            )

            response = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
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

def calculate_confidence(
    retrieved: List[Dict[str, Any]]
) -> float:
    """
    This is a heuristic confidence indicator, NOT a probability.

    It uses reranker score when available and combines it with
    semantic/hybrid evidence.
    """
    if not retrieved:
        return 0.0

    top = retrieved[0]

    semantic = max(
        0.0,
        min(1.0, top.get("semantic_score", 0.0))
    )

    hybrid = max(
        0.0,
        min(1.0, top.get("hybrid_score", 0.0))
    )

    reranker_score = top.get("reranker_score")

    if reranker_score is not None:
        # Convert an unbounded cross-encoder score to 0..1.
        import math
        reranker_norm = 1.0 / (1.0 + math.exp(-reranker_score))

        confidence = (
            0.45 * reranker_norm +
            0.30 * hybrid +
            0.25 * semantic
        )
    else:
        confidence = (
            0.65 * hybrid +
            0.35 * semantic
        )

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
            "error": (
                "No text found in the PDF. "
                "The document may be image-based."
            )
        }

    chunks = chunk_document(pages)

    if not chunks:
        return {
            "error": "Could not create chunks from the document."
        }

    embeddings = embed_chunks(chunks)

    if not embeddings:
        return {
            "error": "Failed to embed document. Please try again."
        }

    doc_id = str(uuid.uuid4())[:8]

    store_vectors(
        doc_id,
        chunks,
        embeddings,
    )

    full_text = "\n\n".join(page["text"] for page in pages)

    words = re.findall(r"\b\w+\b", full_text)
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", full_text)
        if p.strip()
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
            "overview": (
                paragraphs[0][:500]
                if paragraphs
                else ""
            ),
        },
        "search_mode": "hybrid",
    }


def search(query: str, doc_id: str) -> Dict[str, Any]:
    if doc_id not in vector_store:
        return {
            "error": "No PDF uploaded. Please upload a PDF first."
        }

    if not query or not query.strip():
        return {
            "error": "Please provide a question."
        }

    candidates = hybrid_retrieve(
        query=query,
        doc_id=doc_id,
        k=RETRIEVAL_K,
    )

    if not candidates:
        return {
            "answer": "No relevant information found in the document.",
            "sources": [],
        }

    retrieved = rerank(
        query=query,
        candidates=candidates,
        final_k=FINAL_K,
    )

    context = build_context(retrieved)
    answer = generate_answer(
        question=query,
        context=context,
    )

    confidence = calculate_confidence(retrieved)

    sources = []

    for item in retrieved:
        chunk = item["chunk"]

        sources.append(
            {
                "page": chunk["page"],
                "section": chunk["section"],
                "chunk_id": chunk["chunk_id"],
                "semantic_score": round(
                    item.get("semantic_score", 0.0) * 100,
                    1,
                ),
                "bm25_score": round(
                    item.get("bm25_score", 0.0),
                    3,
                ),
                "hybrid_score": round(
                    item.get("hybrid_score", 0.0) * 100,
                    1,
                ),
                "reranker_score": (
                    round(item["reranker_score"], 3)
                    if "reranker_score" in item
                    else None
                ),
            }
        )

    if answer:
        return {
            "answer": answer,
            "best_page": retrieved[0]["chunk"]["page"],
            "confidence": confidence,
            "method": "hybrid+rereanker",
            "sources": sources,
        }

    # Safe fallback: return evidence rather than hallucinating.
    return {
        "answer": retrieved[0]["chunk"]["text"],
        "best_page": retrieved[0]["chunk"]["page"],
        "confidence": confidence,
        "method": "retrieval-fallback",
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Optional evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_recall(
    doc_id: str,
    questions: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, float]:
    """
    Basic retrieval evaluation.

    Each question should contain:
        {
            "question": "...",
            "expected_pages": [2, 5]
        }

    Recall@K here means whether at least one expected page
    appears among the retrieved top-K chunks.
    """
    if not questions:
        return {
            "recall_at_k": 0.0,
            "questions": 0,
        }

    hits = 0

    for item in questions:
        question = item["question"]
        expected_pages = set(item.get("expected_pages", []))

        candidates = hybrid_retrieve(
            query=question,
            doc_id=doc_id,
            k=max(k, RETRIEVAL_K),
        )

        ranked = rerank(
            query=question,
            candidates=candidates,
            final_k=k,
        )

        returned_pages = {
            result["chunk"]["page"]
            for result in ranked
        }

        if returned_pages & expected_pages:
            hits += 1

    return {
        "recall_at_k": round(hits / len(questions), 4),
        "questions": len(questions),
    }


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------
#
# result = analyze_pdf("example.pdf", "example.pdf")
# doc_id = result["doc_id"]
#
# answer = search(
#     "What is the refund period?",
#     doc_id,
# )
#
# print(answer)
#