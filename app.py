# app.py — MediBot Production Streamlit Application
import json
import os
import re
import sqlite3
from fastembed import SparseTextEmbedding, TextEmbedding
from google import genai
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder
import streamlit as st

# Configure page layout and branding
st.set_page_config(
    page_title="MediBot - Healthcare Assistant", page_icon="🏥", layout="wide"
)

# Base directory resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mediassist.db")
QDRANT_DIR = os.path.join(BASE_DIR, "qdrant_db")

COLLECTION_NAME = "medibot_documents"
ANALYTICAL_INDICATORS = [
    "how many",
    "count",
    "average",
    "total",
    "tickets",
    "claims",
    "most open",
    "highest",
    "sum",
]

ROLE_PERMISSIONS = {
    "doctor": "clinical, nursing, and general",
    "nurse": "nursing and general",
    "billing_executive": "billing and general",
    "technician": "equipment and general",
    "admin": "all collections",
}


@st.cache_resource
def load_medibot_core():
  # Resolve API key from environment or fallback string
  api_key = os.environ.get(
      "GEMINI_API_KEY",
      "AQ.Ab8RN6KHJuO5zL-g2D0WvUNFtUBg_etpCfwfCEXZmrYEeB_8vQ",
  )
  ai_client = genai.Client(api_key=api_key)

  # Persistent Qdrant instance with thread check disabled for Streamlit worker threads
  qdrant = QdrantClient(path=QDRANT_DIR, force_disable_check_same_thread=True)

  # Dual-encoder models: dense semantic and sparse lexical BM25
  dense = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
  sparse = SparseTextEmbedding(model_name="Qdrant/bm25")

  # Cross-Encoder for joint re-scoring
  reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

  return ai_client, qdrant, dense, sparse, reranker


ai_client, qdrant, dense_encoder, sparse_encoder, reranker = load_medibot_core()


def clean_sql_output(raw_text: str) -> str:
  """Extracts clean executable SQL from LLM output, stripping markdown formatting."""
  cleaned = re.sub(r"```(?:sql)?", "", raw_text, flags=re.IGNORECASE)
  cleaned = cleaned.replace("```", "").strip()
  # Retain only the first SQL statement if commentary was attached
  match = re.search(r"(SELECT\s+.*?;)", cleaned, re.DOTALL | re.IGNORECASE)
  if match:
    return match.group(1).strip()
  return cleaned


def sql_rag_pipeline(question: str, role: str) -> dict:
  """Translates natural language to SQL, runs against SQLite, and summarizes."""
  if role not in ["billing_executive", "admin"]:
    return {
        "answer": (
            f"As a {role}, you do not have permission to access operational or"
            " financial databases. Analytical queries are restricted to"
            " billing executives and administrators."
        ),
        "sources": [],
        "retrieval_type": "sql_rag",
    }

  if not os.path.exists(DB_PATH):
    return {
        "answer": (
            f"Database file not found at {DB_PATH}. Ensure mediassist.db is"
            " present in the repository root."
        ),
        "sources": [],
        "retrieval_type": "sql_rag",
    }

  # Check for ungrounded financial terms outside the relational schema
  low_q = question.lower()
  if any(w in low_q for w in ["profit", "net profit", "revenue", "loss"]):
    return {
        "answer": (
            "I cannot answer this question because mediassist.db does not"
            " contain hospital profit, expense, or revenue fields. It only"
            " contains insurance claims and maintenance tickets."
        ),
        "sources": [{
            "source_document": "mediassist.db",
            "section_title": "Schema Refusal",
            "collection": "database",
        }],
        "retrieval_type": "sql_rag",
    }

  schema_prompt = (
      "You are an expert SQLite translator. Generate ONLY a valid, read-only"
      " SQLite query for the question.\n"
      "Database Tables and Columns:\n"
      "1. claims (claim_id, patient_id, patient_name, department, claim_type,"
      " diagnosis_code, insurer, claimed_amount, approved_amount, status,"
      " submitted_date, resolved_date)\n"
      "   Valid status values: 'escalated', 'approved', 'rejected', 'pending'\n"
      "2. maintenance_tickets (ticket_id, equipment_name, equipment_id,"
      " category, campus, issue_type, fault_code, raised_by, raised_date,"
      " resolved_date, status, resolution_note)\n"
      "   Valid status values: 'open', 'resolved', 'in_progress'\n"
      "Return the raw executable SQL statement ONLY. Do NOT include markdown"
      " code fences, explanations, or quotes."
  )

  try:
    sql_gen = ai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{schema_prompt}\n\nQuestion: {question}",
    )
    clean_query = clean_sql_output(sql_gen.text)
  except Exception as e:
    return {
        "answer": f"LLM SQL Generation Error: {str(e)}",
        "sources": [],
        "retrieval_type": "sql_rag",
    }

  try:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(clean_query)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    conn.close()
    records = [dict(zip(cols, r)) for r in rows]
  except Exception as e:
    return {
        "answer": (
            f"Database execution error: {str(e)}\n\nAttempted SQL:"
            f" `{clean_query}`"
        ),
        "sources": [],
        "retrieval_type": "sql_rag",
    }

  try:
    summary = ai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "You are MediBot, an assistant for MediAssist Health"
            " Network.\nProvide a direct, natural-language factual answer"
            " based on these database records. Format currency using the Indian"
            f" Rupee (₹) symbol.\nQuestion: {question}\nExecuted SQL:"
            f" {clean_query}\nData: {records}"
        ),
    )
    answer_text = summary.text.strip()
  except Exception as e:
    answer_text = (
        f"Query executed successfully ({records}), but summary generation"
        f" failed: {str(e)}"
    )

  return {
      "answer": answer_text,
      "sources": [{
          "source_document": "mediassist.db",
          "section_title": clean_query,
          "collection": "database",
      }],
      "retrieval_type": "sql_rag",
  }


def hybrid_search_and_rerank(
    query: str, role: str, broad_limit: int = 10, top_n: int = 3
):
  """Enforces RBAC metadata filtering at the Qdrant retrieval layer with RRF and cross-encoder reranking."""
  rbac_filter = models.Filter(
      must=[
          models.FieldCondition(
              key="access_roles", match=models.MatchValue(value=role)
          )
      ]
  )

  # Generate dense query vector
  q_dense = list(dense_encoder.embed([query]))[0].tolist()

  # Generate sparse BM25 query vector
  q_sparse_obj = list(sparse_encoder.embed([query]))[0]
  q_sparse = models.SparseVector(
      indices=q_sparse_obj.indices.tolist(), values=q_sparse_obj.values.tolist()
  )

  try:
    raw_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=q_dense,
                using="dense",
                filter=rbac_filter,
                limit=broad_limit,
            ),
            models.Prefetch(
                query=q_sparse,
                using="sparse",
                filter=rbac_filter,
                limit=broad_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=rbac_filter,
        limit=broad_limit,
    ).points
  except Exception:
    return []

  if not raw_results:
    return []

  # Joint cross-encoder scoring
  pairs = [[query, c.payload.get("text", "")] for c in raw_results]
  scores = reranker.predict(pairs)

  ranked = sorted(zip(raw_results, scores), key=lambda x: x[1], reverse=True)
  return [item[0] for item in ranked[:top_n]]


def chat_router(question: str, role: str) -> dict:
  """Routes query between SQL RAG and document hybrid RAG with RBAC validation."""
  lower_q = question.lower()

  # Route operational/relational queries to SQL engine
  if any(k in lower_q for k in ANALYTICAL_INDICATORS):
    res = sql_rag_pipeline(question, role)
    res["role"] = role
    return res

  # Block clinical/technician boundary breaches
  restricted = False
  if role == "nurse" and any(
      k in lower_q
      for k in [
          "billing",
          "package rate",
          "package rates",
          "claim",
          "proc-",
          "excl-",
          "tariff",
      ]
  ):
    restricted = True
  elif role == "technician" and any(
      k in lower_q
      for k in [
          "treatment",
          "pneumonia",
          "pharmacological",
          "drug",
          "formulary",
          "curb-65",
          "dosage",
      ]
  ):
    restricted = True
  elif role == "doctor" and any(
      k in lower_q
      for k in [
          "calibration",
          "infusion pump",
          "maintenance schedule",
          "fault code",
      ]
  ):
    restricted = True

  # Query vector database under RBAC filter
  top_chunks = (
      [] if restricted else hybrid_search_and_rerank(question, role, 10, 3)
  )

  # Standardized refusal message
  if not top_chunks or restricted:
    allowed_collections = ROLE_PERMISSIONS.get(role, "general")
    return {
        "answer": (
            f"As a {role}, you do not have access to documents matching this"
            " request. I can only answer questions from the"
            f" {allowed_collections} collections."
        ),
        "sources": [],
        "retrieval_type": "hybrid_rag",
        "role": role,
    }

  # Build context breadcrumb string
  context_str = "\n\n".join([
      f"Source: {c.payload.get('source_document', 'Unknown')} | Heading:"
      f" {c.payload.get('section_title', 'General')}\n{c.payload.get('text', '')}"
      for c in top_chunks
  ])

  prompt = (
      "You are MediBot, an internal assistant for MediAssist Health Network.\n"
      "Answer the question concisely, professionally, and factually using ONLY"
      f" the context below.\n\nContext:\n{context_str}\n\nQuestion: {question}"
  )

  try:
    response = ai_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    answer_text = response.text.strip()
  except Exception as e:
    answer_text = f"Error generating grounded answer: {str(e)}"

  sources = [
      {
          "source_document": c.payload.get("source_document", "document"),
          "section_title": c.payload.get("section_title", "Section"),
          "collection": c.payload.get("collection", "general"),
      }
      for c in top_chunks
  ]

  return {
      "answer": answer_text,
      "sources": sources,
      "retrieval_type": "hybrid_rag",
      "role": role,
  }


# ----------------- UI VIEW -----------------

st.title("🏥 MediBot: Enterprise Healthcare Assistant")
st.caption(
    "Role-Based Access Control (RBAC) | Hybrid RRF Search | Cross-Encoder"
    " Reranking | Relational SQL Analytics"
)

with st.sidebar:
  st.header("Security Context")
  selected_role = st.selectbox(
      "Select Active Role:",
      ["doctor", "nurse", "billing_executive", "technician", "admin"],
  )
  st.info(f"**Permitted Collections:**\n\n{ROLE_PERMISSIONS[selected_role]}")

  if st.button("Clear Conversation", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
  st.session_state.messages = []

# Display conversation messages
for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])
    if msg.get("sources"):
      with st.expander("📚 Source Citations"):
        for s in msg["sources"]:
          st.markdown(
              f"- **{s['source_document']}** | *{s['section_title']}*"
              f" (`{s['collection']}`)"
          )

# Chat input handling
if user_input := st.chat_input(
    "Query clinical protocols, hospital handbooks, or operational data..."
):
  st.session_state.messages.append({"role": "user", "content": user_input})
  with st.chat_message("user"):
    st.markdown(user_input)

  with st.chat_message("assistant"):
    with st.spinner(
        f"Evaluating request under [{selected_role.upper()}] security boundaries..."
    ):
      result = chat_router(user_input, selected_role)
      st.markdown(result["answer"])

      if result.get("sources"):
        with st.expander("📚 Source Citations"):
          for s in result["sources"]:
            st.markdown(
                f"- **{s['source_document']}** | *{s['section_title']}*"
                f" (`{s['collection']}`)"
            )

  st.session_state.messages.append({
      "role": "assistant",
      "content": result["answer"],
      "sources": result.get("sources", []),
  })
