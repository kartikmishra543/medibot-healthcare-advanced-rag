# Cell 14: Complete Streamlit Application with Thread-Safe Vector Access
import streamlit as st
import json
import sqlite3
import re
import os
from google import genai
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from sentence_transformers import CrossEncoder

# Set page title, browser tab icon, and default wide container layout
st.set_page_config(
    page_title="MediBot - Internal Healthcare Assistant",
    page_icon="🏥",
    layout="wide"
)

# Cache heavyweight components (models and DB connections) so they don't reload on each user query
@st.cache_resource
def load_medibot_core():
    # Authenticate official Google GenAI SDK client
    api_key = "AQ.Ab8RN6KHJuO5zL-g2D0WvUNFtUBg_etpCfwfCEXZmrYEeB_8vQ"
    ai_client = genai.Client(api_key=api_key)

    # force_disable_check_same_thread=True prevents the portalocker / threading crash
    # when Streamlit's script-runner thread interacts with the SQLite/Qdrant backend
    qdrant = QdrantClient(path="./qdrant_db", force_disable_check_same_thread=True)

    # Initialize FastEmbed encoders: Dense semantic MiniLM and Sparse BM25
    dense = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    sparse = SparseTextEmbedding(model_name="Qdrant/bm25")

    # Load Cross-Encoder for deep contextual reranking
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return ai_client, qdrant, dense, sparse, reranker

# Unpack cached components into session scope
ai_client, qdrant, dense_encoder, sparse_encoder, reranker = load_medibot_core()

# Constants
COLLECTION_NAME = "medibot_documents"
DB_FILE = "mediassist.db"

# Indicators for routing queries to relational SQL RAG instead of vector search
ANALYTICAL_INDICATORS = ["how many", "count", "average", "total", "tickets", "claims", "most open", "highest"]

# Human-readable scopes used for UI sidebar instructions and access refusal messages
ROLE_PERMISSIONS = {
    "doctor": "clinical, nursing, and general",
    "nurse": "nursing and general",
    "billing_executive": "billing and general",
    "technician": "equipment and general",
    "admin": "all collections"
}

def clean_sql_output(raw_text: str) -> str:
    """Removes markdown code fences (```sql ... ```) to extract pure SQL text."""
    cleaned = re.sub(r"```(?:sql)?", "", raw_text, flags=re.IGNORECASE)
    return cleaned.replace("```", "").strip()

def sql_rag_pipeline(question: str, role: str) -> dict:
    """Component 4: Translates questions into SQLite, executes against mediassist.db, and summarizes."""
    # RBAC Boundary: Only billing_executive and admin can query the financial/operational database
    if role not in ["billing_executive", "admin"]:
        return {
            "answer": f"As a {role}, you do not have permission to access operational or financial databases. Analytical queries are restricted to billing executives and administrators.",
            "sources": [],
            "retrieval_type": "sql_rag"
        }

    schema_prompt = (
        "You are an expert SQLite translator. Generate ONLY a valid SQLite query for the question.\n"
        "Tables:\n"
        "1. claims (claim_id, patient_id, patient_name, department, claim_type, diagnosis_code, insurer, claimed_amount, approved_amount, status, submitted_date, resolved_date)\n"
        "2. maintenance_tickets (ticket_id, equipment_name, equipment_id, category, campus, issue_type, fault_code, raised_by, raised_date, resolved_date, status, resolution_note)\n"
        "Return the raw query ONLY without markdown blocks, quotes, or explanations."
    )

    # Step 1: Text-to-SQL translation via LLM
    sql_gen = ai_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"{schema_prompt}\n\nQuestion: {question}"
    )
    clean_query = clean_sql_output(sql_gen.text)

    # Step 2: Database execution
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute(clean_query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        conn.close()
        records = [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        return {"answer": f"Database execution error: {str(e)}", "sources": [], "retrieval_type": "sql_rag"}

    # Step 3: Natural language response synthesis
    summary = ai_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"You are MediBot. Provide a direct, factual answer based on these database records:\nQuestion: {question}\nExecuted SQL: {clean_query}\nData: {records}"
    )

    return {
        "answer": summary.text.strip(),
        "sources": [{"source_document": "mediassist.db", "section_title": clean_query, "collection": "database"}],
        "retrieval_type": "sql_rag"
    }

def hybrid_search_and_rerank(query: str, role: str, broad_limit: int = 10, top_n: int = 3):
    """Component 2 & 3: Retrieval-layer RBAC filter, hybrid fusion, and cross-encoder reranking."""
    # Retrieval-Layer RBAC: Non-authorized documents are dropped inside the vector engine
    rbac_filter = models.Filter(
        must=[models.FieldCondition(key="access_roles", match=models.MatchValue(value=role))]
    )

    # Compute dense embedding
    q_dense = list(dense_encoder.embed([query]))[0].tolist()

    # Compute sparse BM25 token frequencies
    q_sparse_obj = list(sparse_encoder.embed([query]))[0]
    q_sparse = models.SparseVector(
        indices=q_sparse_obj.indices.tolist(),
        values=q_sparse_obj.values.tolist()
    )

    # Query Qdrant with Reciprocal Rank Fusion (RRF)
    raw_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=q_dense, using="dense", filter=rbac_filter, limit=broad_limit),
            models.Prefetch(query=q_sparse, using="sparse", filter=rbac_filter, limit=broad_limit)
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=broad_limit
    ).points

    if not raw_results:
        return []

    # Cross-encoder joint scoring over (query, text) pairs
    pairs = [[query, c.payload["text"]] for c in raw_results]
    scores = reranker.predict(pairs)

    # Sort candidates by reranker relevance score descending
    ranked = sorted(zip(raw_results, scores), key=lambda x: x[1], reverse=True)
    return [item[0] for item in ranked[:top_n]]

def chat_router(question: str, role: str) -> dict:
    """Core intent router that directs between SQL analytics and hybrid document RAG."""
    # Route analytical queries to the SQL engine
    if any(k in question.lower() for k in ANALYTICAL_INDICATORS):
        res = sql_rag_pipeline(question, role)
        res["role"] = role
        return res

    # Retrieve authorized document chunks
    top_chunks = hybrid_search_and_rerank(question, role, broad_limit=10, top_n=3)

    # Secondary defense against cross-domain adversarial attempts
    lower_q = question.lower()
    restricted = False
    if role == "nurse" and any(k in lower_q for k in ["billing", "package rate", "rate", "claim", "proc-", "excl-"]):
        restricted = True
    elif role == "technician" and any(k in lower_q for k in ["treatment", "diabetes", "pneumonia", "drug", "curb-65"]):
        restricted = True
    elif role in ["doctor", "nurse", "technician"] and any(k in lower_q for k in ["claims", "reimbursement", "pre-auth"]):
        restricted = True

    # Standardized refusal output when access is barred
    if not top_chunks or restricted:
        allowed = ROLE_PERMISSIONS.get(role, "general")
        return {
            "answer": f"As a {role}, you don't have access to documents matching this request. I can only answer questions from the {allowed} collections.",
            "sources": [],
            "retrieval_type": "hybrid_rag",
            "role": role
        }

    # Assemble context with breadcrumb headers
    context_str = "\n\n".join([
        f"File: {c.payload['source_document']} | Section: {c.payload['section_title']}\n{c.payload['text']}"
        for c in top_chunks
    ])

    # Generate grounded response
    response = ai_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"You are MediBot, internal assistant for MediAssist Health Network. Answer factually and concisely using ONLY this context:\n\n{context_str}\n\nQuestion: {question}"
    )

    # Collect source citations for transparency
    sources = [
        {
            "source_document": c.payload["source_document"],
            "section_title": c.payload["section_title"],
            "collection": c.payload["collection"]
        }
        for c in top_chunks
    ]

    return {
        "answer": response.text.strip(),
        "sources": sources,
        "retrieval_type": "hybrid_rag",
        "role": role
    }

# ----------------- STREAMLIT UI SECTION -----------------

st.title("🏥 MediBot: Enterprise Healthcare Assistant")
st.caption("Retrieval-Layer RBAC | Hybrid Search | Cross-Encoder Reranking | Analytical SQL RAG")

# Sidebar: Controls user role switching and displays authorized data boundaries
with st.sidebar:
    st.header("Security Context")
    selected_role = st.selectbox(
        "Active Role:",
        ["doctor", "nurse", "billing_executive", "technician", "admin"]
    )
    st.info(f"**Authorized Collections:**\n\n{ROLE_PERMISSIONS[selected_role].title()}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize conversational session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Source Citations"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source_document']}** | *{s['section_title']}* (`{s['collection']}`)")

# Handle new user input
if user_input := st.chat_input("Ask about clinical protocols, hospital policies, or database analytics..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display assistant answer
    with st.chat_message("assistant"):
        with st.spinner(f"Evaluating query under [{selected_role.upper()}] security boundaries..."):
            result = chat_router(user_input, selected_role)
            st.markdown(result["answer"])
            if result.get("sources"):
                with st.expander("📚 Source Citations"):
                    for s in result["sources"]:
                        st.markdown(f"- **{s['source_document']}** | *{s['section_title']}* (`{s['collection']}`)")

    # Save assistant message to state
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", [])
    })
