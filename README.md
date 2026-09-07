# MediBot — Advanced RAG for MediAssist Health Network

An enterprise-grade internal healthcare assistant that answers staff questions strictly from authorized documents and relational operational databases. Role-Based Access Control (RBAC) is enforced directly inside the Qdrant vector retrieval engine via metadata filters, guaranteeing that adversarial prompts and jailbreaks cannot surface documents outside an authenticated role's permitted scope.

* **Structure-aware ingestion** — Parsed via Docling and chunked using a hierarchical strategy (headings, tables, and lists remain intact; every chunk carries its complete parent heading path).
* **Hybrid retrieval (Dense + BM25)** — Dense semantic embeddings combined with sparse BM25 lexical tokens stored in a unified Qdrant collection and fused via Reciprocal Rank Fusion (RRF).
* **Cross-encoder reranking** — Top-10 hybrid search candidates rescored jointly and filtered down to the top-3 most relevant contexts before prompt assembly.
* **SQL RAG** — Operational analytics executed over `mediassist.db` via a sanitized natural-language-to-SQL pipeline with schema-hallucination guards.
* **Retrieval-layer RBAC** — Security enforced by attaching `access_roles` metadata filters to both dense and sparse query legs inside the vector database.

---

## Complete Query Flow Architecture

```text
Staff Authentication (Role: doctor, nurse, billing_executive, technician, admin)
     │
     ▼
Session Context & Role Established (JWT / UI Session)
     │
     ▼
Incoming Question + Authenticated Role
     │
     ├──► Analytical / Operational Database Question?
     │         │
     │         ├──► Role Permitted? (billing_executive, admin)
     │         │         ├── Yes ──► Text-to-SQL Engine ──► Schema Grounding ──► SQLite (mediassist.db)
     │         │         └── No  ──► Standardized RBAC Refusal Message
     │         │
     └──► Policy / Protocol / Clinical Guideline Question?
               │
               ▼
          Qdrant Vector Database
          [Payload Filter: access_roles metadata]
               │
          Hybrid Retrieval
          ├── Dense Embeddings (sentence-transformers/all-MiniLM-L6-v2)
          └── Sparse Lexical BM25 (FastEmbed)
               │
               ▼
          Reciprocal Rank Fusion (RRF)
               │
               ▼
          Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2: top-10 ──► top-3)
               │
               ▼
          LLM Generation ──► Grounded Response + Source Breadcrumb Citations
