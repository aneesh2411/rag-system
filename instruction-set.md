# MDP Spec — Elastic + Open LLM RAG (ELSER + Hybrid)

A single source-of-truth for building and demoing the Minimum Demo-able Product (MDP). This document merges the company’s requirements with our finalized implementation decisions so an AI (or teammate) can execute end-to-end without back-and-forth.

---

## 1) Purpose & Scope

Build a **simplified Retrieval-Augmented Generation (RAG)** system that:

* Uses **Elasticsearch** for indexing & retrieval with **ELSER (sparse)**, **dense vectors**, and **BM25**.&#x20;
* Uses an **open-source LLM** (Hugging Face or **Ollama**) for answer generation.&#x20;
* **Ingests PDFs** from a **Google Drive** folder.&#x20;
* Exposes both **FastAPI** and a **basic UI** (**Streamlit**), returning **grounded answers with citations**.&#x20;
* Delivered in a **GitHub repo**.&#x20;

**Goals:** end-to-end RAG, **hybrid retrieval (ELSER + dense + BM25)**, FastAPI service, simple UI, citations, guardrails, clear README.&#x20;
**Non-Goals:** Azure OpenAI/paid APIs, enterprise security, multi-tenant.&#x20;

---

## 2) Finalized Implementation Decisions

* **Search backend:** **Elasticsearch (local, Docker)** with **ELSER** enabled for sparse retrieval.
* **Ingest source:** Public **Google Drive** folder (ID: `1h6GptTW3DPCdhu7q5tY-83CXrpV8TmY_`).
* **Documents:** English PDFs; includes **scanned PDFs** → **OCR** when needed.
* **Embeddings (dense):** `sentence-transformers/all-MiniLM-L6-v2` (384-dim CPU-friendly).
* **LLM:** **Ollama** with **small model** (e.g., `phi3.5-mini`, fallback `llama3:8b`).
* **Retrieval modes:**

  * **ELSER-only**
  * **Hybrid** = ELSER + BM25 + dense, fused via **RRF(k=60)**; default **top\_k=5**.&#x20;
* **UI:** **Streamlit**, with retrieval-mode toggle and clickable **Drive-link citations**.&#x20;
* **Guardrails:** deny unsafe/illicit/PII; **“I don’t know”** when evidence is weak.&#x20;
* **Tests:** minimal unit tests (ingestion + retrieval).&#x20;
* **Repo:** **Public GitHub**.

---

## 3) Architecture Overview

**Pipeline:**

1. **Ingestion** → PDFs from Drive → text extraction → **chunking (\~300 tokens with overlap)** → metadata.&#x20;
2. **Indexing** → write **BM25 text**, **ELSER text\_expansion**, **dense vectors** into one index.&#x20;
3. **Retrieval** → **ELSER-only** or **Hybrid** (ELSER + BM25 + Dense) with **RRF**; configurable **top\_k**.&#x20;
4. **Answering** → small LLM with prompt = user question + selected chunks; **cite** sources; say **“I don’t know”** if evidence is weak; apply **guardrails**.&#x20;
5. **Interface** → **FastAPI** (`/ingest`, `/query`, `/healthz`) + **Streamlit** UI.&#x20;

---

## 4) Index & Mapping (single index)

**Fields**

* `content` — `text` (BM25)
* `text_expansion` — ELSER sparse features (rank-features or equivalent)
* `vector` — `dense_vector` (dims=384, cosine)
* `filename`, `drive_url`, `chunk_id` — `keyword`
  (“Create an Elastic index with mappings for all fields.”)&#x20;

**Chunk metadata**: `filename`, `drive_url`, `chunk_id`.&#x20;

---

## 5) Retrieval Logic

* **ELSER-only mode:** query against `text_expansion`.&#x20;
* **Hybrid mode:** run **three** searches: ELSER, BM25 (`content`), Dense (`vector`).
  Fuse results with **Reciprocal Rank Fusion**, `score = Σ 1/(k + rank_i)`, with **k = 60**; take **top\_k=5** (configurable).&#x20;

---

## 6) Guardrails

* **Refuse** unsafe, harmful, illicit, or PII-seeking queries.
* **Grounding:** only answer from retrieved evidence; else reply **“I don’t know.”**&#x20;

---

## 7) API Contract (FastAPI)

* `POST /ingest`
  **Purpose:** Pull PDFs from Drive, (OCR if needed), chunk, embed, index.
  **Body:** `{ "drive_folder_id": "1h6GptTW3DPCdhu7q5tY-83CXrpV8TmY_", "reindex": true }`
  **Returns:** `{ "documents_indexed": int, "chunks": int }`
  (Ref: API endpoints.)&#x20;

* `POST /query`
  **Body:** `{ "question": "…", "mode": "elser" | "hybrid", "top_k": 5 }`
  **Returns:**

  ```json
  {
    "answer": "…",
    "citations": [
      { "title": "file.pdf", "link": "https://drive.google.com/…", "snippet": "…" }
    ],
    "used_mode": "hybrid"
  }
  ```

  (Ref: API endpoints & citations requirement.) &#x20;

* `GET /healthz` → `{ "status": "ok" }`&#x20;

---

## 8) UI Spec (Streamlit)

* Single page with:

  * **Question input**
  * **Mode toggle** (ELSER-only vs Hybrid)
  * **Answer block**
  * **Citations** list showing **title + Drive link + snippet** (clickable)
    (Ref: UI requirements.)&#x20;

---

## 9) Ingestion Details

* **Source:** Public **Google Drive** folder by ID.
* **Extract:** Text from PDFs; **OCR** pages with little/no text (English).
* **Chunking:** ≈**300 tokens** with overlap; attach metadata (`filename`, `drive_url`, `chunk_id`).&#x20;

---

## 10) Answering Strategy

* Prompt = user question + **N** top chunks (cite all used).
* If no chunk passes evidence threshold → reply **“I don’t know.”**&#x20;
* Model: **Ollama** small LLM for low latency (≤ 3s target on small corpus).&#x20;

---

## 11) Non-Functional Requirements

* **Latency:** ≤ **3s** per query (small dataset).
* **Cost:** Use **free/open** models only.
* **Repro:** `requirements.txt`; **unit tests** for ingestion + retrieval; UI should be minimal & responsive. &#x20;

---

## 12) Deliverables & Evaluation (for README and demo)

**Deliverables:** GitHub repo (code + `requirements.txt` + README steps), **5-minute demo video** (ingest, API query, UI query, citations, guardrails).&#x20;
**Evaluation:** correctness end-to-end, code quality, **Elastic use (ELSER + dense + BM25 + hybrid)**, working API & UI with citations, guardrails, creativity (optional).&#x20;

---

## 13) Setup & Run (deterministic checklist)

1. **Clone repo** (public).
2. **Start services** with Docker Compose:

   * **Elasticsearch** (single node)
   * **Ollama** (pull chosen small model)
3. **Enable ELSER** in Elasticsearch (deploy the ELSER model).
4. **Create index & pipeline** (ELSER inference → `text_expansion`; mapping includes `content`, `vector`, metadata).&#x20;
5. **Configure app**: set `DRIVE_FOLDER_ID=1h6GptTW3DPCdhu7q5tY-83CXrpV8TmY_`.
6. **Run API** (`uvicorn app.main:app`) and **UI** (`streamlit run ui/app.py`).
7. **Ingest** via `POST /ingest` (or CLI wrapper).
8. **Query** via `POST /query` and UI; verify **citations** link to Drive.
9. **Run tests**: `pytest`.

> Notes: If OCR is required, container should include `ocrmypdf` + `tesseract` (English). Keep OCR only for low-text pages to preserve speed.

---

## 14) API/LLM/Ranking Parameters (tunable)

* `top_k`: default **5** (UI slider 3–10).&#x20;
* **RRF k**: **60** (constant).
* **Chunk size**: **\~300 tokens**; overlap \~15–20%.&#x20;
* **Dense model**: `all-MiniLM-L6-v2`.&#x20;
* **Modes**: `"elser"` | `"hybrid"`.&#x20;

---

## 15) Testing Strategy (minimum)

* **Ingestion tests**: number of PDFs discovered, number of chunks produced (≥1), OCR path triggers for low-text pages.
* **Retrieval tests**:

  * ELSER-only returns ≥1 doc for a known term.
  * Hybrid RRF ranks intersection of candidates (stability on tie-breaks).
  * No-evidence → **“I don’t know.”**
    (“Provide unit tests for ingestion + retrieval.”)&#x20;

---

## 16) Demo Script (5 minutes)

1. Show **/healthz** OK.&#x20;
2. Trigger **/ingest** → logs show Drive files, chunk counts.&#x20;
3. **API**: `POST /query` (mode: **hybrid**) → returns **answer with citations**.&#x20;
4. **UI**: Same question; toggle **ELSER-only** vs **Hybrid** → compare answers and citings.&#x20;
5. Ask an off-corpus question → show **“I don’t know.”** + guardrail refusal for unsafe prompt.&#x20;

---

## 17) Folder Structure (target)

```
/app
  main.py          # FastAPI: /ingest, /query, /healthz
  ingest.py        # Drive list, PDF→text, OCR, chunking
  indexer.py       # ELSER pipeline call, dense embed, ES bulk index
  retrieval.py     # ELSER/BM25/Dense searches + RRF
  llm.py           # Ollama wrapper
  guardrails.py    # allow/deny + evidence gating + "I don't know"
  settings.py      # config (ES URL, DRIVE_FOLDER_ID, model names)
/ui
  app.py           # Streamlit UI (mode toggle, answer, citations)
/tests
  test_ingest.py
  test_retrieval.py
docker-compose.yml
requirements.txt
README.md
```

---

## 18) Acceptance Checklist

* [ ] Ingests PDFs from the **specified Drive folder**.&#x20;
* [ ] Index has **BM25**, **ELSER**, and **dense** fields.&#x20;
* [ ] **ELSER-only** and **Hybrid (RRF)** modes work; **top\_k** configurable.&#x20;
* [ ] Answers use **open LLM** and include **citations** (title + link + snippet). &#x20;
* [ ] **FastAPI** endpoints & **Streamlit** UI are functional.&#x20;
* [ ] **Guardrails** active; **“I don’t know”** on weak evidence.&#x20;
* [ ] Repo has **README**, **requirements.txt**, and **unit tests**.&#x20;
* [ ] Demo video covers ingest, queries, citations, guardrails.&#x20;

---

## 19) Future Work (optional, de-scoped for MDP)

* Result caching; cross-encoder reranking; richer UI analytics; multi-tenant & auth. (Optional creativity mentioned.)&#x20;

---

**Owner:** You (Sesha)
**Deadline:** Submit by **Aug 28** (per email).
**Source Requirements:** “Rag System Requirements – Elastic + Open LLM” v1.1, **2025-08-23**.&#x20;

---

If you want this dropped into your repo as `README.md` (or `MDP_SPEC.md`) with a runnable `docker-compose.yml` and example index mapping JSON, say the word and I’ll paste them in ready to run.
