Markdown
# 📘 Student Document Assistant (SDA) - Advanced Hybrid RAG Engine

An advanced, production-grade Retrieval-Augmented Generation (RAG) academic application designed to ingest study materials (PDFs, TXT notes) and provide lightning-fast, context-verified, conversational answers using a hybrid lexical-semantic retrieval pipeline.

---

## 🚀 Key Architectural Features (Evaluation Highlights)

* **Dual-Engine Hybrid Retrieval:** Combines semantic concept searching (**ChromaDB Dense Vectors** using cosine distance metrics) with keyword-exact searching (**BM25 Sparse Lexical Store**). This completely eliminates the "vocabulary mismatch" problem common in standard vector-only pipelines.
* **Cross-Encoder Re-Ranking:** Implements a local, ultra-fast **FlashRank Cross-Encoder** model (`ms-marco-MiniLM-L-12-v2`). This scores, normalizes, and filters the raw retrieval pool down to the top 3 high-confidence context chunks, drastically minimizing LLM context window noise and maximizing accuracy.
* **Stateful Conversation Memory:** Uses a programmatic **Chat History Condensation** loop. If a user asks a follow-up question, the system queries the model to rewrite the input into an optimized, standalone searchable query before triggering document lookup.
* **Token Streaming Presentation:** Integrates **Live Token Streaming** via Streamlit UI wrappers and Groq LPU cloud acceleration endpoints (`llama-3.3-70b-versatile`), offering word-by-word streaming generation.
* **On-the-Fly Ingestion Pipeline:** Allows students to drag and drop new files (`.pdf`, `.txt`) directly via the web interface dashboard, automatically triggering text extraction, recursive splitting, and automatic database indexing.

---

## 🛠️ Tech Stack & System Requirements

* **Frontend Interface:** Streamlit (UI & State Handlers)
* **Orchestration Core:** LangChain Ecosystem (`langchain-core`, `langchain-community`, `langchain-groq`)
* **Local Embedding Weights:** HuggingFace Hub (`all-MiniLM-L6-v2`) via `local_files_only` execution
* **Vector Engine:** ChromaDB (HNSW index utilizing Cosine Space metrics)
* **Lexical Engine:** Rank-BM25 (Inverted Index Array serialized via Pickle)
* **Inference Model:** Llama-3.3-70b-Versatile via Groq Hardware Infrastructure

---

## 📂 Project Directory Structure

This is how your workspace layout looks inside the VS Code File Explorer sidebar. Ensure your working files and generated data directories mirror this arrangement exactly:

```text
RAG AI SDA/                     # Root Project Workspace Directory (Open this in VS Code)
│
├── myenv/                      # Isolated Python Virtual Environment (Omitted from git)
│
├── db/                         # Unified Database Storage Folder
│   ├── chroma_db/              # Persistent Vector Engine Database
│   │   ├── index/              # HNSW graph structural index arrays
│   │   └── chroma.sqlite3      # Embedded database schema attributes
│   └── bm25_store.pkl          # Serialized BM25 Lexical Matrix Array
│
├── docs/                       # Local Raw Knowledge Repository 
│   ├── syllabus.pdf            # Example Student Ingestion Source Material
│   └── lecture_notes.txt       # Example Raw Study Reference Document
│
├── .env                        # Secure Runtime Environment Secrets Tracker (Hidden)
├── ingestion.py                # Pipeline Core: Document Parsing & Dual Index Compilation
├── rag_engine.py               # Evaluation Core: Baseline Single-Query Diagnostic Engine
├── rag_engine_withchatmemory.py # Sequential Simulation: Dynamic Chat Query Re-writer Loop
├── app.py                       # Production Application: Real-time UI Presentation Client
└── requirements.txt            # Environment Packages Configuration Manifesto
```
---

## 📦 Installation & Environment Setup

Follow these steps sequentially to configure your local sandbox environment on a Windows machine:

### 1. Open your Project Folder in Terminal
Launch VS Code, open the `RAG AI SDA` folder, and open an integrated PowerShell terminal.

### 2. Create the Isolated Virtual Environment
Generate a clean environment container called `myenv` to keep your project packages separated from your global Python installation:
```bash
python -m venv myenv
```
### 3. Initialize Virtual Environment & Paths
Clone or open your project folder in your command line terminal interface and execute:
```bash
# Activate your virtual environment
.\myenv\Scripts\Activate.ps1
```
### 4. Install Project Dependencies
Sync your environment packages using the structured requirements manifesto:

```Bash
pip install -r requirements.txt
```
### 5. Configure Secrets Configuration Configuration
Create a file named .env in the root folder directory and securely add your developer authentication token key:

```Plaintext
GROQ_API_KEY=gsk_your_actual_copied_key_here
```
(Ensure there are no spaces or trailing quotation marks around the token).

---

## 🎮 Subsystem Execution & Files Overview
Your project repository is split into 4 optimized, standalone functional scripts:

### 1. Data Processing Layer (ingestion.py)
Parses raw documents from the local docs/ folder, splits string arrays recursively based on sentence structure, and provisions your matching dual index.

```Bash
python ingestion.py
```
### 2. Sandbox Verification Terminal (rag_engine.py)
A command-line tester designed to run a single mock question through the full hybrid lookup loop, verify your FlashRank cross-encoder scores, and test out responses without loading the web engine wrapper.

```Bash
python rag_engine.py
```
### 3. Memory Loop Simulation Terminal (rag_engine_withchatmemory.py)
A script that runs a persistent terminal command loop, tracking chat history and rewriting conversational context on the fly for continuous multi-turn interaction testing.

```Bash
python rag_engine_withchatmemory.py
```
### 4. Enterprise Production Web Client (app.py)
The primary entry point. Launches the interactive control panel, visualizes system health analytics, handles drag-and-drop ingestion, streams answers, and includes an expander to audit real-time pipeline confidence scores.

```Bash
streamlit run app.py
```
---

## 📊 Evaluation Presentation Walkthrough
When demonstrating this project to evaluators, highlight this unique architectural data flow:

```Plaintext
[User Chat Question Input]
           │
           ▼
[History Check Layer] ──► (If History Exists) ──► [Rewrite Standalone Query via LLM]
           │
           ├──► [ChromaDB Vector Index Search (k=3)] ──┐
           │                                           ├──► [Combined Pool & Deduplication]
           └──► [BM25 Inverted Lexical Search (k=3)] ──┘                  │
                                                                          ▼
[Groq Cloud Framework Layer] ◄── [Context Augmented Prompt] ◄── [FlashRank Cross-Encoder Filter (Top 3)]
           │
           ▼
[Word-by-Word Live Token Streaming UI Output]
```
