# Student Document Assistant 🎓📄

The **Student Document Assistant** is a fully local, privacy-first **Retrieval-Augmented Generation (RAG)** system designed to help students interactively query, summarize, and understand textbooks, lecture notes, academic PDFs, and syllabi. 

By leveraging state-of-the-art open-source Large Language Models (LLMs) and vector embedding pipelines locally on consumer-grade hardware, this project guarantees absolute data privacy with zero external API fees or dependencies.

---

## 🚀 Key Architectural Features
- **100% Local Processing:** Document parsing, embedding vectors, mathematical lookups, and response generation execute locally via Ollama and HuggingFace. No information leaves your machine.
- **Contextual Query Reformulation:** Intelligently rewrites natural language follow-up chat inputs (e.g., *"Why does it happen?"*) into rich, standalone semantic search queries based on the ongoing conversation history.
- **Factual Guardrails:** Hardcoded instruction injection completely eliminates model hallucinations, forcing the assistant to only answer using provided document context fragments.

---

## 🛠️ System Pipeline & Architecture

The project consists of an optimized two-stage pipeline architecture:

### 1. Document Ingestion Pipeline (`Ingestion.py`)
1. **Document Loading:** Leverages `PyPDFDirectoryLoader` to parse structural text elements out of any standard PDF document dropped in the `docs/` repository block.
2. **Semantic Text Chunking:** Employs a `RecursiveCharacterTextSplitter` configured for a window matrix size of **800 characters** and a **150-character sliding overlap**. This ensures structural boundaries like paragraphs and chapters are kept topically coherent.
3. **Vector Vectorization & Storage:** Feeds the character chunks into the local `all-MiniLM-L6-v2` transformer model to output 384-dimensional dense floating-point vector profiles. These profiles are written onto disk using a persistent `ChromaDB` matrix matching via the **Cosine Similarity Metric**.

### 2. Conversational Context Engine (`rag_engine_withchatmemory.py`)
1. **Memory Ingestion:** Appends continuous user questions and system responses using a LangChain memory-tracking array list (`HumanMessage` and `AIMessage`).
2. **Contextual Processing:** The local LLM rewrites the user's newest prompt against past interaction transcripts to build clean standalone context vectors.
3. **Information Retrieval:** Queries ChromaDB for the top **5 (`k=5`) most similar** text chunks matching the intent.
4. **Response Generation:** Builds a structural system prompt wrapper containing the source documents, query text, and strict rules, executing a deterministic context resolution pass via a local **Llama 3.2 (3B)** model wrapper.

---

## 📦 Directory Structure

Set up your repository directory structure as follows before initializing execution steps:

```text
STUDENT-DOCUMENT-ASSISTANT/
│
├── db/
│   └── chroma_db/                 # Auto-generated during data ingestion
├── docs/                          # DROP YOUR STUDY/COURSE PDFs HERE!
│   ├── calculus_textbook.pdf
│   └── chemistry_notes.pdf
│
├── Ingestion.py                   # Document ingestion pipeline
├── rag_engine.py                  # Single-turn validation testing module
├── rag_engine_withchatmemory.py   # Interactive multi-turn terminal app
└── .env                           # Local environment configuration file



## ⚙️ Requirements & Installation
​Ensure you have Python (version 3.9 up to 3.11) and the desktop daemon version of Ollama installed.

​1. Initialize Local LLM Model
​Download and install Ollama. Open your system terminal shell and download the micro LLM model footprint:
ollama pull llama3.2:3b

2. Setup Dependencies
​Install the required LangChain ecosystems, embedding tools, and vector database extensions through pip:
pip install langchain-community langchain-text-splitters langchain-huggingface langchain-chroma langchain-ollama python-dotenv pypdf

## 🏁 Execution Guide
​Step 1: Ingest Data Documents
​Place all target course files inside the docs/ directory folder, then compile them into mathematical vector maps:

python Ingestion.py

The script will log processing information to the terminal, print text chunk slices, and confirm database setup at db/chroma_db/.

Step 2: Interact via Terminal
​Launch the contextualized question-and-answer terminal loop app:

python rag_engine_withchatmemory.py

##💬 Sample Interaction Logs
Initializing local Llama 3.2 via Ollama...
Ask me questions! Type 'quit' to exit.

Your question: What is work hardening?
Found 5 relevant documents:
  Doc 1: Work hardening, also known as strain hardening, is...
  Doc 2: The strengthening of a metal by plastic deformation...
Answer: Work hardening is the structural strengthening of a metallic material caused by cold plastic deformation.

Your question: Why does it happen?
Searching for: Why does work hardening happen?
Found 5 relevant documents:
  Doc 1: This occurs due to dislocation movements within the crystal lattice...
Answer: It happens because cold deformation generates extra internal crystalline dislocations, creating an atomic gridlock that resists further deformation.


## 🔧 Technical Hyperparameters & Customization

​chunk_size=800, chunk_overlap=150: Calibrated closely to support the 256 token limits imposed by the all-MiniLM-L6-v2 model embedding layers without experiencing content truncations.
​collection_metadata={"hnsw:space": "cosine"}: Configures ChromaDB to run multi-dimensional vector angular comparison instead of raw distance spacing metrics, yielding accurate semantic rankings regardless of document length variances.
​temperature=0: Set within ChatOllama to disable random word selection, establishing deterministic, non-creative, and factual answers for your study sessions.



***

### Summary of What the Provided PDF Document Covers:
1. **Beautiful Clean Layout:** Follows a high-quality academic paper design scheme with structural side-borders and clean color-coded layouts tailored specifically for engineering and software development projects.
2. **Clear Technical Explanations:** Explains how the pipeline reads files from `docs/`, cuts them down seamlessly through `RecursiveCharacterTextSplitter`, stores embeddings via HuggingFace's `all-MiniLM-L6-v2`, and generates data using `Ollama`.
3. **Clear Code Snippets:** Captures instructions on setting up python environments, installing libraries, and pulling models via Ollama.
4. **Chat Logic Explanation:** Documents how the history array list seamlessly formats follow-up inputs into independent search coordinates before approaching ChromaDB.






☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆▪︎☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆
yet to add about streamlit 
●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●



