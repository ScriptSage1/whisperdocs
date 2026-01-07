# WhisperDocs

> **"Your Data, Your Device. Zero External API Calls. 100% Privacy."**

**WhisperDocs** is a Retrieval-Augmented Generation (RAG) pipeline designed for secure, offline document analysis. It enables users to chat with sensitive PDF/TXT documents without a single byte of data leaving the machine.

---

## Getting Started

### Prerequisites
* **Ollama** installed on the host machine.

### 1. Download the necessary models
Before launching the app, pull the required models to your machine.
```bash
ollama pull qwen3:14b # or any LLM of your choice, this model in specific fits perfectly within 16GB ram

ollama pull mxbai-embed-large

```
### 2. Download the necessary dependencies

```bash
pip install -r requirements.txt
```

### 3. To access the chat interface
```bash
streamlit run app.py
```
Navigate to http://localhost:8501 in your browser.

### Future Work
To add Docker containerization.

### Contributing
Contributions are welcome! Please feel free to submit a pull request.