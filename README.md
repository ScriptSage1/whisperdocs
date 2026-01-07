# WhisperDocs

> **"Your Data, Your Device. Zero External API Calls. 100% Privacy."**

**WhisperDocs** is a Retrieval-Augmented Generation (RAG) pipeline designed for secure, offline document analysis. It enables users to chat with sensitive PDF/TXT documents without a single byte of data leaving the machine.

---

## Getting Started

### Prerequisites
* **Docker**
* **Ollama** installed on the host machine.

### 1. Download the necessary models
Before launching the app, pull the required models to your machine.
```bash
ollama pull qwen3:14b # or any LLM of your choice, this model in specific fits perfectly within 16GB ram

ollama pull mxbai-embed-large

```
### 2. Launching the app

```bash
docker-compose up --build
```

### 3. To access the chat interface
Navigate to http://localhost:8501 in your browser.

### Configuration
The system is highly configurable via docker-compose.yml or environment variables.

### Contributing
Contributions are welcome! Please feel free to submit a pull request.