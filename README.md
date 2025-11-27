# Agentic Web Researcher (LangGraph + MCP)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-orange)
![MCP](https://img.shields.io/badge/Protocol-MCP-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-grey)

An advanced **Agentic AI Client** that performs autonomous web research using the **Model Context Protocol (MCP)**.
This project demonstrates a decoupled microservices architecture where the reasoning engine (Client) communicates with tools (Server) via standardized protocols, featuring robust observability and asynchronous state management.

🔗 **Live Demo:** [Insert Render App URL Here]

---

## 🚀 Architecture Overview

The system is split into two decoupled microservices communicating over HTTP/SSE:

1.  **The Client (Brain):** A Streamlit application hosting a **LangGraph ReAct Agent**. It manages conversation history, reasoning loops, and connects to the remote tool server via MCP.
2.  **The Server (Tools):** A **FastAPI + FastMCP** server hosted on Render. It exposes the Tavily Search API as a standardized MCP Tool.

### Key Engineering Features
*   **Decoupled Security:** The Client (LLM) holds NO API keys for the search engine. It communicates purely via the MCP contract, simulating secure enterprise tool usage.
*   **Hybrid Grounding Strategy:** The agent uses a dual-layer approach: it consumes a high-level summary from the search engine for context, but strictly verifies facts against **raw JSON sources** to ensure accurate provenance and minimize hallucinations.
*   **Transparent Reasoning (XAI):** The UI exposes the Agent's "Chain of Thought" in real-time, allowing users to audit the decision-making process (e.g., why a specific tool was called) and verify intermediate steps.
*   **Live FinOps Monitoring:** Includes a real-time dashboard for token usage (Input/Output) and estimated cost calculation, essential for managing LLM operational expenses in production.
*   **Concurrency Management:** Implements a custom background `asyncio` event loop to handle non-blocking MCP streams within Streamlit's synchronous runtime.
*   **Full Observability:** Logs are dispatched to multiple sinks: Local JSONL, **Redis** (as a high-throughput buffer), and **BetterStack** (for cloud monitoring).
*   **State Persistence:** Conversation state is preserved across turns using an Async SQLite Checkpointer.

---

## 🛠️ Tech Stack

*   **Orchestration:** LangGraph, LangChain Core
*   **LLM:** Google Gemini 2.0 Flash (via LangChain Adapter)
*   **Protocol:** Model Context Protocol (MCP) - Python SDK
*   **Backend Server:** FastAPI, FastMCP, Uvicorn
*   **Frontend:** Streamlit
*   **Database/State:** Async SQLite, Redis (ValKey)

---

## 📦 Installation & Setup

### Prerequisites
*   Python 3.10+
*   A Google AI Studio API Key (Gemini)
*   A Tavily API Key
*   (Optional) Redis URL and BetterStack Source Token for logging

### 1. Clone the Repository
```bash
git clone https://github.com/danielecelsa/chatbot-with-search.git
cd chatbot-with-search
```
### 2. Environment Variables
Create a .env file in the root directory:
```bash
# LLM Configuration
GOOGLE_API_KEY=your_google_api_key
GENAI_MODEL=your_model

# MCP Server Configuration
# If running the server locally: http://localhost:8080/tav/sse
# If using the live demo server:
MCP_TAVILY_URL=https://tavily-mcp-server-wqe5.onrender.com/tav/mcp/
MCP_SERVER_NAME=tavily

# Logging (Optional)
REDIS_URL=redis://...
LOGTAIL_SOURCE_TOKEN=...
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Application
```bash
streamlit run app.py
```
---
## 🧩 How It Works (The ReAct Loop)
*   **Reasoning:** The LangGraph agent analyzes the user query.
*   **Protocol Handshake:** The client establishes a connection to the MCP Server.
*   **Tool Execution:** The server executes the Tavily API call and returns a structured JSON object containing both a synthesis and raw content excerpts.
*   **Verification:** The agent reads the JSON, verifies the summary against the raw excerpts, and generates a citation-backed response.

---

## 👨‍💻 Author
Daniele Celsa

*   [Portfolio Website](https://danielecelsa.com)
*   [LinkedIn](https://diretta.it)