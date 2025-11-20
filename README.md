# Agentic Web Researcher (LangGraph + MCP)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-orange)
![MCP](https://img.shields.io/badge/Protocol-MCP-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-grey)

An advanced **Agentic AI Client** that performs autonomous web research using the **Model Context Protocol (MCP)**.
This project demonstrates a decoupled architecture where the reasoning engine (Client) communicates with tools (Server) via standardized protocols, featuring robust observability and asynchronous state management.

---

## 🚀 Architecture Overview

The system is split into two decoupled microservices:

1.  **The Client (Brain):** A Streamlit application hosting a **LangGraph ReAct Agent**. It manages conversation history, reasoning loops, and connects to tools via MCP over SSE (Server-Sent Events).
2.  **The Server (Tools):** A **FastAPI + FastMCP** server hosted on Render. It exposes the Tavily Search API as a standardized MCP Tool.

### Key Features
*   **Decoupled Tooling:** The LLM (Client) and the Tool Execution (Server) are completely separate, communicating only via the MCP standard.
*   **Safe Grounding (Provenance):** The agent uses a hybrid approach, consuming both a search summary and raw source data (JSON) to verify facts and cite URLs, minimizing hallucinations.
*   **Asynchronous Core:** Implements a custom background `asyncio` loop to handle non-blocking MCP streams within the synchronous Streamlit environment.
*   **Full Observability:** Logs are dispatched to multiple sinks: Local JSONL, **Redis** (for live monitoring), and **BetterStack** (cloud logging).
*   **Persistence:** Conversation state is preserved using an Async SQLite Checkpointer.

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

### 2. Environment Variables
Create a .env file in the root directory:
# LLM Configuration
GOOGLE_API_KEY=your_google_api_key
GENAI_MODEL=your_model

# MCP Server Configuration
# If running the server locally: http://localhost:8080/tav/sse
# If using the live demo server:
MCP_TAVILY_URL=https://tavily-mcp-server-wqe5.onrender.com/tav/mcp
MCP_SERVER_NAME=tavily

# Logging (Optional)
REDIS_URL=redis://...
LOGTAIL_SOURCE_TOKEN=...

### 3. Install dependencies
```bash
pip install -r requirements.txt

### 4. Run the Application
```bash
streamlit run app.py

---

## 🧩 How It Works (The ReAct Loop)
*   **User Query:** The user asks a question (e.g., "Latest news on AI?").
*   **Reasoning:** The LangGraph agent analyzes the query. It decides it needs external info.
*   **Tool Call (MCP):** The agent constructs a tool call (web_search). This is sent via SSE to the MCP Server.
*   **Execution:** The Server executes the Tavily API call and formats the result as a structured JSON object containing both a summary and raw sources.
*   **Observation:** The JSON result is sent back to the Client.
*   **Response Generation:** The agent reads the observation, verifies the facts against the sources, and generates the final answer with citations.

---

## 👨‍💻 Author
Daniele Celsa

*   [Portfolio Website] ()
*   [LinkedIn] ()