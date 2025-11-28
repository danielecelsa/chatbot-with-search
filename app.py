# client_react_langgraph.py
# Streamlit client using LangGraph prebuilt ReAct agent + MCP Tavily tool.
# Key points:
# - Uses a LangGraph checkpointer for persistence (Async SQLite if available, otherwise in-memory).
# - Keeps features: conversation memory, safe intermediate-step trace (tools & observations), separate Provenance section, token usage & cost, JSONL logging.

print("=== PRINT FROM PROCESS START ===")

import os
import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid
import valkey
import time
from time import perf_counter

import streamlit as st
from dotenv import load_dotenv

# LangGraph / LangChain Core
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate


# LLM adapter (example: Google Gemini). Replace with your LLM adapter.
from langchain_google_genai import ChatGoogleGenerativeAI

# MCP tools
from langchain_mcp_adapters.client import MultiServerMCPClient

from logging_config import (
    get_logger,
)

# Helpers (token estimation, cost computation, safe serialization)
from helpers import (
    process_agent_events,
    compute_cost,
    get_user_info,
)

from async_bg import collect_events_from_agent

from prompts import RESEARCHER_PROMPT


if os.getenv("RENDER") != "true":
    load_dotenv()

# ------------------------------
# Configuration
# ------------------------------
MODEL = os.environ.get("GENAI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

COST_PER_1K_INPUT = float(os.getenv("COST_PER_1K_TOKENS_USD_INPUT", "0.002"))
COST_PER_1K_OUTPUT = float(os.getenv("COST_PER_1K_TOKENS_USD_OUTPUT", "0.002"))

MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "tavily")
MCP_URL = os.getenv("MCP_TAVILY_URL", "https://tavily-mcp-server-wqe5.onrender.com/tav/mcp/")

MCP_CONFIG = {
    MCP_SERVER_NAME: {
        "url": MCP_URL,
        "transport": "streamable_http",
    }
}

# ------------------------------
# LOGGING SETUP
# ------------------------------

logger_local = get_logger("local")
logger_betterstack = get_logger("betterstack")
logger_redis = get_logger("redis")
logger_all = get_logger("all") 

# --------------------------------------
# Streamlit session state initialization
# --------------------------------------
if "conversation_thread_id" not in st.session_state:
    st.session_state.conversation_thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am your personal chatbot. You can ask me anything, I am even able to search the web for you. How can I help you?")] 
if "trace" not in st.session_state:
    st.session_state.trace = []
if "provenance" not in st.session_state:
    st.session_state.provenance = {}
if "latency" not in st.session_state:
    st.session_state.latency = 0.0

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0

if "input_tokens_last" not in st.session_state:
    st.session_state.input_tokens_last = 0
if "output_tokens_last" not in st.session_state:
    st.session_state.output_tokens_last = 0
if "total_tokens_last" not in st.session_state:
    st.session_state.total_tokens_last = 0

if "usd" not in st.session_state:
    st.session_state.usd = 0.0
if "usd_last" not in st.session_state:
    st.session_state.usd_last = 0.0

# ------------------------------
# LLM
# ------------------------------
@st.cache_resource
def get_llm():
    """Get the LLM instance."""
    try:
        llm = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        #convert_system_message_to_human=True,
        safety_settings=None,
        )
    except Exception as e:
        logger_all.exception("Could not initialize LLM: %s", e)
        llm = None 

    return llm


@st.cache_resource
def get_checkpointer():
    """Get a checkpointer for the chatbot."""

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver as SqliteCheckpointer
        checkpointer = SqliteCheckpointer.from_conn_string(os.environ.get("CHECKPOINT_DB", "./langgraph_state.sqlite"))
    except Exception:
        try:
            from langgraph.checkpoint.memory import InMemorySaver as InMemoryCheckpointer
            checkpointer = InMemoryCheckpointer()
        except Exception:
            checkpointer = None
    return checkpointer

# ------------------------------
# System prompt - general, not biased to any particular question type
# ------------------------------
@st.cache_resource
def get_prompt_template():
    """Get the chat prompt template with system message and history placeholder."""
    
    system_message = RESEARCHER_PROMPT.format(current_date=datetime.now().date().isoformat())
    system_template = SystemMessagePromptTemplate.from_template(system_message)

    hist = MessagesPlaceholder(variable_name="messages")
    prompt = ChatPromptTemplate.from_messages([system_template, hist])
    
    return prompt

# ------------------------------
# Build agent with checkpointer
# ------------------------------
# @st.cache_resource
def build_agent():
    """Build the ReAct agent with tools and checkpointer."""
    # MCP tools
    tools = []
    try:
        mcp_client = MultiServerMCPClient(MCP_CONFIG)
        tools = asyncio.run(mcp_client.get_tools())
    except Exception as e:
        logger_all.error("Could not initialize MCP client/tools: %s", e)

    agent = create_react_agent(
        model=get_llm(),
        tools=tools,
        prompt=get_prompt_template(),
        checkpointer=get_checkpointer(),
    )

    return agent

agent = build_agent()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Decoupled Research Agent (MCP)", page_icon="🔌", layout="wide")
st.title("🔌 Decoupled Research Agent (MCP)")

body = """
**Autonomous Web Researcher - powered by MCP**

This application demonstrates a **decoupled Agentic Architecture** using the **Model Context Protocol (MCP)**.
Instead of embedding tools directly into the application code, the ReAct Agent (Client) connects to a remote **MCP Server** via SSE (Server-Sent Events) to access *search capabilities*.

**Key Features:**
*   **🔌 Microservices Architecture:** The LLM logic (Client) is completely decoupled from the Tool execution (MCP Server). This creates a standardized interface for connecting AI models to external data.
*   **🧠 ReAct Loop:** Uses **LangGraph** to implement a cyclic reasoning pattern: *Thought → Action (Remote Call) → Observation → Answer*.
*   **🛡️ Anti-Hallucination:** Implements a **"Hybrid Grounding"** strategy. The agent validates the Search Engine's AI summary against the raw source metadata (URLs/Snippets) before generating a response.
*   **📊 Transparency:** Full visibility into the **Protocol Trace**, showing exactly what data was sent over the wire and how costs were incurred.
"""

with st.expander('About this demo (Read me)', expanded=False):
    st.markdown(body)

# ------------------------------
# Chat submission
# ------------------------------
user_query = st.chat_input("Type your message here...")
if user_query:
    
    # Get user info as soon as they submit a query
    user_details = get_user_info(logger_all)
    logger_all.info(
        f"New query received from IP: {user_details['ip']} "
        f"with User-Agent: {user_details['user_agent']}"
    )

    st.session_state.chat_history.append(HumanMessage(content=user_query))

    thread_id = st.session_state.conversation_thread_id 
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": st.session_state.chat_history}

    final_answer_obj = None
    ai_content = ""

    # ------------------------------
    # Collect events using background loop helper
    # - this internally runs agent.astream(inputs, config=config, stream_mode="updates")
    # ------------------------------
    
    start = perf_counter()
    with st.spinner("Thinking..."):
        try:
            # 1. Collect all events from the agent in the background
            events = collect_events_from_agent(agent, inputs, config=config, timeout=120)

            # 2. Process the events with our new, focused parser
            final_answer_obj, trace, provenance, usage_last = process_agent_events(events)

            # 3. Update session state for UI
            st.session_state.trace = trace
            st.session_state.provenance[thread_id] = provenance

            # 4. Update token counters for both last interaction and total
            st.session_state.input_tokens_last = usage_last.get("input_tokens", 0)
            st.session_state.output_tokens_last = usage_last.get("output_tokens", 0)
            st.session_state.total_tokens_last = usage_last.get("total_tokens", 0)

            st.session_state.total_input_tokens += st.session_state.input_tokens_last
            st.session_state.total_output_tokens += st.session_state.output_tokens_last
            st.session_state.total_tokens += st.session_state.total_tokens_last

            if final_answer_obj:
                ai_content = final_answer_obj.content
            else:
                ai_content = "I'm sorry, I couldn't find a final answer."
                st.error("Could not extract a final answer from the agent's response.")

        except Exception as e:
            ai_content = f"An error occurred: {e}"
            st.error(ai_content)
            st.session_state.trace = []

    st.session_state.latency = perf_counter() - start

    # Calculate costs for UI display
    try:
        st.session_state.usd = compute_cost(st.session_state.total_input_tokens, st.session_state.total_output_tokens, COST_PER_1K_INPUT, COST_PER_1K_OUTPUT)
    except Exception:
        logger_all.exception("Error computing total cost: %s", e)
        st.session_state.usd = 0.0
    
    try:
        st.session_state.usd_last = compute_cost(st.session_state.input_tokens_last, st.session_state.output_tokens_last, COST_PER_1K_INPUT, COST_PER_1K_OUTPUT)
    except Exception:
        logger_all.exception("Error computing last cost: %s", e)
        st.session_state.usd_last = 0.0
    
    st.session_state.chat_history.append(AIMessage(content=ai_content))

    # --- LOGGING ---
    logger_all.info("Latency for full response: %.2f seconds", st.session_state.latency)
    logger_all.info("Last interaction tokens: %d (input), %d (output), %d (total)", st.session_state.input_tokens_last, st.session_state.output_tokens_last, st.session_state.total_tokens_last)
    logger_all.info("Estimated last interaction cost: $%.5f", st.session_state.usd_last)
    logger_all.info("Total tokens so far: %d (input), %d (output), %d (total)", st.session_state.total_input_tokens, st.session_state.total_output_tokens, st.session_state.total_tokens)
    logger_all.info("Estimated total cost so far: $%.5f", st.session_state.usd)

    data_dict={
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "messages": [{"role": m.__class__.__name__, "content": m.content} for m in st.session_state.chat_history],
        "trace": st.session_state.trace,
    }
    logger_all.info(json.dumps(data_dict, indent=2, ensure_ascii=False))


# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("⚙️ System Architecture", divider="rainbow")

    st.info("This PoC demonstrates the **MCP Standard** for decoupled, scalable tool integration.")

    # ------------------------------
    # Info Section
    # ------------------------------
    with st.expander("🛠️ Architecture & Tech Stack"):
        st.markdown("""
        **Core Orchestration:**
        - `LangGraph`: State machine for ReAct loops.
        - `LangChain Core`: Prompt management.
        
        **Connectivity (The Highlight):**
        - ***Protocol:*** [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
        - ***Transport:*** SSE (Server-Sent Events) over HTTP.
        - ***Server:*** A dedicated `FastAPI` / `FastMCP` server running the Tavily Search tool.
        
        **Observability & FinOps:**
        - ***Protocol Trace:*** JSON-level visibility into Tool Inputs/Outputs.
        - ***Provenance Parsing:*** Automatic extraction of URLs for citation verification.
        - ***Live Metrics:*** Real-time Token Usage & USD Cost estimation.
        - ***Distributed Logging:***  Structured logs (Redis + BetterStack) for remote monitoring.
        """)

    with st.expander("🧪 How to Test (Scenarios)"):
        st.caption("Try these inputs to see the specific architectural patterns in action:")
        st.markdown("**1. Real-Time Knowledge Retrieval**:")
        st.write("##### *The agent must call the remote MCP tool to answer*")
        st.markdown("> *Who won the Davis Cup in 2025?*")
        
        st.markdown("**2. Complex Multi-Step Reasoning**:")
        st.write("##### *Requires multiple search queries synthesized into one answer*")
        st.markdown("> *When was the last time a total solar eclipse was visible from New York City? Once you find it, tell me also what other important events happened that year*")

        st.markdown("**3. Ambiguity Handling**:")
        st.write("##### *The agent should refuse to search blindly*")
        st.markdown("> *What happened in the summer?*")

        with st.expander("👀 **What to watch:**"):
            st.markdown("**1. The MCP Protocol Trace**:")
            st.caption("👇 *Check the **🧠 Agent's Reasoning Steps** section below*")
            st.markdown("""
                        This is not just a chat log. It is a trace of the **Remote Procedure Call**. You will see the JSON payload sent to the MCP Server and the structured JSON received back.
                        """)

            st.markdown("**2. Source Provenance (Anti-Hallucination)**:")
            st.caption("👇 *Check the **🗂️ Sources' Provenance** section below*")
            st.markdown("""
                        Unlike standard chatbots, this system separates the **AI Summary** from the **Raw Evidence**. Click the expandable lists to verify the actual URLs used to ground the answer.
                        """)
            
            st.markdown("**3. Cost Attribution & FinOps**:")
            st.caption("👇 *Check the **📊 Live Metrics** section below*")
            st.markdown("""
                        Even for single-agent architectures, tracking **Input vs. Output** token density is crucial for forecasting production costs of research-heavy workloads.
                        """)
    st.markdown("---")

    # ------------------------------
    # Metrics Section
    # ------------------------------
    st.subheader("📊 Live Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=":blue[Latency]", value=f"{st.session_state.latency:.2f}s", help="Time taken to generate the LAST response.")
    with col2:
        st.metric(label=":blue[Session Cost]", value=f"${st.session_state.usd:.4f}", help="Estimated cost for the entire session calculated using %.4f per 1K input tokens and %.4f per 1K output tokens" % (COST_PER_1K_INPUT, COST_PER_1K_OUTPUT))
    
    st.caption("💡 *Metrics update in real-time to track API costs.*")
    st.caption(f"Token Usage: :blue[{st.session_state.total_tokens}] total tokens used.", help="Total number of tokens consumed in the entire session.")
    
    with st.expander(":blue[🔎 Token Usage Breakdown:]"):        
        st.markdown(f"#### :blue[Last Interaction Tokens:] {st.session_state.total_tokens_last}")
        col1, col2 = st.columns(2)
        col1.metric("Input Tokens", f"{st.session_state.input_tokens_last}")
        col2.metric("Output Tokens", f"{st.session_state.output_tokens_last}")
        st.metric("Last Est. Cost (USD)", f"${st.session_state.usd_last:.4f}", help="Estimated cost for the last interaction.")

        st.markdown(f"#### :blue[Session Total Tokens:] {st.session_state.total_tokens}")
        col3, col4 = st.columns(2)
        col3.metric("Total Input", f"{st.session_state.total_input_tokens}")
        col4.metric("Total Output", f"{st.session_state.total_output_tokens}")
        st.metric("Total Est. Cost (USD)", f"${st.session_state.usd:.4f}", help="Estimated cost for the entire session.")

    st.markdown("---")
    st.markdown(" ")

    # ------------------------------
    # Reasoning Steps Section
    # ------------------------------
    st.subheader("🧠 Agent's Reasoning Steps:")
    if not st.session_state.trace:
        st.caption("No tool usage in the last turn.")
    else:
        for step in st.session_state.trace:
            if step["type"] == "tool_call":
                with st.expander(f"🛠️ Calling Tool: `{step['tool']}`"):
                    st.markdown("**Tool Input:**")
                    st.code(json.dumps(step['tool_input'], indent=2), language="json")
            elif step["type"] == "tool_output":
                with st.expander(f"👀 Observation from `{step['tool']}`"):
                    st.markdown("**Summary of results (by Tavily):**")
                    obs = json.loads(step['observation'])
                    answ = str(obs['answer'])

                    if answ:
                        if len(answ) > 1000:
                            st.markdown(answ[:1000] + "...")
                        else:
                            st.markdown(answ)
                    else:
                        st.markdown("No answer summary available.")


    st.markdown("---")
    st.markdown(" ")

    # ------------------------------
    # Provenance Section
    # ------------------------------
    st.subheader("🗂️ Sources' Provenance:")
    
    # choose the provenance for the current thread or last message
    prov_key = st.session_state.conversation_thread_id or list(st.session_state.provenance.keys())[-1]
    items = st.session_state.provenance.get(prov_key, [])

    if not items:
        st.caption("No provenance available (no search performed).")
    else:    
        for item in items:
            sources = item.get("extracted_sources")
            # If we have a list of structured sources (from the new server)
            if isinstance(sources, list) and len(sources) > 0 and isinstance(sources[0], dict):
                with st.expander(f"🗂️ Sources for query: *{item.get('query', 'Unknown')}*"):
                    st.write(f"**Sources from tool `{item['tool']}`**:")
                    for s in sources:
                        title = s.get('title', 'No Title')
                        url = s.get('url', '#')
                        st.markdown(f"- [{title}]({url})")
            else:
                # Fallback to the old raw method
                st.json(item)

    st.markdown("---")

    st.markdown("[View Source Code](https://github.com/danielecelsa/chatbot-with-search) • Developed by **[Daniele Celsa](https://www.domenicodanielecelsa.com)**")

# ------------------------------
# Render chat
# ------------------------------
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
    else:
        content = getattr(msg, "content", None) or str(msg)
        with st.chat_message("assistant"):
            st.write(content)