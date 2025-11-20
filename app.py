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
    
    system = SystemMessagePromptTemplate.from_template(
        """You are a helpful and precise chatbot assistant.
        You have access to a web search tool named `web_search` provided through MCP/Tavily.

        ### TOOL USAGE INSTRUCTIONS
        The `web_search` tool returns a JSON object containing:
        1. "answer": A preliminary summary generated by the search engine.
        2. "results": A list of raw sources with URLs and content excerpts.

        **How to process search results:**
        - Read the "answer" to quickly grasp the context and the direct answer.
        - **CRITICAL:** Always verify the information against the "results" list. 
        - Use the "results" to ensure accuracy and to find specific details that might be missing from the summary.
        - If the user asks for sources or links, look them up in the "results" section.

        ### OPERATIONAL RULES
        - **Be concise.** Answer the user's question directly.
        - **Transparency:** Do not reveal your internal chain-of-thought or JSON parsing steps. Present the answer naturally.
        - **Citations:** When stating facts retrieved from the web, you may reference the source context if relevant, but do not dump raw URLs unless explicitly asked.
        - **Ambiguity:** If a question is impossible to answer without clarification (e.g., "What happened in the summer?" without a year or place), ask ONE clarifying question.
        - **Multi-step reasoning:** If the user asks a compound question (e.g., "When did X happen and what else happened that year?"), perform the necessary searches to answer ALL parts of the question. Do NOT ask for permission to search again.
        - If you cannot find an answer after searching, admit it honestly: "I could not find information about that."
        - Today's date is {current_date}.
        """
    ).format(current_date=datetime.now().date().isoformat())

    hist = MessagesPlaceholder(variable_name="messages")
    prompt = ChatPromptTemplate.from_messages([system, hist])
    
    return prompt

print(f"Today is {datetime.now().date().isoformat()}")
# ------------------------------
# Build agent with checkpointer
# ------------------------------
# @st.cache_resource
def build_agent():
    """Build the ReAct agent with tools and checkpointer."""
    # MCP tools
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
st.set_page_config(page_title="AI Research Agent (LangGraph + MCP)", page_icon="🤖", layout="wide")
st.title("🤖 AI Research Agent: LangGraph + MCP")

body = """
### Architecture Overview
This application demonstrates a decoupled **Agentic System** designed for autonomous web research.
It connects a **ReAct Agent** (Client) to a remote **Tavily Search Tool** via the **Model Context Protocol (MCP)**.

**Key Engineering Features:**
- **🧠 Cognitive Architecture:** Built with **LangGraph** for state management and cyclic reasoning (ReAct loop).
- **🔌 MCP Standard:** Implements a **Multi-Server MCP Client** over SSE (Server-Sent Events), decoupling the LLM logic from the tool execution environment.
- **🛡️ Safe Grounding:** Answers are generated using a "hybrid summary" approach (Search Engine Summary + Raw Source Verification) to minimize hallucinations.
- **💾 Persistence:** Uses an async SQLite checkpointer to maintain conversation state across turns.
"""

with st.expander("ℹ️ **Project Architecture & Goals** (Read me)", expanded=False):
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

            logger_local.info("Collected events from agent: %s", events)

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
    st.header("👨‍💻 Engineering Highlights", divider="rainbow")
    
    st.info(
        "This demo is designed to showcase production-ready capabilities in GenAI development."
    )

    with st.expander("🛠️ Tech Stack & Implementation"):
        st.markdown("""
        **Core Orchestration:**
        - `LangGraph`: For cyclic agentic flows.
        - `LangChain Core`: For prompt & message handling.
        
        **Connectivity & Tools:**
        - **Protocol:** [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
        - **Transport:** SSE (Server-Sent Events) for async tool streams.
        - **Search:** Tavily API (via custom FastMCP Server).
        
        **Backend & Observability:**
        - **AsyncIO:** Custom background loop management for non-blocking Streamlit execution.
        - **Logging Pipeline:** Multi-sink logging (Local JSONL + **Redis** + **BetterStack**) for full traceability.
        - **State:** Async SQLite Checkpointer.
        """)

    with st.expander("🧪 How to Test (Scenarios)"):
        st.markdown("""
        The agent decides *when* to search. Try these to trigger the ReAct loop:
        
        1. **Multi-step Reasoning:**
           *"When was the last time a total solar eclipse was visible from New York City? Once you find it, tell me also what other important events happened that year"*
            
            (Checks reasoning across multiple facts)
           
        2. **Recent Events (Knowledge Cutoff):**
           *"What are the latest updates on SpaceX Starship from last week?"*
           
            (Forces web search usage)
           
        3. **Ambiguity Handling:**
           *"What happened in the summer?"*
           
           (Should ask clarifying questions)
        """)
    
    st.markdown("---")

    # Metrics Section
    st.subheader("📊 Live Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Latency", value=f"{st.session_state.latency:.2f}s")
    with col2:
        st.metric(label="Session Cost", value=f"${st.session_state.usd:.4f}")

    st.caption(f"Token Usage: {st.session_state.total_tokens} total tokens used.")
    
    st.markdown("---")
    st.markdown(" ")

    with st.expander("Token Usage & Latency:"):
        st.metric(label=":blue[Latency (s)]", value=f"{st.session_state.latency:.2f}", help="Time for the last response.")
        
        st.markdown("#### :blue[Last Interaction Token]")
        col1, col2 = st.columns(2)
        col1.metric("Input Tokens", f"{st.session_state.input_tokens_last}")
        col2.metric("Output Tokens", f"{st.session_state.output_tokens_last}")
        st.metric("Last Est. Cost (USD)", f"${st.session_state.usd_last:.5f}", help="Estimated cost for the last interaction.")

        st.markdown("#### :blue[Session Total Token]")
        col3, col4 = st.columns(2)
        col3.metric("Total Input", f"{st.session_state.total_input_tokens}")
        col4.metric("Total Output", f"{st.session_state.total_output_tokens}")
        st.metric("Total Est. Cost (USD)", f"${st.session_state.usd:.5f}", help="Estimated cost for the entire session.")

    st.markdown("---")
    st.markdown(" ")
# ------------------------------
# Safe intermediate steps (trace) - global section visible outside submit
# ------------------------------
    st.markdown("Agent's Reasoning Steps:")
    if not st.session_state.trace:
        st.caption("No tool usage in the last turn. Reasoning steps are shown only when tools are used.")
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
# Provenance (separate section, visible on demand)
# ------------------------------
    with st.expander("Sources' Provenance:"):
        if not st.session_state.provenance:
            st.caption("No provenance available.")
        else:
            # choose the provenance for the current thread or last message
            prov_key = st.session_state.conversation_thread_id or list(st.session_state.provenance.keys())[-1]
            items = st.session_state.provenance.get(prov_key, [])
            
            for item in items:
                sources = item.get("extracted_sources")
                # If we have a list of structured sources (from the new server)
                if isinstance(sources, list) and len(sources) > 0 and isinstance(sources[0], dict):
                    st.write(f"**Source from tool `{item['tool']}`**, for query: *{item.get('query', 'Unknown')}*:")
                    for s in sources:
                        title = s.get('title', 'No Title')
                        url = s.get('url', '#')
                        st.markdown(f"- [{title}]({url})")
                else:
                    # Fallback to the old raw method
                    st.json(item)

    st.markdown("---")
    st.markdown("Developed by **[Daniele Celsa](https://www.domenicodanielecelsa.com)**")
    st.markdown("📂 [View Source on GitHub](https://github.com/danielecelsa/chatbot-with-search)")

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

