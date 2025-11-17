# client_react_langgraph.py
# Streamlit client using LangGraph prebuilt ReAct agent + MCP Tavily tool.
# Key points:
# - Uses a LangGraph checkpointer for persistence (Async SQLite if available, otherwise in-memory).
# - Keeps features: conversation memory, safe intermediate-step trace (tools & observations), separate Provenance section, token usage & cost, JSONL logging.

import os
import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid
import redis
import valkey
import logging
from logtail import LogtailHandler
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


# Helpers (token estimation, cost computation, safe serialization)
from helpers import (
    process_agent_events,
    compute_cost,
    get_user_info,
    append_log
)

from async_bg import collect_events_from_agent


if os.getenv("RENDER") != "true":
    load_dotenv()

# ------------------------------
# Configuration
# ------------------------------
MODEL = os.environ.get("GENAI_MODEL", "gemini-2.0-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

COST_PER_1K_INPUT = float(os.getenv("COST_PER_1K_TOKENS_USD_INPUT", "0.002"))
COST_PER_1K_OUTPUT = float(os.getenv("COST_PER_1K_TOKENS_USD_OUTPUT", "0.002"))

LOG_DIR = Path(os.environ.get("CHAT_LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

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

# ------------------------------
# Key-Value Store (Valkey/Redis) Setup
# ------------------------------
@st.cache_resource
def get_kv_client():
    """
    Gets a singleton Valkey/Redis client connection.
    Uses REDIS_URL from environment variables for production (Render).
    Falls back to a local connection for development.
    """
    # Render provides the connection string for its Valkey service in the REDIS_URL env var
    redis_url = os.environ.get("REDIS_URL", None)

    try:
        if redis_url:
            # Production environment (Render) - use the provided URL
            logger.info("Connecting to Key-Value store via REDIS_URL.")
            kv_client = valkey.from_url(redis_url, decode_responses=True)
        else:
            # Local development - connect to your local Redis/Valkey instance
            logger.info("REDIS_URL not found. Connecting to localhost.")
            kv_client = valkey.Redis(host="localhost", port=6379, db=0, decode_responses=True)

        # Ping the server to check if the connection is alive
        kv_client.ping()
        logger.info("Successfully connected to Key-Value store.")
        return kv_client
        
    except valkey.exceptions.ConnectionError as e:
        logger.warning(f"Could not connect to Key-Value store: {e}. Using in-memory queue as fallback.")
        return None

# --- Custom Redis Logging Handler ---
class RedisLogHandler(logging.Handler):
    """
    A logging handler that publishes records to a capped Redis list.
    """
    def __init__(self, client, key, max_entries=500):
        super().__init__()
        self.client = client
        self.key = key
        self.max_entries = max_entries

    def emit(self, record):
        """
        Takes a log record, formats it, and pushes it to Redis.
        """
        try:
            # Format the log record into a string
            log_entry = self.format(record)
            # Push the entry to the left of the list
            self.client.lpush(self.key, log_entry)
            # Trim the list to keep only the latest max_entries
            self.client.ltrim(self.key, 0, self.max_entries - 1)
        except Exception:
            # If Redis fails, we don't want the logger to crash the app
            pass

# Define the name for our Redis list for logs
REDIS_LOGS_KEY = "tavily_logs"

# Get the top-level logger
logger = logging.getLogger("tool_logger")
logger.setLevel(logging.INFO)

# Prevent log messages from propagating to the root logger
logger.propagate = False

# Remove any existing handlers to avoid duplicates
if logger.hasHandlers():
    logger.handlers.clear()

# --- Handler 1: Always write to a local file (great for local debugging) ---
file_handler = logging.FileHandler("logs/tavily_debug.log", mode="a")
file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Get the singleton instances
kv_client = get_kv_client() # Valkey client (Redis-compatible)

# --- Handler 2: Add the Logtail handler IF the token is available (for production) ---
logtail_token = os.environ.get("LOGTAIL_SOURCE_TOKEN")
logtail_urL = os.environ.get("LOGTAIL_URL")
if logtail_token:
    try:
        logtail_handler = LogtailHandler(
            source_token=logtail_token, 
            host=logtail_urL,
        )
        logtail_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        logtail_handler.setFormatter(logtail_formatter)
        logger.addHandler(logtail_handler)
        logger.info("Logtail persistent logging handler successfully configured.")
    except Exception as e:
        logger.warning(f"Failed to configure Logtail logging handler: {e}")
else:
    # --- Fallback Handler: Use Valkey/Redis for logs ONLY if Logtail isn't configured ---
    if kv_client:
        try:
            # We keep RedisLogHandler class definition, but only use it as a fallback
            redis_handler = RedisLogHandler(client=kv_client, key=REDIS_LOGS_KEY)
            redis_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            redis_handler.setFormatter(redis_formatter)
            logger.addHandler(redis_handler)
            logger.info("Logtail token not found. Key-Value Valkey/Redis logging handler successfully configured..")
        except Exception as e:
            logger.warning(f"Failed to configure Valkey/Redis fallback logging handler: {e}")


# --------------------------------------
# Streamlit session state initialization
# --------------------------------------
if "conversation_thread_id" not in st.session_state:
    st.session_state.conversation_thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a chatbot. How can I help you?")]  # list of Messages
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
        logger.exception("Could not initialize LLM: %s", e)
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
        """You are a helpful chatbot assistant.
        You have access to a web search tool named `web_search` provided through MCP/Tavily.

        Operational rules:
        - Be concise and accurate. Provide the direct answer to the user's question.
        - Use the `web_search` tool whenever up-to-date facts, dates, or sourceable evidence is required.
        - Do not reveal internal chain-of-thought. You may present the steps you took as a short list of actions and tools used.
        - If a question is REALLY ambiguous and cannot be answered without clarification, ask ONE clarifying question. Otherwise, prefer to answer the best you can.
        For example: if the user asks "When did X happen? And what else happened that year?", you can 
        ask "Did you mean X event in Y country?" if this is not clear, but you do NOT need to ask confirmation to perform a second search - he already said he wants to 
        know what else happened that year, so if you need further searches to answer just do them.
        - If you cannot find an answer, say "I could not find anything about that".
        """
    )

    hist = MessagesPlaceholder(variable_name="messages")
    prompt = ChatPromptTemplate.from_messages([system, hist])
    
    return prompt


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
        print("Could not initialize MCP client/tools:", e)

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
st.set_page_config(page_title="Agentic Web Researcher", page_icon="🤖", layout="wide")
st.title("Agentic Web Researcher — Chat with web research")
body="""
## Chatbot with LangGraph ReAct agent + MCP/Tavily web search tool
This demo showcases a chatbot powered by a LangGraph ReAct agent connected to an MCP server with Tavily web search capabilities.

### Goal of the Demo:
The idea is to demonstrate how an agent can autonomously use web search to answer user queries that require up-to-date information
- The agent uses a ReAct architecture to reason about when to invoke the `web_search` tool.
- The agent maintains a conversation history and can provide provenance for its answers by showing the intermediate steps and tool outputs.
- Token usage and estimated cost are tracked and displayed.
- Using Streamlit for an interactive web interface and Render to host the app.
"""
with st.expander('About this demo:', expanded=False):
    st.markdown(body)

# ------------------------------
# Chat submission
# ------------------------------
user_query = st.chat_input("Type your message here...")
if user_query:
    
    # Get user info as soon as they submit a query
    user_details = get_user_info(logger)
    logger.info(
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
    
    st.session_state.chat_history.append(AIMessage(content=ai_content))

    # Calculate costs for UI display
    try:
        st.session_state.usd = compute_cost(st.session_state.total_input_tokens, st.session_state.total_output_tokens, COST_PER_1K_INPUT, COST_PER_1K_OUTPUT)
    except Exception:
        logger.exception("Error computing total cost: %s", e)
        st.session_state.usd = 0.0
    
    try:
        st.session_state.usd_last = compute_cost(st.session_state.input_tokens_last, st.session_state.output_tokens_last, COST_PER_1K_INPUT, COST_PER_1K_OUTPUT)
    except Exception:
        logger.exception("Error computing last cost: %s", e)
        st.session_state.usd_last = 0.0

    # Append a structured log
    try:
        append_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "messages": [{"role": m.__class__.__name__, "content": m.content} for m in st.session_state.chat_history],
            "trace": st.session_state.trace,
            "provenance": st.session_state.provenance.get(thread_id, []),
            "usage_last": {
                "input": st.session_state.input_tokens_last,
                "output": st.session_state.output_tokens_last,
            },
        }, LOG_DIR)
    except Exception as e:
        print("Failed to append log:", e)

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("Project :green[info]:", divider="rainbow")
    st.markdown(" ")
    st.markdown(" ")
    with st.expander("Tech stack & How to use:"):
        with st.expander("This demo features:"):
            st.markdown("""
            - **LangGraph**: for building the ReAct agent with checkpointer support.
            - **LangChain MCP Adapters**: to connect to the Tavily web search tool via MCP.
            - **Google Gemini**: as the underlying LLM (replaceable with any LangChain-compatible LLM).
            - **Streamlit**: for the web interface.
            - **Render**: for hosting the app.
            
            **How to use**:
            - Type your question in the chat input box.
            - The agent will decide when to use web search to find up-to-date information.
            - View intermediate steps, provenance, and token usage in the respective sections.
            """)
        with st.expander(":blue[How to test it?]"):
            st.markdown("""
                        Type your question in the chat input box.
                        The agent will decide when to use web search to find up-to-date information.
                        View intermediate steps, provenance, and token usage in the respective sections.""")
            st.markdown("Try with questions such as:")
            st.markdown("- *When was the last time a total solar eclipse was visible from New York City? Once you find it, tell me also what other important events happened that year*")
            st.markdown("- *Who won the Best Picture Oscar in 2023? Can you list the other nominees as well?*")
            st.markdown("- *What are the latest advancements in AI research as of 2024?*")
            st.markdown("- *Can you find recent news about space exploration missions?*")
    
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
    with st.expander("Agent's Reasoning Steps:"):
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
                        obs = str(step['observation'])
                        if len(obs) > 1000:
                            st.markdown(obs[:1000] + "...")
                        else:
                            st.markdown(obs)

    st.markdown("---")
    st.markdown(" ")

# ------------------------------
# Provenance (separate section, visible on demand)
# ------------------------------
    with st.expander("📚 Provenance (tool outputs)"):
        if not st.session_state.provenance:
            st.caption("No provenance available.")
        else:
            # choose the provenance for the current thread or last message
            prov_key = st.session_state.conversation_thread_id or list(st.session_state.provenance.keys())[-1]
            with st.expander("Show provenance for the current thread"):
                st.json(st.session_state.provenance.get(prov_key, []))

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

