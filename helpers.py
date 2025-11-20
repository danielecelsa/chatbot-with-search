import json
from typing import Any, Dict, List, Tuple, Optional
from langchain_core.messages import AIMessage, ToolMessage
import logging
from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner import get_script_run_ctx

# --- Core Event Parser ---
def process_agent_events(events: List[Dict]) -> Tuple[Optional[AIMessage], List[Dict], List[Dict], Dict]:
    """
    Parses the event stream from create_react_agent to extract a trace,
    provenance, the final answer, and ACCURATE token usage for the last interaction.
    """
    trace = []
    provenance = []
    final_answer = None
    last_interaction_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    for event_chunk in events:
        if "agent" in event_chunk:
            agent_messages = event_chunk["agent"].get("messages", [])
            for msg in agent_messages:
                if not isinstance(msg, AIMessage):
                    continue
                if msg.usage_metadata:
                    last_interaction_usage["input_tokens"] += msg.usage_metadata.get("input_tokens", 0)
                    last_interaction_usage["output_tokens"] += msg.usage_metadata.get("output_tokens", 0)
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        trace.append({
                            "type": "tool_call", 
                            "tool": tool_call.get("name"),
                            "tool_input": tool_call.get("args"),
                        })
                elif msg.content:
                    final_answer = msg
        elif "tools" in event_chunk:
            tool_messages = event_chunk["tools"].get("messages", [])
            for msg in tool_messages:
                if not isinstance(msg, ToolMessage):
                    continue

                # Try to parse the JSON to extract sources cleanly
                parsed_provenance = []
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, dict) and "results" in data:
                        # If it's the format of our new server
                        parsed_provenance = data["results"]
                    else:
                        # Fallback
                        parsed_provenance = data
                    if isinstance(data, dict) and "query" in data:
                        query = data["query"]
                    else:
                        query = None

                except:
                    # If it's not JSON, use the raw text
                    parsed_provenance = msg.content
                    query = None

                trace.append({
                    "type": "tool_output", 
                    "tool": msg.name, 
                    "observation": msg.content,
                })

                provenance.append({
                    "tool": msg.name,
                    "extracted_sources": parsed_provenance,
                    "raw_content": msg.content,
                    "query": query
                    })
    
    last_interaction_usage["total_tokens"] = last_interaction_usage["input_tokens"] + last_interaction_usage["output_tokens"]
    return final_answer, trace, provenance, last_interaction_usage

def compute_cost(input_tokens: int, output_tokens: int, cost_per_1k_input: float, cost_per_1k_output: float) -> float:
    """Computes the cost of a given number of input and output tokens."""
    input_cost = (input_tokens / 1000.0) * cost_per_1k_input
    output_cost = (output_tokens / 1000.0) * cost_per_1k_output
    return input_cost + output_cost

def get_user_info(logger: logging.Logger) -> dict:
    """
    Tries to get the user's IP address and User-Agent from Streamlit's internal API.
    This is an undocumented feature and is known to change between Streamlit versions.
    Returns a dictionary with 'ip' and 'user_agent'.
    """
    user_info = {"ip": "Unknown", "user_agent": "Unknown"}
    try:
        # Get the context for the current script run
        ctx = get_script_run_ctx()
        if ctx is None:
            logger.warning("Could not get Streamlit script run context.")
            return user_info

        # Get the unique session ID from the context
        session_id = ctx.session_id

        # Get the Streamlit runtime instance
        runtime = get_instance()
        
        # From the runtime, get the session manager, and then the specific session info
        session_info = runtime._session_mgr.get_session_info(session_id)

        if session_info is None:
            logger.warning("Could not find session info for the current session ID.")
            return user_info
        
        # The request headers are located in the 'client' attribute of the session info
        headers = session_info.client.request.headers

        # Logic for extracting IP and User-Agent remains the same
        if 'X-Forwarded-For' in headers:
            user_info['ip'] = headers['X-Forwarded-For'].split(',')[0].strip()
        elif hasattr(session_info.client.request, 'remote_ip'):
            user_info['ip'] = session_info.client.request.remote_ip

        if 'User-Agent' in headers:
            user_info['user_agent'] = headers['User-Agent']

    except Exception as e:
        logger.warning(f"Could not get user info from Streamlit headers: {e}")

    return user_info
