from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv
import streamlit as st
import os, asyncio


# Connect to MCP server 
# Ensure you have the MCP server running before starting this client
mcp_config = {
    "tavily": {
        "url": "https://tavily-mcp-server-wqe5.onrender.com/tav/mcp/",  # Adjust the URL if your server is hosted elsewhere
        "transport": "streamable_http",  # Use "streamable-http" for streaming responses
    }
}

client = MultiServerMCPClient(mcp_config)
tools = asyncio.run(client.get_tools())  # load tool schema

#async with client.session("tavily") as session:
#    tools = await load_mcp_tools(session)


load_dotenv()

# === AGENT ===
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

system = SystemMessagePromptTemplate.from_template(f"""
        Use the web search tool to find information if needed, otherwise answer directly.
        Answer the user's questions based on the below chat history:\n\n"
        """
    )
hist = MessagesPlaceholder(variable_name="chat_history")
human = HumanMessagePromptTemplate.from_template("{input}\n{agent_scratchpad}")
prompt = ChatPromptTemplate.from_messages([system, hist, human])

raw_agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=raw_agent,
    tools=tools,
    verbose=False,
    return_intermediate_steps=True,
    handle_parsing_errors=True  # utile se vuoi evitare crash su output malformati
)

def get_response(user_query):
    #response = agent_executor.invoke({
    #    "chat_history": st.session_state.chat_history,
    #    "input": user_query
    #})
    response = asyncio.run(agent_executor.ainvoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    }))

    print(response["intermediate_steps"])  # Debugging: print the response to see its structure
    
    return response['output']


### FRONTEND ###

# app config
st.set_page_config(page_title="Chat w search", page_icon="🤖")
st.title("Chat with Search")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]
  
# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

