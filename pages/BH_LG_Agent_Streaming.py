import streamlit as st
from typing import Annotated, List, Tuple, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_teddynote.tools import GoogleNews
from langchain.schema import AIMessage, HumanMessage
import logging
from dataclasses import dataclass
from functools import lru_cache
from langchain_core.runnables import RunnableConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration class
@dataclass
class ModelConfig:
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20240620"
    GPT_MODEL: str = "gpt-4o-mini"
    MAX_SEARCH_RESULTS: int = 3


model_config = ModelConfig()

# Memory storage initialization - no arguments needed
memory = MemorySaver()

# Configure runnable settings with configurable parameters
config = RunnableConfig(
    recursion_limit=10, configurable={"thread_id": "1"}, tags=["BH-chatbot-tag"]
)


# State definition with improved type hints
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    dummy_data: Annotated[str, "dummy_data"]


@tools
def search_keyword(query: str) -> List[Dict[str, str]]:
    """
    Search for a keyword in the Google News.
    """
    news_tool = GoogleNews()
    return news_tool.search_by_keyword(query, k=5)


@lru_cache()
def get_llm(model_choice: str, api_key: str) -> ChatAnthropic | ChatOpenAI:
    """
    Get LLM instance based on model choice with caching.
    """
    try:
        if model_choice == "Anthropic Claude":
            return ChatAnthropic(model=model_config.CLAUDE_MODEL, api_key=api_key)
        elif model_choice == "OpenAI ChatGPT":
            return ChatOpenAI(model=model_config.GPT_MODEL, api_key=api_key)
        else:
            raise ValueError(f"Unsupported model choice: {model_choice}")
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise


def initialize_graph(llm) -> StateGraph:
    """
    Initialize and return the LangGraph with error handling.
    """
    try:
        # Tool initialization
        tools = [search_keyword]

        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)

        # Chatbot function with error handling
        def chatbot(state: ChatState):
            try:
                response = llm_with_tools.invoke(state["messages"], config=config)
                return {
                    "messages": [response],
                    "dummy_data": "[chatbot] 호출, dummy data",  # for the testing, added dummy_data
                }
            except Exception as e:
                logger.error(f"Error in chatbot: {str(e)}")
                return {
                    "messages": [
                        AIMessage(
                            content="I encountered an error processing your request. Please try again."
                        )
                    ]
                }

        # Graph construction
        graph_builder = StateGraph(ChatState)
        graph_builder.add_node("chatbot", chatbot)

        # Configure tool node
        tool_node = ToolNode(tools=[tool])
        graph_builder.add_node("tools", tool_node)

        # Edge configuration
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)

        return graph_builder.compile(checkpointer=memory)
    except Exception as e:
        logger.error(f"Error initializing graph: {str(e)}")
        raise


def handle_user_input(graph, messages: List[dict], prompt: str):
    """
    Handle user input and generate response.
    """
    try:
        messages.append({"role": "user", "content": prompt})
        input = ChatState(messages=messages, dummy_data="test string")
        with st.chat_message("user"):
            st.markdown(prompt)

        full_conversation = [(msg["role"], msg["content"]) for msg in messages]

        for event in graph.stream(input=input, stream_mode="updates", config=config):
            # Get the node name from the event key
            for node_name, value in event.items():
                response = value["messages"][-1].content
                messages.append({"role": node_name, "content": response})
                with st.chat_message(node_name):
                    st.markdown(response)
    except Exception as e:
        logger.error(f"Error processing user input: {str(e)}")
        st.error("An error occurred while processing your request. Please try again.")


def display_chat_history(messages: List[dict]):
    """
    Display chat history with proper formatting.
    """
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_graph_visualization(graph):
    """
    Render graph visualization with error handling.
    """
    try:
        graph_bytes = graph.get_graph().draw_mermaid_png()
        st.image(graph_bytes, caption="Chatbot Graph")
    except Exception as e:
        logger.error(f"Failed to display graph: {str(e)}")
        st.warning("Graph visualization is currently unavailable.")


def main():
    st.title("LangGraph Chatbot Super Basic")

    # Sidebar reset button
    with st.sidebar:
        if st.button("Reset Session"):
            st.session_state.clear()

    # Model selection
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "OpenAI ChatGPT"

    model_choice = st.radio("Choose your model", ["OpenAI ChatGPT", "Anthropic Claude"])

    # Handle model change
    if model_choice != st.session_state.model_choice:
        st.session_state.clear()
        st.session_state.model_choice = model_choice

    # API key handling
    if "api_key_submitted" not in st.session_state:
        st.session_state.api_key_submitted = False

    if not st.session_state.api_key_submitted:
        llm_api_key = st.text_input(
            f"Please input your {model_choice} API Key:", type="password"
        )

        if st.button("Submit"):
            if llm_api_key:
                st.session_state.api_key = llm_api_key
                st.session_state.api_key_submitted = True
            else:
                st.warning("Please input your API key.")

    # Initialize chat session
    if st.session_state.get("api_key_submitted"):
        try:
            llm = get_llm(st.session_state.model_choice, st.session_state.api_key)
            graph = initialize_graph(llm)

            if "messages_01" not in st.session_state:
                st.session_state.messages_01 = []

            # Display chat interface
            display_chat_history(st.session_state.messages_01)
            render_graph_visualization(graph)

            if prompt := st.chat_input("What is up?"):
                handle_user_input(graph, st.session_state.messages_01, prompt)

        except Exception as e:
            logger.error(f"Error in main chat interface: {str(e)}")
            st.error("An error occurred. Please check your API key and try again.")


if __name__ == "__main__":
    main()
