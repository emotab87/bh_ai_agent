import streamlit as st
from typing import Annotated, List, Tuple, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_teddynote.tools.tavily import TavilySearch
from langchain.schema import AIMessage, HumanMessage
import logging
from dataclasses import dataclass
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration class
@dataclass
class ModelConfig:
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20240620"
    GPT_MODEL: str = "gpt-4o-mini"
    MAX_SEARCH_RESULTS: int = 3


config = ModelConfig()

# Memory storage initialization
memory = MemorySaver()


# State definition with improved type hints
class ChatState(TypedDict):
    messages: Annotated[List[Tuple[str, str]], add_messages]


@lru_cache()
def get_llm(model_choice: str, api_key: str) -> ChatAnthropic | ChatOpenAI:
    """
    Get LLM instance based on model choice with caching.
    """
    try:
        if model_choice == "Anthropic Claude":
            return ChatAnthropic(model=config.CLAUDE_MODEL, api_key=api_key)
        elif model_choice == "OpenAI ChatGPT":
            return ChatOpenAI(model=config.GPT_MODEL, api_key=api_key)
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
        tool = TavilySearch(max_results=config.MAX_SEARCH_RESULTS)
        tools = [tool]

        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)

        # Chatbot function with error handling
        def chatbot(state: ChatState):
            try:
                response = llm_with_tools.invoke(state["messages"])
                return {"messages": [response]}
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


def display_chat_history(messages: List[dict]):
    """
    Display chat history with proper formatting.
    """
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input(graph, messages: List[dict], prompt: str):
    """
    Handle user input and generate response.
    """
    try:
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        full_conversation = [(msg["role"], msg["content"]) for msg in messages]

        for event in graph.stream({"messages": full_conversation}):
            for value in event.values():
                response = value["messages"][-1].content
                messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
    except Exception as e:
        logger.error(f"Error processing user input: {str(e)}")
        st.error("An error occurred while processing your request. Please try again.")


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
        tavily_api_key = st.text_input(
            "Please input your Tavily API Key:", type="password"
        )

        if st.button("Submit"):
            if llm_api_key and tavily_api_key:
                st.session_state.api_key = llm_api_key
                st.session_state.tavily_api_key = tavily_api_key
                st.session_state.api_key_submitted = True
            else:
                st.warning("Please input both API keys.")

    # Initialize chat session
    if st.session_state.get("api_key_submitted"):
        try:
            llm = get_llm(st.session_state.model_choice, st.session_state.api_key)
            # Set Tavily API key for the tool
            import os

            os.environ["TAVILY_API_KEY"] = st.session_state.tavily_api_key
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
