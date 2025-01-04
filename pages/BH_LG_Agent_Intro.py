import streamlit as st
from langchain_teddynote.tools.tavily import TavilySearch
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import json
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import base64
import io
from datetime import datetime


def claudeHaikuModelName():
    return "claude-3-5-sonnet-20240620"


def modelName():
    return "gpt-4o-mini"


def LangGraph_run():
    # First, validate the Tavily API key
    try:
        tool = TavilySearch(api_key=st.session_state.tavily_api_key, max_results=3)
        tool.invoke("test")
    except Exception as e:
        st.error("Invalid Tavily API key. Please check your API key and try again.")
        st.error(f"Error: {str(e)}")
        return

    # Create tabs for different views
    chat_tab, graph_tab, debug_tab = st.tabs(["💬 Chat", "📊 Graph", "🔍 Debug"])

    with graph_tab:
        st.subheader("LangGraph Visualization")
        try:
            graph_bytes = graph.get_graph().draw_mermaid_png()
            st.image(graph_bytes, caption="Chatbot Graph", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to display graph: {e}")

    with chat_tab:
        st.subheader("Chat Interface")

        # Initialize messages if not exists
        if "messages_01" not in st.session_state:
            st.session_state.messages_01 = []

        # Display chat history with timestamps
        for message in st.session_state.messages_01:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timestamp" in message:
                    st.caption(f"Sent at {message['timestamp']}")

        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            # Add timestamp to message
            current_time = datetime.now().strftime("%H:%M:%S")

            # Add user message
            st.session_state.messages_01.append(
                {"role": "user", "content": prompt, "timestamp": current_time}
            )
            with st.chat_message("user"):
                st.markdown(prompt)
                st.caption(f"Sent at {current_time}")

            # Prepare conversation history
            full_conversation = [
                (msg["role"], msg["content"]) for msg in st.session_state.messages_01
            ]

            with debug_tab:
                st.subheader("Execution Steps")
                # Create columns for step organization
                step_info, step_content = st.columns([1, 2])

                # Initialize step counter
                if "step_counter" not in st.session_state:
                    st.session_state.step_counter = 0

                # Store responses for final chat display
                responses = []

                # Show processing indicator
                with st.spinner("Processing your request..."):
                    # Process conversation
                    for event in graph.stream(
                        {"messages": full_conversation}, stream_mode="values"
                    ):
                        for key, value in event.items():
                            st.session_state.step_counter += 1

                            # Step information column
                            with step_info:
                                st.markdown(f"### Step {st.session_state.step_counter}")
                                st.markdown(f"**Type:** {key}")
                                st.markdown(
                                    f"**Time:** {datetime.now().strftime('%H:%M:%S')}"
                                )

                            # Step content column
                            with step_content:
                                # Display message content
                                if (
                                    isinstance(value, dict)
                                    and "messages" in value
                                    and value["messages"]
                                ):
                                    message = value["messages"][-1]

                                    # Display content in a nice format
                                    st.markdown("**Content:**")
                                    st.code(message.content, language="markdown")

                                    # Store assistant messages
                                    if key == "chatbot":
                                        responses.append(message.content)

                                    # Display tool calls if any
                                    if (
                                        hasattr(message, "tool_calls")
                                        and message.tool_calls
                                    ):
                                        st.markdown("**Tool Calls:**")
                                        for tool_call in message.tool_calls:
                                            with st.expander(
                                                f"Tool: {tool_call['name']}",
                                                expanded=False,
                                            ):
                                                st.code(
                                                    tool_call["args"], language="json"
                                                )
                                else:
                                    st.code(str(value), language="python")

                                st.divider()

                # Display final response in chat
                if responses:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    final_response = responses[-1]
                    st.session_state.messages_01.append(
                        {
                            "role": "assistant",
                            "content": final_response,
                            "timestamp": current_time,
                        }
                    )

                    with chat_tab:
                        with st.chat_message("assistant"):
                            st.markdown(final_response)
                            st.caption(f"Sent at {current_time}")


def collect_api_keys():
    """Collect and validate all required API keys."""
    if "api_key_submitted" not in st.session_state:
        st.session_state.api_key_submitted = False

    if not st.session_state.api_key_submitted:
        col1, col2 = st.columns(2)

        with col1:
            if st.session_state.model_choice == "Anthropic Claude":
                llm_api_key = st.text_input(
                    "Anthropic API Key:", type="password", key="llm_key"
                )
            else:
                llm_api_key = st.text_input(
                    "OpenAI API Key:", type="password", key="llm_key"
                )

        with col2:
            tavily_api_key = st.text_input(
                "Tavily Search API Key:",
                type="password",
                key="tavily_key",
                help="Get your Tavily API key from https://tavily.com",
            )

        # Add help text for API keys
        st.markdown(
            """
        > **Note**: 
        > - For OpenAI key, visit: https://platform.openai.com/api-keys
        > - For Anthropic key, visit: https://console.anthropic.com/
        > - For Tavily key, visit: https://tavily.com
        """
        )

        if st.button("Submit API Keys"):
            if llm_api_key and tavily_api_key:
                # Basic validation of API key formats
                if len(tavily_api_key) < 20:  # Tavily keys are typically longer
                    st.warning(
                        "The Tavily API key seems too short. Please check if it's correct."
                    )
                    return False

                st.session_state.api_key = llm_api_key
                st.session_state.tavily_api_key = tavily_api_key
                st.session_state.api_key_submitted = True
                return True
            else:
                if not llm_api_key:
                    st.warning("Please input your LLM API Key.")
                if not tavily_api_key:
                    st.warning("Please input your Tavily API Key.")
                return False
    return st.session_state.api_key_submitted


def main():
    # Sidebar to reset session state
    with st.sidebar:
        if st.button("Reset Session"):
            st.session_state.clear()
            # st.experimental_rerun()  # Rerun the app to reflect the changes

    # Page title
    st.title("LangGraph Chatbot Super Basic")

    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "OpenAI ChatGPT"  # 기본 선택값 설정

    # 모델 선택
    model_choice = st.radio("Choose your model", ["OpenAI ChatGPT", "Anthropic Claude"])

    # 선택이 변경되었는지 확인하고, 변경되었다면 세션을 초기화
    if model_choice != st.session_state.model_choice:
        st.session_state.clear()  # 현재 세션 정보 모두 초기화
        st.session_state.model_choice = model_choice  # 새로운 선택값을 세션에 저장

    # Collect API keys using the new function
    if collect_api_keys():
        LangGraph_run()


if __name__ == "__main__":
    main()
