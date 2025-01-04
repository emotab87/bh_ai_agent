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

    ###### STEP 1. State Definition ######
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    ###### STEP 2. Node Definition ######
    # Initialize LLM
    claudeModelName = claudeHaikuModelName()
    chatGPTModelName = modelName()

    if st.session_state.model_choice == "Anthropic Claude":
        llm = ChatAnthropic(model=claudeModelName, api_key=st.session_state.api_key)
    elif st.session_state.model_choice == "OpenAI ChatGPT":
        llm = ChatOpenAI(model=chatGPTModelName, api_key=st.session_state.api_key)

    # Initialize tools
    tools = [tool]  # Use the already validated Tavily tool

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Define chatbot node
    def chatbot(state: State):
        answer = llm_with_tools.invoke(state["messages"])
        return {"messages": [answer]}

    # Define tool node
    class BasicToolNode:
        """Run tools requested in the last AIMessage node"""

        def __init__(self, tools: list) -> None:
            self.tools_list = {tool.name: tool for tool in tools}

        def __call__(self, inputs: dict):
            if messages := inputs.get("messages", []):
                message = messages[-1]
            else:
                raise ValueError("No message found in input")

            outputs = []
            for tool_call in message.tool_calls:
                tool_result = self.tools_list[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result, ensure_ascii=False),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            return {"messages": outputs}

    # Create tool node
    tool_node = BasicToolNode(tools=tools)

    ###### STEP 3. Graph Definition ######
    # Create graph
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)

    # Add edges
    graph_builder.add_edge(START, "chatbot")

    # Define routing logic
    def route_tools(state: State):
        if messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError("No messages found in input state")

        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END

    # Add conditional edges
    graph_builder.add_conditional_edges(
        source="chatbot",
        path=route_tools,
        path_map={"tools": "tools", END: END},
    )

    # Add tool to chatbot edge
    graph_builder.add_edge("tools", "chatbot")

    # Compile graph
    graph = graph_builder.compile()

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

            # Store responses for final chat display
            responses = []

            # Show processing indicator
            with st.spinner("Processing your request..."):
                # Process conversation
                for event in graph.stream(
                    {"messages": full_conversation}, stream_mode="values"
                ):
                    for key, value in event.items():
                        # Update step counter
                        st.session_state.step_counter += 1

                        # Display debug information in debug tab
                        with debug_tab:
                            # Create columns for step organization
                            step_info, step_content = st.columns([1, 2])

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

                # After processing, update chat with final response
                if responses:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    final_response = responses[-1]

                    # Add response to session state
                    st.session_state.messages_01.append(
                        {
                            "role": "assistant",
                            "content": final_response,
                            "timestamp": current_time,
                        }
                    )

                    # Switch back to chat tab to show response
                    chat_tab.chat_message("assistant").markdown(final_response)
                    chat_tab.caption(f"Sent at {current_time}")


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
