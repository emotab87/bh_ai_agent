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


def claudeHaikuModelName():
    return "claude-3-5-sonnet-20240620"


def modelName():
    return "gpt-4o-mini"


def LangGraph_run():

    ###### STEP 1. 상태 (State) 정의 ######
    class State(TypedDict):
        # 메시지 정의(list type이며 add_messages 함수를 사용하여 메시지를 추가)
        messages: Annotated[list, add_messages]

    ###### STEP 2. 노드(Node) 정의 ######
    # LLM 정의
    # session_state에 저장된 model_choice를 사용합니다.
    claudeModelName = claudeHaikuModelName()
    chatGPTModelName = modelName()

    if st.session_state.model_choice == "Anthropic Claude":
        llm = ChatAnthropic(model=claudeModelName, api_key=st.session_state.api_key)
    elif st.session_state.model_choice == "OpenAI ChatGPT":
        # OpenAI ChatGPT를 사용하는 코드
        llm = ChatOpenAI(model=chatGPTModelName, api_key=st.session_state.api_key)

    # 검색 도구 생성 with Tavily API key
    tool = TavilySearch(api_key=st.session_state.tavily_api_key, max_results=3)
    tools = [tool]

    # LLM에 tool binding
    llm_with_tools = llm.bind_tools(tools)

    # Chatbot 함수 정의
    def chatbot(state: State):
        answer = llm_with_tools.invoke(state["messages"])
        return {"messages": [answer]}

    # Tool Node 정의
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

    # 도구 노드 생성
    tool_node = BasicToolNode(tools=[tool])

    ###### STEP 3. 그래프 (Graph) 정의, 노드 추가 ######
    # 그래프 생성
    graph_builder = StateGraph(State)

    # 그래프에 chatbot 노드 추가
    graph_builder.add_node("chatbot", chatbot)
    # 그래프에 tool_node 노드 추가
    graph_builder.add_node("tools", tool_node)

    ###### STEP 4. 그래프 Edge 추가 ######
    graph_builder.add_edge(START, "chatbot")

    # 그래프에 conditional edge 추가
    def route_tools(
        state: State,
    ):
        # 메시지가 존재할 경우 가장 최근 메시지 1개 추출
        if messages := state.get("messages", []):
            # 가장 최근 AI 메시지 추출
            ai_message = messages[-1]
        else:
            # 입력 상태에 메시지가 없는 경우 예외 발생
            raise ValueError(f"No messages found in input state to tool_edge: {state}")

        # AI 메시지에 도구 호출이 있는 경우 "tools" 반환
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            # 도구 호출이 있는 경우 "tools" 반환
            return "tools"
        # 도구 호출이 없는 경우 "END" 반환
        return END

    # `tools_condition` 함수는 챗봇이 도구 사용을 요청하면 "tools"를 반환하고, 직접 응답이 가능한 경우 "END"를 반환
    graph_builder.add_conditional_edges(
        source="chatbot",
        path=route_tools,
        # route_tools 의 반환값이 "tools" 인 경우 "tools" 노드로, 그렇지 않으면 END 노드로 라우팅
        path_map={"tools": "tools", END: END},
    )

    # tools > chatbot
    graph_builder.add_edge("tools", "chatbot")

    ###### STEP 5. 그래프 컴파일 ######
    graph = graph_builder.compile()

    ###### STEP 6. 그래프 시각화 ######
    # 그래프 시각화
    try:
        # Get the mermaid PNG as bytes
        graph_bytes = graph.get_graph().draw_mermaid_png()

        # Convert the bytes to an image and display it in Streamlit
        st.image(graph_bytes, caption="Chatbot Graph")

    except Exception as e:
        st.error(f"Failed to display graph: {e}")

    if "messages_01" not in st.session_state:
        st.session_state.messages_01 = []

    ##### STEP 7. 이전 메시지 표시 #####
    for message in st.session_state.messages_01:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    ##### STEP 8. 새로운 사용자 입력 처리 #####
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages_01.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 전체 대화 기록을 모델에 보내기 위해 준비
        full_conversation = [
            (msg["role"], msg["content"]) for msg in st.session_state.messages_01
        ]

        # Create an expander for showing the execution steps
        with st.expander("Show Execution Steps", expanded=True):
            # Create a container for step progress
            step_container = st.empty()

            # 대화 기록을 모델에 보내기
            for event in graph.stream(
                {"messages": full_conversation}, stream_mode="values"
            ):
                for key, value in event.items():
                    # Display step information in a nice format
                    with step_container.container():
                        st.markdown(f"**Current Step: {key}**")
                        st.divider()

                        # Display the message content
                        if (
                            isinstance(value, dict)
                            and "messages" in value
                            and value["messages"]
                        ):
                            last_message = value["messages"][-1]
                            if hasattr(last_message, "content"):
                                st.markdown(f"```\n{last_message.content}\n```")
                            if (
                                hasattr(last_message, "tool_calls")
                                and last_message.tool_calls
                            ):
                                st.markdown("**Tool Calls:**")
                                for tool_call in last_message.tool_calls:
                                    st.markdown(f"- Tool: `{tool_call['name']}`")
                                    st.markdown(f"  Args: `{tool_call['args']}`")
                        else:
                            st.markdown(f"```\n{value}\n```")

                # Display the final response in the chat interface
                if isinstance(value, dict) and "messages" in value:
                    response = value["messages"][-1].content
                    st.session_state.messages_01.append(
                        {"role": "assistant", "content": response}
                    )
                    with st.chat_message("assistant"):
                        st.markdown(response)


def collect_api_keys():
    """Collect and validate all required API keys."""
    if "api_key_submitted" not in st.session_state:
        st.session_state.api_key_submitted = False

    if not st.session_state.api_key_submitted:
        col1, col2 = st.columns(2)

        with col1:
            if st.session_state.model_choice == "Anthropic Claude":
                llm_api_key = st.text_input("Anthropic API Key:", type="password")
            else:
                llm_api_key = st.text_input("OpenAI API Key:", type="password")

        with col2:
            tavily_api_key = st.text_input("Tavily Search API Key:", type="password")

        if st.button("Submit API Keys"):
            if llm_api_key and tavily_api_key:
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
