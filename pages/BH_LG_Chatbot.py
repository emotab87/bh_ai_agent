import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, CompiledStateGraph
from langgraph.graph.message import add_messages
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

    # Chatbot 함수 정의
    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    ###### STEP 3. 그래프 (Graph) 정의, 노드 추가 ######
    # 그래프 생성
    graph_builder = StateGraph(State)

    # 노드 추가
    graph_builder.add_node("chatbot", chatbot)

    ###### STEP 4. 그래프 Edge 추가 ######
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    ###### STEP 5. 그래프 컴파일 ######
    graph = graph_builder.compile()

    ###### STEP 6. 그래프 시각화 ######
    # 그래프 시각화
    try:
        if visualize_graph_streamlit(graph):
            st.success("Graph visualization completed successfully")
        else:
            st.warning("Failed to generate graph visualization")
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

        # 대화 기록을 모델에 보내기
        for event in graph.stream({"messages": full_conversation}):
            for value in event.values():
                # AIMessage 객체에서 직접 내용에 접근
                response = value["messages"][-1].content
                st.session_state.messages_01.append(
                    {"role": "assistant", "content": response}
                )
                with st.chat_message("assistant"):
                    st.markdown(response)


def visualize_graph_streamlit(graph, xray=False):
    """
    CompiledStateGraph 객체를 시각화하여 이미지 데이터를 반환합니다.
    """
    try:
        if isinstance(graph, CompiledStateGraph):
            # Get the Mermaid diagram as a string
            mermaid_str = graph.get_graph(xray=xray).to_mermaid()

            # Create a Mermaid diagram using Streamlit's built-in support
            st.mermaid(mermaid_str)
            return True
        return None
    except Exception as e:
        print(f"[ERROR] Visualize Graph Error: {e}")
        return None


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

    # Check if the API key has been submitted
    if "api_key_submitted" not in st.session_state:
        st.session_state.api_key_submitted = False

    # 만약 API 키가 제출되지 않았다면, 입력 필드를 표시
    if not st.session_state.api_key_submitted:
        if st.session_state.model_choice == "Anthropic Claude":
            api_key = st.text_input(
                "Please input your Anthropic API Key:", type="password"
            )
        elif st.session_state.model_choice == "OpenAI ChatGPT":
            api_key = st.text_input(
                "Please input your OpenAI API Key:", type="password"
            )

        # Button to submit the API key
        if st.button("Submit"):
            if api_key:
                st.session_state.api_key = api_key
                st.session_state.api_key_submitted = True
            else:
                st.warning("Please input your Anthropic API Key.")

    # After API key submission, start the chatbot interaction
    if st.session_state.api_key_submitted:
        LangGraph_run()


if __name__ == "__main__":
    main()
