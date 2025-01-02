import streamlit as st
import datetime
from openai import OpenAIError
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback


def modelName():
    """Return the model name to use"""
    return "gpt-3.5-turbo"


def generate_text(api_key, birth_date, gender, language):
    """
    Function to interact with OpenAI API using langchain
    """
    try:
        model_name = modelName()
        # Initialize ChatOpenAI with the provided API key
        llm = ChatOpenAI(openai_api_key=api_key, model_name=model_name)

        # Create instruction based on language
        instruction = """사용자의 생년 월일 그리고 성별을 입력 받아 그 나이에 사용자가 생각하거나 행동하면 좋은 일들을 알려 주세요. 
        생년월일과 관계 되는 별자리와 꽃도 언급해 주세요. 동양에서 사용하는 띠도 알려 주세요. 
        그리고 용기를 내고 위안이 될 수 있는 말을 해 주세요. 
        현재까지 잘 살아 왔고 지금 나이에는 어떤 일들을 계획하고 행동하는게 좋은지 자세하게 알려 주세요. 
        올해는 2025년으로 설정하고 나이를 계산해 주세요."""

        query = f"{instruction}. {birth_date} 태어난 {gender}가 지금 생각하거나 행동해야 할 일은 무엇이 있을까요? 대답은 {language}로 해 주세요."

        # Show loading spinner while generating
        with st.spinner("Generating counsel..."):
            # Use the callback to track token usage
            with get_openai_callback() as cb:
                generated_text = llm.invoke(query)
                # Display only the content of the generated text
                st.write("Generated counsel:")
                st.write(generated_text.content)

                # Display token usage statistics
                st.write("Token Usage Statistics:")
                st.write(f"- Total Tokens: {cb.total_tokens}")
                st.write(f"- Prompt Tokens: {cb.prompt_tokens}")
                st.write(f"- Completion Tokens: {cb.completion_tokens}")
                st.write(f"- Total Cost (USD): ${cb.total_cost:.4f}")

        return generated_text

    except OpenAIError as e:
        st.warning(
            "Incorrect API key provided or OpenAI API error. You can find your API key at https://platform.openai.com/account/api-keys."
        )
        return None


st.title("Today's Counsel")

# Add a text input for the OpenAI API key
api_key = st.text_input("Please input your OpenAI API Key:", type="password")

# Birth date selector with extended range
min_date = datetime.date(1950, 1, 1)
max_date = datetime.date.today()
birth_date = st.date_input(
    "Select your birth date",
    min_value=min_date,
    max_value=max_date,
    value=datetime.date(1987, 5, 30),
)

# Gender selection
gender = st.radio("Select your gender", ["Male", "Female"])

# Language selector
language = st.selectbox("Select a language:", ["English", "한국어", "日本語", "中文"])

# Create counsel button
if st.button("Create a counsel."):
    if not api_key:
        st.warning("Please insert your OpenAI API key.")
    else:
        # Convert birth_date to string format
        birth_date_str = birth_date.strftime("%Y-%m-%d")
        # Generate the counsel
        generated_text = generate_text(api_key, birth_date_str, gender, language)
