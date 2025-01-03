import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
from openai import OpenAIError


def modelName():
    """Return the model name to use"""
    return "gpt-3.5-turbo"


def modelName4o():
    """Return the model name to use"""
    return "gpt-4o-2024-05-13"


def modelName_embedding_small():
    """Return the embedding model name to use"""
    return "text-embedding-ada-002"


def load_and_process_documents(url):
    """Load and process documents from a URL."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        return text_splitter.split_documents(docs)
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None


def setup_vector_store(documents, embeddings):
    """Create and return a FAISS vector store."""
    try:
        return FAISS.from_documents(documents, embeddings)
    except Exception as e:
        st.error(f"Error setting up vector store: {str(e)}")
        return None


def create_qa_chain(api_key, model_name, language):
    """Create the QA chain with the specified model."""
    try:
        llm = ChatOpenAI(openai_api_key=api_key, model_name=model_name)

        # Define the system message for the prompt
        system_message = (
            """You are an expert programmer and problem-solver, tasked with answering any question about Langchain. Answer all questions in """
            + language
            + """.
            Generate a comprehensive and informative answer of 80 words or less for the given question based solely on the provided search results 
            (URL and content). You must only use information from the provided search results. Use an unbiased and journalistic tone. 
            Combine search results together into a coherent answer. Do not repeat text. Cite search results using [${{number}}] notation. 
            Only cite the most relevant results that answer the question accurately. Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end. If different results refer to different entities within the same name, write separate answers for each entity.
            You should use bullet points in your answer for readability. Put citations where they apply rather than putting them all at the end.
            If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure." Don't try to make up an answer.
            """
        )

        # Create the chat prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "Here is the context:\n\n{context}"),
                ("human", "Question: {input}"),
            ]
        )

        # Create and return the document chain
        return create_stuff_documents_chain(llm, prompt)

    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None


def generate_text(api_key, language, question, select_model):
    """Generate text response using the specified model and question."""
    try:
        # Model selection
        model_name = modelName() if select_model == "Cheapest" else modelName4o()
        st.info(f"Using model: {model_name}")

        # Document processing
        url = "https://docs.smith.langchain.com/user_guide"
        documents = load_and_process_documents(url)
        if not documents:
            return None

        # Setup embeddings and vector store
        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key, model=modelName_embedding_small()
        )
        vector_store = setup_vector_store(documents, embeddings)
        if not vector_store:
            return None

        # Create retrieval chain
        qa_chain = create_qa_chain(api_key, model_name, language)
        if not qa_chain:
            return None

        retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), qa_chain)

        # Generate response with callback
        with get_openai_callback() as cb:
            response = retrieval_chain.invoke(
                {"input": f"{question} (Please respond in {language})"}
            )
            st.write("Token Usage Statistics:")
            st.write(f"- Total Tokens: {cb.total_tokens}")
            st.write(f"- Prompt Tokens: {cb.prompt_tokens}")
            st.write(f"- Completion Tokens: {cb.completion_tokens}")
            st.write(f"- Total Cost (USD): ${cb.total_cost:.4f}")

        return response

    except OpenAIError as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None


# Predefined questions
questions = [
    "How can langsmith help with testing?",
    "Please summarize on Prototyping.",
    "Please tell me about Beta Testing.",
    "Please explain about Production.",
    "Please summarize whole LangSmith User Guide.",
]

# Available languages
available_languages = [
    "Korean",
    "Spanish",
    "French",
    "German",
    "Chinese",
    "Japanese",
    "English",
]


def main():
    st.title("LangChain Quickstart 02 - Retrieval Chain")

    # Configuration section
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key:", type="password")
        select_model = st.radio(
            "Choose Model:",
            ["Cheapest", "gpt-4o-2024-05-13"],
            help="Select the model to use for generation",
        )

    # Main content
    st.write("Source: https://docs.smith.langchain.com/user_guide")

    col1, col2 = st.columns(2)

    with col1:
        # Question selection
        selected_question = st.selectbox(
            "Select a question:", questions, help="Choose a predefined question"
        )

    with col2:
        # Language selection
        selected_language = st.selectbox(
            "Select output language:",
            available_languages,
            help="Choose the language for the response",
        )

    if st.button("Generate Response", type="primary"):
        if not api_key:
            st.warning("Please provide your OpenAI API key.")
            return

        with st.spinner("Generating response..."):
            response = generate_text(
                api_key, selected_language, selected_question, select_model
            )
            if response:
                st.success("Response generated successfully!")
                st.subheader("Generated Response:")
                st.write(response["answer"])


if __name__ == "__main__":
    main()
