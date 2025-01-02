import streamlit as st
from langchain_openai import ChatOpenAI
from openai import OpenAIError
from langchain_community.callbacks import get_openai_callback


def modelName():
    """Return the model name to use"""
    return "gpt-3.5-turbo"


def modelName4o():
    """Return the model name to use"""
    return "gpt-4o-2024-05-13"


def modelName_embedding_small():
    """Return the embedding model name to use"""
    return "text-embedding-ada-002"


# Function to interact with OpenAI API
def generate_text(api_key, language, question, select_model):
    try:
        openai_api_key = api_key
        embedding_model_name = modelName_embedding_small()
        if select_model == "Cheapest":
            model_name = modelName()
        else:
            model_name = modelName4o()

        st.write("*** Work Process ***")

        # 1. Get Data
        from langchain_community.document_loaders import WebBaseLoader

        loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
        docs = loader.load()
        st.write("1. Get data from the Webpage.")

        # 2. Set Embedding model
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key, model=embedding_model_name
        )
        st.write("2. Set Embedding Model. (Model Name is " + embedding_model_name + ")")

        # 3. Store vector into vector storage
        from langchain_community.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)
        st.write("3. Split text and store as vector using FAISS.")

        # 4. create documents chain
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.chains.combine_documents import create_stuff_documents_chain

        prompt = ChatPromptTemplate.from_template(
            """Answer the following question based only on the provided context in English and Translate the answer in """
            + language
            + """ :

        <context>
        {context}
        </context>

        Question: {input}"""
        )

        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name)
        document_chain = create_stuff_documents_chain(llm, prompt)
        st.write(
            "4. Create Document chain. -create_stuff_documents_chain(llm, prompt)-"
        )

        # 5. Create Retrieval Chain
        from langchain.chains import create_retrieval_chain

        retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        st.write(
            "5. Create Retrieval Chain. -create_retrieval_chain(retriever, document_chain)-"
        )

        with get_openai_callback() as cb:
            generated_text = retrieval_chain.invoke({"input": question})
            st.write(cb)

        vector.delete([vector.index_to_docstore_id[0]])
        # Is now missing
        # 0 in vector.index_to_docstore_id

        return generated_text
    except OpenAIError as e:
        st.warning("Incorrect API key provided or OpenAI API error.")
        st.warning(e)


def main():
    st.title("LangChain Quickstart 02 - Retrieval Chain")

    # Get user input for OpenAI API key
    api_key = st.text_input("Please input your OpenAI API Key:", type="password")
    st.write(
        "Fetching this Web Page Contents : https://docs.smith.langchain.com/user_guide"
    )

    select_model = st.radio(
        "Please choose the Model you'd like to use.", ["Cheapest", "gpt-4o-2024-05-13"]
    )

    # List of Questions
    questions = [
        "How can langsmith help with testing?",
        "Please summarize on Prototyping.",
        "Please tell me about Beta Testing.",
        "Please explain about Production.",
        "Please summarize whole LangSmith User Guide.",
    ]

    # User-selected question
    selected_question = st.selectbox("Select a question:", questions)

    st.write("*Answers will be in English and the language of your choice.* ")

    # List of languages available for ChatGPT
    available_languages = [
        "Korean",
        "Spanish",
        "French",
        "German",
        "Chinese",
        "Japanese",
    ]

    # User-selected language
    selected_language = st.selectbox("Select a language:", available_languages)

    # Button to trigger text generation
    if st.button("Submit."):
        if api_key:
            with st.spinner("Wait for it..."):
                # When an API key is provided, display the generated text
                generated_text = generate_text(
                    api_key, selected_language, selected_question, select_model
                )
                st.write(generated_text)
                st.write("**: Answer Only**")
                st.write(generated_text["answer"])
        else:
            st.warning("Please insert your OpenAI API key.")


if __name__ == "__main__":
    main()
