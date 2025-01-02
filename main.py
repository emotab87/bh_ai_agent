import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Main Page",
    page_icon="🏠",
)

# Main page content
st.write("# Welcome to the Main Page! 🏠")
st.sidebar.success("Select a page above.")

st.markdown(
    """
    This is the main entry point for your Streamlit multipage app.
    **👈 Select a page from the sidebar** to navigate through the app!
    
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community forums](https://discuss.streamlit.io)
    """
)
