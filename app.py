import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

# Define a function to handle user input
def user_input():
    if st.session_state.conversation is None:
        st.error("Please upload and process PDF files first.")
        return

    # Ensure the query is a valid string
    user_question = st.session_state.user_question.strip()
    if not user_question:
        st.error("Please enter a valid question.")
        return

    # Retrieve response from the conversational chain
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']

    # Get the latest reply
    latest_reply = response['chat_history'][-1].content
    st.session_state.latest_reply = latest_reply

    # Clear the input field after the question is submitted
    st.session_state.user_question = ""


# Main application function
def main():
    # Set up page configuration
    st.set_page_config(page_title="Information Retrieval System", layout="wide")
    st.title("📚 Information Retrieval System ")
    st.subheader("Upload your PDF files, process them, and ask questions interactively!")

    # Sidebar menu
    with st.sidebar:
        st.title("📂 Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF files:", type=["pdf"], accept_multiple_files=True
        )
        process_button = st.button("📤 Submit & Process")

        # File processing
        if process_button:
            if pdf_docs:
                with st.spinner("📖 Processing PDF files..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("✅ Processing complete! You can now ask questions.")
            else:
                st.error("⚠️ Please upload at least one PDF file.")

    # User question input section
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    if "latest_reply" not in st.session_state:
        st.session_state.latest_reply = ""

    # Tabs for better organization
    tabs = st.tabs(["Ask Questions", "Chat History", "About"])
    with tabs[0]:
        st.write("### Ask Your Questions")
        # Input field with session state binding
        st.text_input(
            "Type your question here:",
            key="user_question",
            placeholder="What would you like to know?",
            on_change=user_input  # Automatically triggers when the user hits Enter
        )

        # Display the reply under the question bar
        if st.session_state.latest_reply:
            st.markdown(f"**Reply**: {st.session_state.latest_reply}")

    with tabs[1]:
        st.write("### Chat History")
        if st.session_state.chatHistory:
            for i, message in enumerate(st.session_state.chatHistory):
                if i % 2 == 0:
                    st.markdown(f"**User**: {message.content}")
                else:
                    st.markdown(f"**Reply**: {message.content}")
        else:
            st.info("No chat history available. Start asking questions!")

    with tabs[2]:
        st.write("### About")
        st.markdown(
            """
            This Information Retrieval System allows you to:
            - Upload PDF documents.
            - Process and extract key information.
            - Interact with your data through natural language queries.

            **How it works:**
            1. Upload one or more PDF files in the sidebar.
            2. Click on "Submit & Process" to analyze the files.
            3. Use the "Ask Questions" tab to interact with the data.

            **Built with:** 🐍 Python, LangChain, and HuggingFace models.
            """
        )

    # Display a progress bar during file upload and processing
    if process_button and pdf_docs:
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)


if __name__ == "__main__":
    main()
