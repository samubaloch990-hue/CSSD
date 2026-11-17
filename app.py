import streamlit as st

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("📚 AI Document & Topic Knowledge Assistant")
st.markdown("""
Welcome! This application allows you to query specialized knowledge based on provided files and documents.
""")

# Sidebar for configuration and file management
st.sidebar.header("📁 Knowledge Base Setup")

uploaded_files = st.sidebar.file_uploader(
    "Upload reference documents (PDF, TXT, CSV, MD)",
    accept_multiple_files=True,
    type=["pdf", "txt", "csv", "md"]
)

if uploaded_files:
    st.sidebar.success(f"Successfully uploaded {len(uploaded_files)} file(s)!")

# Uploaded files stay in the current Streamlit session instead of shared disk storage.
existing_files = uploaded_files or []
if existing_files:
    st.sidebar.markdown("### Loaded Documents:")
    for uploaded_file in existing_files:
        st.sidebar.text(f"• {uploaded_file.name}")
else:
    st.sidebar.info("No documents uploaded yet. Upload files above to give the AI specific knowledge.")

# Main chat interface
st.header("💬 Ask Questions")
user_query = st.text_input("What would you like to know about the topic or uploaded files?")

if user_query:
    if not existing_files:
        st.warning("Please upload at least one document in the sidebar so the AI has context to answer from!")
    else:
        with st.spinner("Analyzing documents and generating answer..."):
            # Placeholder response logic - can be upgraded to LangChain/LlamaIndex RAG
            st.markdown("### Answer:")
            st.info(f"You asked: **{user_query}**\n\nBased on the {len(existing_files)} uploaded document(s), here is the analysis...")

            # Show sources
            st.markdown("---")
            st.markdown("**Source Files Consulted:**")
            for uploaded_file in existing_files:
                st.markdown(f"- `{uploaded_file.name}`")
