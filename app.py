import streamlit as st
import tempfile
from rag_engine import build_rag_chain

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide"
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 📄 PDF RAG Assistant")
    st.markdown(
        """
        Upload a PDF and interact with it using an **advanced AI assistant**
        powered by Retrieval-Augmented Generation.
        """
    )
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.divider()

    if st.button("🔄 Reset / Upload New PDF", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown(
        """
        ---
        **Features**
        - Chat & small talk
        - Section-wise Q&A
        - Full document reasoning
        - Safe & grounded answers
        """
    )

# ---------------- Session State Init ----------------
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Header ----------------
st.markdown(
    """
    <div style="padding: 1rem 0;">
        <h1 style="margin-bottom: 0;">📘 PDF Intelligence Assistant</h1>
        <p style="color: #666; font-size: 1.05rem;">
            Ask questions, summaries, or explore sections of your document
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- PDF Processing ----------------
if uploaded_file and st.session_state.pdf_path is None:
    with st.spinner("📚 Processing PDF and building knowledge base..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.pdf_path = tmp.name

        chain = build_rag_chain(st.session_state.pdf_path)

        if chain is None:
            st.error("❌ Failed to initialize RAG engine.")
            st.stop()

        st.session_state.rag_chain = chain
        st.session_state.messages = []

    st.success("✅ PDF loaded successfully. You can start chatting!")

# ---------------- Chat Area ----------------
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ---------------- Chat Input ----------------
user_input = st.chat_input("Ask something about the document…")

if user_input:
    if st.session_state.rag_chain is None:
        st.warning("📄 Please upload a PDF to begin.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = st.session_state.rag_chain.invoke(user_input)
                st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
