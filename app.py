
from pathlib import Path
import hashlib
import re

import streamlit as st
from rag import ask_question, build_index

# 1. Project paths

DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")
INDEX_PATH = STORAGE_DIR / "index.faiss"
METADATA_PATH = STORAGE_DIR / "metadata.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# 2. Page configuration

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 3. Frontend styling

st.markdown(
    """
    <style>
    /* Main page */
    .stApp {
        background-color: #f4f7fc;
        color: #1e293b;
    }

    /* Main content text */
    .stApp, .stApp p, .stApp label,
    .stApp span, .stApp div {
        color: #1e293b;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #e8eef8;
    }

    [data-testid="stSidebar"] * {
        color: #1e293b;
    }

    /* Main title */
    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        color: #1d3557 !important;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #475569 !important;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Information card */
    .info-card {
        background-color: #ffffff;
        color: #1e293b !important;
        border: 1px solid #d5deeb;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
    }

    .info-card strong {
        color: #1d3557 !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #ffffff;
        color: #1d3557 !important;
        border: 1px solid #b8c7dc;
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
    }

    div.stButton > button:hover {
        background-color: #e2eaf7;
        border-color: #7189ae;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        color: #1e293b;
        border: 1px solid #dce4ef;
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 10px;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong {
        color: #1e293b !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background-color: #ffffff;
        color: #1e293b !important;
        caret-color: #1e293b;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
        opacity: 1;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 8px;
    }

    /* Headings */
    .stApp h1, .stApp h2, .stApp h3 {
        color: #1d3557 !important;
    }

    /* Dividers */
    hr {
        border-color: #d5deeb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# 4. Initialize session state

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

if "processed_upload_hash" not in st.session_state:
    st.session_state.processed_upload_hash = None


# 5. Sidebar: PDF upload and controls

current_upload_hash = None
uploaded_bytes = None

with st.sidebar:
    st.title("📄 Knowledge Base")
    st.caption("Upload a PDF to start asking questions.")

    uploaded_pdf = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Select a PDF containing readable text.",
        key="pdf_uploader",
    )

    if uploaded_pdf is not None:
        uploaded_bytes = uploaded_pdf.getvalue()

        current_upload_hash = hashlib.sha256(
            uploaded_bytes
        ).hexdigest()

        st.caption(f"Selected: {uploaded_pdf.name}")

        if (
            st.session_state.index_ready
            and current_upload_hash
            == st.session_state.processed_upload_hash
        ):
            st.success("This PDF has been processed.")

        else:
            st.info("Process this PDF before asking questions.")

        if st.button(
            "⚙️ Process PDF",
            type="primary",
            use_container_width=True,
        ):
            try:
                safe_filename = Path(uploaded_pdf.name).name
                pdf_path = DATA_DIR / safe_filename

                with open(pdf_path, "wb") as file:
                    file.write(uploaded_bytes)

                with st.spinner(
                    "Reading PDFs and building the search index..."
                ):
                    build_index()

                st.session_state.index_ready = True
                st.session_state.processed_upload_hash = (
                    current_upload_hash
                )

                st.success("PDF processed successfully!")

            except Exception as error:
                st.error(f"Could not process the PDF: {error}")

    st.divider()

    if st.session_state.index_ready:
        st.success("Document index ready for this session.")
    else:
        st.warning("Upload and process a PDF to begin.")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Clearing chat removes conversation messages only. "
        "It does not delete your PDFs or search index."
    )



# 6. Main page header

st.markdown(
    '<div class="main-title">📚 RAG Document Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your documents and get answers '
    'with source references.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-card">
        <strong>How to use the assistant</strong><br>
        1. Upload a PDF in the sidebar.<br>
        2. Click <strong>Process PDF</strong>.<br>
        3. Ask a question in the chat below.
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("💬 Conversation")


# 7. Display existing chat history

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# 8. Normalize casual messages

def normalize_message(message):
    # casefold() supports more complete Unicode
    # case-insensitive comparisons than lower().
    message = message.casefold()

    # Ignore punctuation and repeated whitespace
    # when matching short greetings and thanks.
    message = re.sub(r"[^\w\s]", "", message)
    return " ".join(message.split())


greetings = {
    "hi",
    "hey",
    "hello",
    "good morning",
    "good afternoon",
    "good evening",
}

thank_you_phrases = {
    "thanks",
    "thank you",
    "thankyou",
    "thanks a lot",
    "thank you so much",
    "thanks so much",
    "many thanks",
}



# 9. Handle new user messages

question = st.chat_input(
    "Ask a question about your PDF..."
)

if question:
    # Store the user's original text exactly once.
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    normalized_question = normalize_message(question)

    # Prepare the response without calling the LLM
    # for simple greetings or thank-you messages.
    if normalized_question in greetings:
        answer = (
            "Hello! 👋 I'm your RAG Document Assistant. "
            "Upload and process a PDF to ask questions "
            "about its contents."
        )

    elif normalized_question in thank_you_phrases:
        answer = (
            "You're very welcome! 😊 "
            "I'm happy to help with your documents."
        )

    else:
        # A different, unprocessed upload must not
        # accidentally use an older document index.
        upload_matches_index = (
            current_upload_hash is None
            or current_upload_hash
            == st.session_state.processed_upload_hash
        )

        can_answer = (
            st.session_state.index_ready
            and upload_matches_index
            and INDEX_PATH.exists()
            and METADATA_PATH.exists()
        )

        if not can_answer:
            answer = (
                "Please upload a PDF and click "
                "'Process PDF' before asking document questions."
            )

        else:
            try:
                with st.spinner(
                    "Searching your documents and preparing an answer..."
                ):
                    # Pass the original question, preserving
                    # its capitalization, to the RAG pipeline.
                    answer = ask_question(
                        question,
                        chat_history=st.session_state.messages[:-1],
                    )

            except Exception as error:
                answer = f"Something went wrong: {error}"

    # Display and save every assistant response once.
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })
