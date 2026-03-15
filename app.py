"""
app.py
------
DocChat — Chat with your documents using an AI assistant.

Upload PDF, DOCX, or TXT files, click Process, then ask questions about
the content. Conversation history is kept so follow-up questions work
naturally.

Supported backends (select from the sidebar):
  - OpenAI   — requires OPENAI_API_KEY in your .env file
  - HuggingFace — free hosted inference, requires HUGGINGFACEHUB_API_TOKEN

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI

from htmlTemplates import bot_template, css, user_template
from utils import extract_text, get_text_chunks, get_vectorstore

load_dotenv()

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def setup_logging():
    """Configure logging to both console and a rotating file in logs/."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_formatter = logging.Formatter(
        "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    )
    log_file = os.path.join(log_dir, "app.log")

    # Rotating file handler (1MB per file, 5 backups)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # Root logger config
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based \
on the provided document excerpts. Be concise and accurate.

If the answer is not contained in the documents, say so honestly rather \
than guessing.

Relevant document excerpts:
{context}"""

# Chat prompt — used by the OpenAI backend (supports message roles)
CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])




# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def build_llm(backend: str):
    """Instantiate the language model for the chosen backend.

    Args:
        backend: ``"openai"`` or ``"huggingface"``.

    Returns:
        A LangChain LLM or chat model instance.

    Raises:
        EnvironmentError: If the required API key is missing.
    """
    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("your-"):
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file and restart."
            )
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

    # HuggingFace — use ChatHuggingFace wrapping HuggingFaceEndpoint
    # with a model available via Inference Providers (chat_completion API).
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    if not token or token.startswith("your-"):
        raise EnvironmentError(
            "HUGGINGFACEHUB_API_TOKEN is not set. Add it to your .env file and restart."
        )
    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=token,
        temperature=0.3,
        max_new_tokens=1024,
    )
    return ChatHuggingFace(llm=endpoint)



# ---------------------------------------------------------------------------
# LCEL chain
# ---------------------------------------------------------------------------

def build_chain(vectorstore, backend: str):
    """Build a retrieval-augmented generation chain using LCEL.

    Both backends (OpenAI and HuggingFace) are chat models, so we use
    the same ``ChatPromptTemplate`` for both.

    Args:
        vectorstore: FAISS index built from the uploaded documents.
        backend: ``"openai"`` or ``"huggingface"``.

    Returns:
        A compiled LCEL runnable accepting
        ``{"question": str, "chat_history": list}``.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = build_llm(backend)

    def format_docs(docs) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
            source_docs=lambda x: retriever.invoke(x["question"]),
        )
        | {
            "answer": CHAT_PROMPT | llm | StrOutputParser(),
            "source_docs": lambda x: x["source_docs"],
        }
    )

    return chain


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------

def render_chat_history() -> None:
    """Render all messages in the session history as styled chat bubbles.

    Even-indexed messages are the user's questions, odd-indexed are the
    assistant's replies — matching how LangChain alternates HumanMessage
    and AIMessage.
    """
    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )


def handle_user_question(question: str) -> None:
    """Send the user's question through the chain and update the UI.

    Appends both the question and the answer to the session chat history
    so memory is preserved across turns.

    Args:
        question: Natural-language question about the uploaded documents.
    """
    if st.session_state.chain is None:
        st.warning("Please upload and process your documents first.")
        return

    result = st.session_state.chain.invoke({
        "question": question,
        "chat_history": st.session_state.chat_history,
    })
    answer = result["answer"]
    st.session_state.source_docs = result.get("source_docs", [])

    # Append to history as typed message objects (compatible with the prompt)
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))
    render_chat_history()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Render the sidebar: backend selector, uploader, and action buttons."""
    with st.sidebar:
        st.title("⚙️ Settings")

        backend = st.selectbox(
            "LLM Backend",
            options=["huggingface", "openai"],
            format_func=lambda x: (
                "🤗 HuggingFace (Qwen2.5-72B)" if x == "huggingface" else "🔷 OpenAI (GPT-3.5)"
            ),
            help="HuggingFace is free. OpenAI needs an API key.",
        )
        st.session_state.backend = backend

        st.divider()

        st.subheader("📂 Your Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs, DOCX, or TXT files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        if st.button("⚡ Process", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.warning("Upload at least one file before processing.")
                return

            with st.spinner("Reading and indexing your documents…"):
                try:
                    # Capture first file name for export naming (without extension)
                    first_file = uploaded_files[0].name
                    st.session_state.doc_name = os.path.splitext(first_file)[0]

                    raw_text = extract_text(uploaded_files)
                    chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(tuple(chunks), backend=backend)
                    st.session_state.chain = build_chain(vectorstore, backend)
                    st.session_state.chat_history = []
                    st.session_state.source_docs = []
                    st.success(
                        f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s)."
                    )
                except EnvironmentError as exc:
                    st.error(str(exc))
                except ValueError as exc:
                    st.warning(str(exc))
                except Exception as exc:
                    st.error(f"Processing error: {exc}")
                    logger.exception("Processing failed")

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.source_docs = []
            st.session_state.chain = None
            st.rerun()

        # Export button — only shown once there's something to export
        if st.session_state.get("chat_history"):
            transcript = _build_transcript(st.session_state.chat_history)
            # Use ddmmyy format as requested
            date_str = datetime.now().strftime("%d%m%y")
            doc_name = st.session_state.get("doc_name", "docchat_export")
            st.download_button(
                label="💾 Export Chat (.txt)",
                data=transcript,
                file_name=f"{doc_name}_{date_str}.txt",
                mime="text/plain",
                use_container_width=True,
            )


def _build_transcript(chat_history: list) -> str:
    """Format the chat history as a plain-text transcript for download.

    Args:
        chat_history: List of ``HumanMessage`` / ``AIMessage`` objects.

    Returns:
        Multi-line string with labelled User / Assistant turns.
    """
    lines = [f"DocChat Export — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
    lines.append("=" * 60)
    for message in chat_history:
        speaker = "User" if isinstance(message, HumanMessage) else "Assistant"
        lines.append(f"\n{speaker}: {message.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Source document expander
# ---------------------------------------------------------------------------

def render_source_docs() -> None:
    """Show the document chunks that were retrieved for the last question."""
    docs = st.session_state.get("source_docs", [])
    if not docs:
        return
    with st.expander(f"📄 Source chunks used ({len(docs)} retrieved)", expanded=False):
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**Chunk {i}**")
            st.caption(
                doc.page_content[:500] + ("…" if len(doc.page_content) > 500 else "")
            )
            if i < len(docs):
                st.divider()


# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------

def initialise_session_state() -> None:
    """Initialise all session-state keys to safe defaults on first load."""
    defaults = {
        "chain": None,
        "chat_history": [],
        "source_docs": [],
        "backend": "huggingface",
        "doc_name": "docchat_export",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Configure the Streamlit page and render the full application."""
    st.set_page_config(
        page_title="DocChat — Chat with your documents",
        page_icon="📚",
        layout="wide",
    )
    st.write(css, unsafe_allow_html=True)
    initialise_session_state()
    render_sidebar()

    st.title("📚 DocChat")
    st.caption("Upload documents, then ask anything about their content.")
    st.divider()

    if st.session_state.chat_history:
        render_chat_history()
        render_source_docs()
    else:
        st.info(
            "👈 Upload your documents in the sidebar and click **⚡ Process** to get started.",
            icon="ℹ️",
        )

    st.divider()

    user_question = st.chat_input("Ask a question about your documents…")
    if user_question:
        handle_user_question(user_question)


if __name__ == "__main__":
    main()
