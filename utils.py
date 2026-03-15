"""
utils.py
--------
Shared document-processing utilities for DocChat.

Functions here handle:
  - Extracting raw text from PDF, DOCX, and plain-text files
  - Splitting long text into overlapping chunks for retrieval
  - Building and caching a FAISS vector store from those chunks

All heavy work (embedding + index creation) is wrapped with
``@st.cache_resource`` so Streamlit only rebuilds it when the uploaded
files actually change — not on every UI interaction.
"""

from __future__ import annotations

import io
import logging
from typing import Literal

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdfs(pdf_files: list) -> str:
    """Extract and concatenate plain text from one or more PDF uploads.

    Args:
        pdf_files: List of file-like objects returned by ``st.file_uploader``.

    Returns:
        A single string containing all extracted text, pages separated by
        a newline.  Returns an empty string if no text could be found.
    """
    from pypdf import PdfReader  # local import keeps startup fast

    full_text = []
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text.append(page_text)
        except Exception:
            logger.exception("Failed to read PDF: %s", getattr(pdf_file, "name", "unknown"))

    return "\n".join(full_text)


def extract_text_from_docx(docx_files: list) -> str:
    """Extract plain text from one or more DOCX uploads.

    Args:
        docx_files: List of file-like objects returned by ``st.file_uploader``.

    Returns:
        Combined text from all paragraphs across all documents.
    """
    import docx  # local import; requires python-docx

    full_text = []
    for docx_file in docx_files:
        try:
            doc = docx.Document(io.BytesIO(docx_file.read()))
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
        except Exception:
            logger.exception("Failed to read DOCX: %s", getattr(docx_file, "name", "unknown"))

    return "\n".join(full_text)


def extract_text_from_txts(txt_files: list) -> str:
    """Decode and concatenate plain-text file uploads.

    Args:
        txt_files: List of file-like objects returned by ``st.file_uploader``.

    Returns:
        Combined raw text from all files.
    """
    full_text = []
    for txt_file in txt_files:
        try:
            content = txt_file.read()
            # Handle both bytes and already-decoded strings
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            full_text.append(content)
        except Exception:
            logger.exception("Failed to read TXT: %s", getattr(txt_file, "name", "unknown"))

    return "\n".join(full_text)


def extract_text(uploaded_files: list) -> str:
    """Route uploaded files to the correct extractor based on file extension.

    Supports ``.pdf``, ``.docx``, and ``.txt`` files mixed in one upload batch.

    Args:
        uploaded_files: List of file-like objects from ``st.file_uploader``.

    Returns:
        All extracted text concatenated into one string.

    Raises:
        ValueError: If the upload list is empty or no text could be extracted.
    """
    pdfs, docxs, txts = [], [], []

    for f in uploaded_files:
        name = (f.name or "").lower()
        if name.endswith(".pdf"):
            pdfs.append(f)
        elif name.endswith(".docx"):
            docxs.append(f)
        elif name.endswith(".txt"):
            txts.append(f)
        else:
            logger.warning("Skipping unsupported file type: %s", f.name)

    parts = []
    if pdfs:
        parts.append(extract_text_from_pdfs(pdfs))
    if docxs:
        parts.append(extract_text_from_docx(docxs))
    if txts:
        parts.append(extract_text_from_txts(txts))

    combined = "\n".join(p for p in parts if p.strip())
    if not combined.strip():
        raise ValueError("No readable text found in the uploaded file(s).")

    return combined


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def get_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split a long document into overlapping chunks for retrieval.

    Uses ``RecursiveCharacterTextSplitter`` which tries to break on paragraph
    boundaries first, then sentences, then words — producing cleaner chunks
    than a naïve character split.

    Args:
        text: The full document text to split.
        chunk_size: Maximum characters per chunk (default 1000).
        chunk_overlap: Characters shared between adjacent chunks (default 200).
            Overlap helps preserve context at chunk boundaries.

    Returns:
        List of text chunk strings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# Vector store (cached so Streamlit doesn't rebuild on every re-render)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_vectorstore(
    chunks: tuple[str, ...],
    backend: Literal["openai", "huggingface"] = "openai",
) -> FAISS:
    """Build a FAISS vector store from text chunks.

    This function is decorated with ``@st.cache_resource`` so the embedding
    step only runs once per unique set of chunks.  Passing a *tuple* (not a
    list) as ``chunks`` makes the argument hashable so the cache key works.

    Args:
        chunks: Tuple of text chunk strings (use ``tuple(get_text_chunks(...))``).
        backend: Which embedding model to use.
            ``"openai"`` uses ``OpenAIEmbeddings`` (requires ``OPENAI_API_KEY``).
            ``"huggingface"`` uses ``HuggingFaceEmbeddings`` (runs locally, no key needed).

    Returns:
        A FAISS vectorstore ready for similarity search.
    """
    if backend == "openai":
        embeddings = OpenAIEmbeddings()
    else:
        # Runs a small sentence-transformer model locally — no API key needed
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return FAISS.from_texts(list(chunks), embedding=embeddings)
