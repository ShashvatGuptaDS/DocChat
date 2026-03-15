# DocChat

I built this because I was tired of ctrl+F-ing through long PDFs to find things. DocChat lets you just ask questions about your documents in plain English and get answers with the exact source sections highlighted.

Upload a PDF, DOCX, or plain text file, hit Process, and start chatting. It remembers the conversation so follow-up questions work naturally — you don't have to repeat yourself.

## What it does

- Upload one or more documents (PDF, DOCX, TXT)
- Ask questions in natural language
- See which exact sections of the document were used to answer
- Export the full conversation as a text file
- Switch between OpenAI (GPT-3.5) and HuggingFace (Qwen2.5-72B) backends

## How it works: Retrieval-Augmented Generation (RAG)

DocChat is a **Retrieval-Augmented Generation (RAG)** system. Instead of relying solely on the LLM's internal memory (which can result in hallucinations), the system "retrieves" relevant facts from your documents and "augments" the prompt before "generating" an answer.

### The RAG Workflow

1.  **Ingestion & Embedding**: 
    When you upload a file, the app extracts the text, splits it into overlapping 1000-character chunks, and converts each chunk into a mathematical vector embedding. These vectors are stored in a local **FAISS** index.
2.  **Retrieval**: 
    When you ask a question, the system converts your query into a vector and finds the 4 most similar chunks in your documents by calculating cosine similarity.
3.  **Augmentation & Generation**: 
    The retrieved chunks are passed to the LLM (OpenAI or Qwen) as context. The LLM is instructed to answer **only** based on that context. This ensures the answer is grounded in your documents.

### Why RAG?
- **Accuracy**: Answers are factually grounded in your specific files.
- **Verifiability**: You can inspect the source chunks used for every answer.
- **Privacy**: Your sensitive data stays within the context of the chat session rather than being part of the model's global training.

## Setup

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the environment template and add your API keys:
   ```bash
   cp .env.example .env
   # then edit .env
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Running with Docker

1. Copy `.env.example` to `.env` and add your API keys.
2. Build and start the container:
   ```bash
   docker-compose up --build
   ```
3. Access the app at `http://localhost:8501`.

## API Keys

You need at least one of:

- `OPENAI_API_KEY` — for the OpenAI backend (GPT-3.5). Get one at [platform.openai.com](https://platform.openai.com)
- `HUGGINGFACEHUB_API_TOKEN` — for the HuggingFace backend (Qwen2.5-72B, free). Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

These go in a `.env` file in the project root. The `.env` file is git-ignored so your keys won't accidentally end up on GitHub.

## Project structure

```
DocChat/
├── app.py            # Main Streamlit app
├── utils.py          # Text extraction, chunking, vectorstore
├── htmlTemplates.py  # Chat bubble HTML/CSS
├── requirements.txt  # Python dependencies
├── logs/             # Persistent application logs (git-ignored)
├── Dockerfile        # Docker image definition
├── docker-compose.yml # Container orchestration
├── .dockerignore     # Files to exclude from Docker image
├── .env.example      # Template — copy this to .env and fill in your keys
└── .gitignore
```

## Tech stack

Python, Streamlit, LangChain, FAISS, OpenAI / HuggingFace
