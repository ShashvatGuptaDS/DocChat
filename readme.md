# DocChat

I built this because I was tired of ctrl+F-ing through long PDFs to find things. DocChat lets you just ask questions about your documents in plain English and get answers with the exact source sections highlighted.

Upload a PDF, DOCX, or plain text file, hit Process, and start chatting. It remembers the conversation so follow-up questions work naturally — you don't have to repeat yourself.

## What it does

- Upload one or more documents (PDF, DOCX, TXT)
- Ask questions in natural language
- See which exact sections of the document were used to answer
- Export the full conversation as a text file
- Switch between OpenAI (GPT-3.5) and HuggingFace (Qwen2.5-72B) backends

## How it works under the hood

When you upload a file, the app extracts the text, splits it into overlapping 1000-character chunks, and converts each chunk into a vector embedding using the chosen model. Those vectors go into a FAISS index stored in memory.

When you ask a question, your question gets embedded the same way, and FAISS finds the 4 most similar chunks by cosine similarity. Those chunks get passed to the LLM as context alongside your question and the conversation history. The LLM reads the relevant sections and synthesises an answer — it won't make things up if the answer isn't in the documents.

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
├── .env.example      # Template — copy this to .env and fill in your keys
└── .gitignore
```

## Tech stack

Python, Streamlit, LangChain, FAISS, OpenAI / HuggingFace
