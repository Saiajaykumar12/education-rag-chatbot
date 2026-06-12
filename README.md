# Education RAG Chatbot

A RAG-powered study assistant that lets students ask questions from any textbook or course PDF in plain English — no manual searching, no reading through pages.

**Live demo:** [your-streamlit-url-here]

## The problem it solves

Students waste hours searching through PDFs trying to find specific information. This chatbot finds the exact relevant section and answers in plain English — and shows exactly which part of the document it used.

## How it works

Student uploads PDF(s)
↓
PDF split into chunks (1000 chars, 200 overlap)
↓
Each chunk embedded using HuggingFace all-MiniLM-L6-v2
↓
Embeddings stored in FAISS vector store
↓
Student asks question in plain English
↓
Question embedded → similarity search → top 4 chunks retrieved
↓
Groq LLaMA 3.3 synthesises answer from retrieved chunks
↓
Answer shown with source sections highlighted

## Features

- Upload multiple PDFs at once
- Ask questions in plain English
- See exactly which section of the document answered your question
- Chat history maintained in session
- Handles questions not covered in the uploaded documents gracefully

## What I learned building this

- Chunk size and overlap significantly affect retrieval quality — 1000/200 worked better than smaller chunks for dense academic text
- Returning source documents is critical for trust — users need to verify the answer
- Modern LangChain (1.3.7) uses LCEL chains, not the older RetrievalQA — had to rewrite the chain using runnables and a custom prompt
- Switched from OpenAI to Groq + HuggingFace for a fully free, locally-running embedding pipeline with fast inference

## Tech stack

- LangChain (LCEL chains, document loaders, text splitters)
- Groq LLaMA 3.3 (answer generation)
- HuggingFace all-MiniLM-L6-v2 (embeddings, runs locally)
- FAISS (vector store and similarity search)
- Streamlit (UI and deployment)
- Python

## Run locally

```bash
git clone https://github.com/Saiajaykumar12/education-rag-chatbot
cd education-rag-chatbot
pip install -r requirements.txt
```

Add your Groq key to `.env`:
GROQ_API_KEY=your_key
```bash
streamlit run app.py
```

Live link : https://education-rag-chatbot-wooy8ve7nsscu7avtekufs.streamlit.app/

## Related projects

- [LangGraph Agentic RAG](https://github.com/Saiajaykumar12/langgraph-agentic-rag) — adds an autonomous agent layer on top of RAG
- [SQL AI Chatbot](https://github.com/Saiajaykumar12/sql-ai-chatbot-langchain) — same natural language interface for databases
