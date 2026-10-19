# DocuMind RAG — Streamlit UI

A single-page Streamlit frontend for a document Q&A RAG application using LangChain and Ollama.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Backend integration

This project contains **UI functionality only**. `DEMO_MODE = True` simulates processing and the Q&A response; it **does not read, embed, index, or retrieve document content**.

To connect your backend, open `app.py`:

1. Implement `process_documents(uploaded_files)` using your own loaders, chunking, embedding and vector storage. Save the retriever to `st.session_state["rag_retriever"]`.
2. Implement `generate_answer(question)` using the saved retriever and your Ollama LLM. Return a tuple: `(answer_text, context_chunks)`. Each context chunk should be a dictionary with `source`, optional `page`, and `content` fields.
3. Change `DEMO_MODE` to `False`.

The **Get Answer** button becomes available only after **Process** finishes successfully, and it is disabled again if the uploaded files change. The context section uses Streamlit's native `st.expander("Show Context")`.
