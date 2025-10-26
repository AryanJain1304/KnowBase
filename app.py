import os
import re
import streamlit as st
from ollama_client import query_ollama_stream
from rag_index import RAGIndex, ingest_file_to_docs

def get_first_sentences(text, num_sentences=2):
    """Return first num_sentences of a text."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:num_sentences])

st.set_page_config(page_title="KnowBase", page_icon="🤖", layout="centered")
st.title("KnowBase")
st.markdown("Turn your files into an interactive knowledge base.")

st.sidebar.header("Document Corpus")
uploaded = st.sidebar.file_uploader("Upload PDF / Markdown", accept_multiple_files=True, type=["pdf", "md", "txt"])
index_button = st.sidebar.button("Ingest & Build Index")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if index_button and uploaded:
    st.sidebar.info("Saving files and building index...")
    saved_paths = []
    for f in uploaded:
        save_path = os.path.join(UPLOAD_DIR, f.name)
        with open(save_path, "wb") as out:
            out.write(f.getbuffer())
        saved_paths.append(save_path)

    rag = RAGIndex()
    all_docs = []
    for p in saved_paths:
        docs = ingest_file_to_docs(p)
        all_docs.extend(docs)

    all_docs = [d for d in all_docs if len(d["text"].strip()) > 20]

    with st.spinner("Embedding chunks..."):
        BATCH = 256
        for i in range(0, len(all_docs), BATCH):
            batch = all_docs[i:i+BATCH]
            rag.add_documents(batch)

    st.sidebar.success(f"Ingested {len(all_docs)} chunks into index.")

st.markdown("""
    <style>
    .chat-container { max-width: none; margin: 0; }
    .message { padding: 12px 16px; border-radius: 12px; margin: 4px 0; display: inline-block; word-wrap: break-word; white-space: pre-wrap; }
    .user { background-color: #2E2E2E; text-align: right; float: right; clear: both; color: #e0e0e0; }
    .assistant { padding-left: 0; text-align: justify; float: left; clear: both; color: #e0e0e0; }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_querying" not in st.session_state:
    st.session_state.is_querying = False
if "pending_user_input" not in st.session_state:
    st.session_state.pending_user_input = None
if "citation_maps" not in st.session_state:
    st.session_state.citation_maps = []

with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for idx, msg in enumerate(st.session_state.messages):
        role_class = "user" if msg["role"] == "user" else "assistant"
        st.markdown(f'<div class="message {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

        if msg["role"] == "assistant" and idx < len(st.session_state.citation_maps):
            citation_map = st.session_state.citation_maps[idx]
            for i, snippet_text in citation_map.items():
                st.markdown(f"[{i}] {snippet_text} ...")
    st.markdown('</div>', unsafe_allow_html=True)

user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.pending_user_input = user_input
    st.session_state.is_querying = True
    st.rerun()

if st.session_state.is_querying and st.session_state.pending_user_input:
    user_input = st.session_state.pending_user_input
    rag = RAGIndex()

    top_k = 5
    results = rag.retrieve(user_input, k=top_k)

    context_pieces = []
    citation_map = {}
    for i, r in enumerate(results, start=1):
        truncated_context = r["text"][:1200]
        context_pieces.append(f"[{i}] Source: {r['source']}\nScore: {r['score']:.4f}\n{truncated_context}\n")
        citation_map[i] = get_first_sentences(r["text"], 2)

    context_block = "\n\n---\n\n".join(context_pieces)
    st.session_state.citation_maps.append({})

    placeholder = st.empty()
    placeholder.markdown('<div class="message assistant">Thinking...</div>', unsafe_allow_html=True)

    prompt = f"""
You are a helpful assistant that must answer using ONLY the provided context snippets.
If the answer is not contained in the context, say "I don't know based on the provided documents."
Cite each factual statement with the snippet number like [1], [2].

CONTEXT:
{context_block}

USER QUESTION:
{user_input}

INSTRUCTIONS:
- Answer concisely and directly.
- Provide the final answer, then list the citations used (e.g., "Sources: [1], [3]").
- After the answer, give a brief confidence estimate (0.0-1.0).
Assistant:
"""

    assistant_response = ""
    first_chunk = True
    for chunk in query_ollama_stream(prompt):
        if first_chunk:
            chunk = chunk.lstrip()
            first_chunk = False
        assistant_response += chunk
        placeholder.markdown(f'<div class="message assistant">{assistant_response}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role":"assistant", "content": assistant_response})
    st.session_state.citation_maps.append(citation_map)

    st.markdown("**Citations:**")
    for i, snippet_text in citation_map.items():
        st.markdown(f"[{i}] {snippet_text} ...")

    st.session_state.citation_maps.append({})

    st.session_state.is_querying = False
    st.session_state.pending_user_input = None