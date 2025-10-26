import os
import re
import json
import uuid
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

class RAGIndex:
    def __init__(self, index_dir="rag_index", embed_model_name=EMBED_MODEL_NAME):
        os.makedirs(index_dir, exist_ok=True)
        self.index_dir = index_dir
        self.model = SentenceTransformer(embed_model_name)
        self.index_path = os.path.join(index_dir, "faiss.index")
        self.meta_path = os.path.join(index_dir, "meta.json")
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(EMBED_DIM)
            self.meta = {"ids": [], "docs": []}
            self._save()

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def _normalize(self, vectors):
        import numpy as np
        arr = np.array(vectors, dtype="float32")
        norms = (arr**2).sum(axis=1, keepdims=True) ** 0.5
        norms[norms == 0] = 1.0
        return arr / norms

    def add_documents(self, docs: List[Dict]):
        """
        docs: list of {"source": "filename#page/section", "text": "chunk text"}
        """
        import numpy as np
        texts = [d["text"] for d in docs]
        ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = self._normalize(embeddings)

        self.index.add(embeddings)
        for i, d in enumerate(docs):
            self.meta["ids"].append(ids[i])
            self.meta["docs"].append({
                "id": ids[i],
                "source": d.get("source", ""),
                "text": d["text"][:5000]
            })
        self._save()

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        import numpy as np
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = self._normalize(q_emb)
        D, I = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.meta["docs"]):
                continue
            doc_meta = self.meta["docs"][idx]
            results.append({
                "id": doc_meta["id"],
                "source": doc_meta["source"],
                "text": doc_meta["text"],
                "score": float(score)
            })
        return results

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

def markdown_to_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_text_into_chunks(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) + 1 <= chunk_size:
            cur = (cur + "\n\n" + p).strip() if cur else p
        else:
            if cur:
                chunks.append(cur)
            if len(p) > chunk_size:
                for i in range(0, len(p), chunk_size - overlap):
                    chunks.append(p[i:i + chunk_size])
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)

    final = []
    for i, c in enumerate(chunks):
        if i == 0:
            final.append(c)
        else:
            prev = final[-1]
            overlap_text = (prev[-overlap:] + " " + c[:overlap]).strip()
            final.append(overlap_text + "\n\n" + c)
    return final

def ingest_file_to_docs(filepath: str) -> List[Dict]:
    ext = filepath.lower().split('.')[-1]
    if ext in ("pdf",):
        text = pdf_to_text(filepath)
    elif ext in ("md", "markdown", "txt"):
        text = markdown_to_text(filepath)
    else:
        try:
            text = markdown_to_text(filepath)
        except:
            text = ""
    chunks = split_text_into_chunks(text)
    docs = []
    for i, c in enumerate(chunks):
        docs.append({
            "source": f"{os.path.basename(filepath)}#chunk{i+1}",
            "text": c
        })
    return docs
