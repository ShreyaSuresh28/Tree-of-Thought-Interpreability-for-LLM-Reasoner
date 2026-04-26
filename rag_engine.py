"""
Advanced RAG Engine with re-ranking and improved retrieval.
"""

import PyPDF2
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util
import faiss
import pickle
from utils import chunk_text_by_tokens, clean_text


class RAGEngine:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(embedding_model)
        self.embeddings = None
        self.index = None
        self.chunks = []
        self.embedding_dim = 384

    # =========================
    # TEXT EXTRACTION
    # =========================
    def extract_pdf_text(self, pdf_file) -> str:
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        for page in pdf_reader.pages:
            text += page.extract_text()

        return clean_text(text)

    def extract_text_file(self, text_file) -> str:
        return clean_text(text_file.read().decode("utf-8"))

    # =========================
    # DOCUMENT PROCESSING
    # =========================
    def process_document(self, file) -> List[str]:

        if file.type == "application/pdf":
            text = self.extract_pdf_text(file)
        else:
            text = self.extract_text_file(file)

        # 🔥 Better chunking
        self.chunks = chunk_text_by_tokens(text, max_tokens=500, overlap=100)

        self.create_embeddings()
        self.build_faiss_index()

        return self.chunks

    # =========================
    # EMBEDDINGS
    # =========================
    def create_embeddings(self):

        if not self.chunks:
            raise ValueError("No chunks available")

        self.embeddings = self.model.encode(self.chunks, convert_to_numpy=True)

    def build_faiss_index(self):

        self.index = faiss.IndexFlatIP(self.embedding_dim)  # cosine similarity

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)

        self.index.add(self.embeddings.astype("float32"))

    # =========================
    # RETRIEVAL
    # =========================
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5):

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding.astype("float32"), top_k)

        results = []

        for i, idx in enumerate(indices[0]):
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[0][i]),
                "index": int(idx)
            })

        # 🔥 Re-ranking step (IMPORTANT)
        results = self._rerank_results(query, results)

        return results

    # =========================
    # RE-RANKING (VERY IMPORTANT 🔥)
    # =========================
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:

        query_emb = self.model.encode(query, convert_to_tensor=True)

        for item in results:
            chunk_emb = self.model.encode(item["chunk"], convert_to_tensor=True)
            similarity = util.cos_sim(query_emb, chunk_emb).item()

            # Combine FAISS + semantic score
            item["score"] = (item["score"] * 0.6) + (similarity * 0.4)

        # Sort again
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results

    # =========================
    # CONTEXT BUILDING
    # =========================
    def get_context_for_query(self, query: str, top_k: int = 5) -> str:

        chunks = self.retrieve_relevant_chunks(query, top_k)

        # 🔥 Smart context selection
        context = ""
        total_length = 0

        for chunk in chunks:
            if total_length + len(chunk["chunk"]) > 1500:
                break

            context += chunk["chunk"] + "\n\n"
            total_length += len(chunk["chunk"])

        return context.strip()

    # =========================
    # SAVE / LOAD
    # =========================
    def save_index(self, path: str = "faiss_index.pkl"):

        data = {
            "index": self.index,
            "chunks": self.chunks,
            "embeddings": self.embeddings
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_index(self, path: str = "faiss_index.pkl"):

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.index = data["index"]
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]
