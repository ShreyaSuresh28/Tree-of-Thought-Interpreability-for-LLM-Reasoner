"""
RAG Engine for document processing and retrieval.
Handles PDF extraction, chunking, embedding generation, and vector search.
"""

import PyPDF2
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from utils import chunk_text_by_tokens, clean_text, count_tokens

class RAGEngine:
    """Retrieval-Augmented Generation engine for document processing."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG Engine with embedding model.
        
        Args:
            embedding_model: Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(embedding_model)
        self.embeddings = None
        self.index = None
        self.chunks = []
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
    def extract_pdf_text(self, pdf_file) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_file: Uploaded PDF file object
            
        Returns:
            Extracted text as string
        """
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")
        
        return clean_text(text)
    
    def extract_text_file(self, text_file) -> str:
        """
        Extract text from text file.
        
        Args:
            text_file: Uploaded text file object
            
        Returns:
            Extracted text as string
        """
        try:
            text = text_file.read().decode("utf-8")
            return clean_text(text)
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")
    
    def process_document(self, file) -> List[str]:
        """
        Process uploaded document: extract text and create chunks.
        
        Args:
            file: Uploaded file object
            
        Returns:
            List of text chunks
        """
        # Extract text based on file type
        if file.type == "application/pdf":
            text = self.extract_pdf_text(file)
        elif file.type == "text/plain":
            text = self.extract_text_file(file)
        else:
            raise ValueError("Unsupported file type. Please upload PDF or TXT file.")
        
        # Create chunks
        self.chunks = chunk_text_by_tokens(text, max_tokens=800, overlap=100)
        
        # Create embeddings for chunks
        self.create_embeddings()
        
        # Build FAISS index
        self.build_faiss_index()
        
        return self.chunks
    
    def create_embeddings(self) -> None:
        """Create embeddings for all text chunks."""
        if not self.chunks:
            raise ValueError("No chunks available. Process a document first.")
        
        self.embeddings = self.model.encode(self.chunks)
    
    def build_faiss_index(self) -> None:
        """Build FAISS index from embeddings."""
        if self.embeddings is None:
            raise ValueError("No embeddings available. Create embeddings first.")
        
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(np.array(self.embeddings).astype('float32'))
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of dictionaries containing chunks and their scores
        """
        if self.index is None:
            raise ValueError("No index available. Process a document first.")
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            min(top_k, len(self.chunks))
        )
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'chunk': self.chunks[idx],
                'score': float(1 / (1 + distances[0][i])),  # Convert distance to similarity
                'index': int(idx)
            })
        
        return results
    
    def get_context_for_query(self, query: str, top_k: int = 5) -> str:
        """
        Get concatenated context from retrieved chunks.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            Concatenated context string
        """
        chunks = self.retrieve_relevant_chunks(query, top_k)
        context = "\n\n".join([chunk['chunk'] for chunk in chunks])
        return context
    
    def save_index(self, path: str = "faiss_index.pkl") -> None:
        """Save FAISS index and chunks to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        data = {
            'index': self.index,
            'chunks': self.chunks,
            'embeddings': self.embeddings
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_index(self, path: str = "faiss_index.pkl") -> None:
        """Load FAISS index and chunks from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.index = data['index']
        self.chunks = data['chunks']
        self.embeddings = data['embeddings']