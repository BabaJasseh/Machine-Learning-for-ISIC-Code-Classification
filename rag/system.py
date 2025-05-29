"""
RAG (Retrieval-Augmented Generation) System for ISIC Classification
"""

import os
import requests
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
from typing import List
from config.settings import RAG_CONFIG, LANGUAGE_NAMES

class RAGSystem:
    """RAG system for ISIC manual integration"""
    
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.ollama_available = False
        self.initialize_rag()
    
    def initialize_rag(self):
        """Initialize RAG components"""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(RAG_CONFIG["embedding_model"])
            
            # Initialize Chroma client
            self.chroma_client = chromadb.PersistentClient(path=RAG_CONFIG["chroma_persist_path"])
            self.collection = self.chroma_client.get_or_create_collection(name="isic_manual")
            
            # Check if documents are already loaded
            if self.collection.count() == 0:
                self.load_pdf_documents()
            
            # Test Ollama connection
            self.test_ollama_connection()
            
        except Exception as e:
            st.warning(f"RAG initialization failed: {e}. Classification will work without RAG features.")
    
    def test_ollama_connection(self):
        """Test if Ollama is available"""
        try:
            response = requests.get(f"{RAG_CONFIG['ollama_base_url']}/api/tags", timeout=5)
            if response.status_code == 200:
                self.ollama_available = True
            else:
                self.ollama_available = False
        except Exception as e:
            self.ollama_available = False
    
    def load_pdf_documents(self):
        """Load and process PDF documents"""
        try:
            if not os.path.exists(RAG_CONFIG["pdf_path"]):
                return
            
            # Extract text from PDF
            with open(RAG_CONFIG["pdf_path"], 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page.extract_text()
            
            # Split text into chunks
            chunks = self.split_text(text, chunk_size=1000, chunk_overlap=100)
            
            # Generate embeddings and store
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    embedding = self.embedding_model.encode(chunk).tolist()
                    self.collection.upsert(
                        ids=[f"chunk_{i}"],
                        documents=[chunk],
                        embeddings=[embedding]
                    )
            
        except Exception as e:
            pass
    
    def split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - chunk_overlap
        return chunks
    
    def query_documents(self, query: str, n_results: int = 3) -> List[str]:
        """Query relevant documents"""
        if not self.collection:
            return []
        
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            return [doc for sublist in results["documents"] for doc in sublist]
        except Exception as e:
            return []
    
    def generate_explanation(self, isic_code: str, description: str, business_description: str, target_language: str = "en") -> str:
        """Generate explanation using RAG and Ollama"""
        if not self.ollama_available:
            return f"ISIC Code {isic_code}: {description}"
        
        try:
            # Query relevant context
            query = f"ISIC {isic_code} {description} {business_description}"
            relevant_chunks = self.query_documents(query, n_results=2)
            
            # Prepare context
            context = "\n\n".join(relevant_chunks) if relevant_chunks else "No specific context available."
            
            # Create prompt with language specification
            language_instruction = ""
            if target_language != "en":
                lang_name = LANGUAGE_NAMES.get(target_language, target_language)
                language_instruction = f"Please respond in {lang_name}. "
            
            # Create prompt
            prompt = f"""
            {language_instruction}Based on the ISIC manual context below, provide a detailed explanation for why the business activity "{business_description}" 
            has been classified as ISIC code {isic_code} ({description}).
            
            Context from ISIC Manual:
            {context}
            
            Please provide:
            1. Why this classification is appropriate
            2. Key characteristics of this ISIC category
            3. Any relevant examples or clarifications from the manual
            
            Keep the response concise but informative (max 300 words).
            """
            
            # Call Ollama
            response = requests.post(
                f"{RAG_CONFIG['ollama_base_url']}/api/generate",
                json={
                    "model": RAG_CONFIG["model_name"],
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No explanation generated.")
            else:
                return f"Error generating explanation: {response.status_code}"
                
        except Exception as e:
            return f"Error generating explanation: {e}"

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with caching"""
    return RAGSystem()