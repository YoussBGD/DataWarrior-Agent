#!/usr/bin/env python3
"""
Semantic Search for small models (Ollama)
Finds the most relevant macros for a query

Uses sentence-transformers (all-MiniLM-L6-v2) to generate
embeddings and calculate cosine similarity.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class SemanticSearch:
    """Semantic search with sentence-transformers"""
    
    def __init__(self, cache_path: Path = None):
        self.cache_path = cache_path
        self.model = None
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self._load_model()
    
    def _load_model(self):
        """Loads the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            # Force CPU to avoid CUDA errors
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            print("[SemanticSearch] Model loaded: all-MiniLM-L6-v2")
        except ImportError:
            print("⚠️ sentence-transformers not installed. Run: pip install sentence-transformers")
            self.model = None
    
    def embed(self, text: str) -> np.ndarray:
        """Generates the embedding for a text"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of texts"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.encode(texts, convert_to_numpy=True)
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculates the cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def build_tool_index(self, tools: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Builds an embeddings index for the tools
        
        Args:
            tools: Dict {tool_name: description}
            
        Returns:
            Dict {tool_name: embedding}
        """
        if self.model is None:
            return {}
        
        print(f"[SemanticSearch] Building index for {len(tools)} tools...")
        
        for name, description in tools.items():
            text = f"{name}: {description}"
            self.embeddings_cache[name] = self.embed(text)
        
        return self.embeddings_cache
    
    def search(self, query: str, tools: Dict[str, str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Finds the most relevant tools for a query
        
        Args:
            query: User query
            tools: Dict {tool_name: description}
            top_k: Number of results to return
            
        Returns:
            List of (tool_name, score) sorted by relevance
        """
        if self.model is None:
            # Fallback: return all tools
            print("[SemanticSearch] Model not loaded, returning all tools")
            return [(name, 1.0) for name in list(tools.keys())[:top_k]]
        
        # Ensure the index is built
        if not self.embeddings_cache or set(self.embeddings_cache.keys()) != set(tools.keys()):
            self.build_tool_index(tools)
        
        # Embed the query
        query_embedding = self.embed(query)
        
        # Calculate similarities
        scores = []
        for name, tool_embedding in self.embeddings_cache.items():
            score = self.cosine_similarity(query_embedding, tool_embedding)
            scores.append((name, float(score)))
        
        # Sort by descending score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def save_cache(self, path: Path):
        """Saves the embeddings cache"""
        with open(path, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
        print(f"[SemanticSearch] Cache saved to {path}")
    
    def load_cache(self, path: Path) -> bool:
        """Loads the embeddings cache"""
        if path.exists():
            with open(path, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
            print(f"[SemanticSearch] Cache loaded from {path}")
            return True
        return False


# ============================================
# HELPER FUNCTION
# ============================================

def get_relevant_tools(query: str, tools: Dict[str, str], top_k: int = 5) -> List[str]:
    """
    Shortcut function to find relevant tools
    
    Args:
        query: User query
        tools: Dict of tools {name: description}
        top_k: Number of tools to return
        
    Returns:
        List of relevant tool names
    """
    searcher = SemanticSearch()
    results = searcher.search(query, tools, top_k=top_k)
    return [name for name, score in results]
