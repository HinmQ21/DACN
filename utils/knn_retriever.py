"""KNN Retriever - K-Nearest Neighbors retrieval for few-shot example selection."""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle

from utils.embedding_service import EmbeddingService, get_embedding_service


class KNNRetriever:
    """
    K-Nearest Neighbors retriever for dynamic few-shot example selection.
    Uses vector similarity to find the most relevant examples from a knowledge base.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService = None,
        index_path: str = None,
        use_faiss: bool = True
    ):
        """
        Initialize the KNN retriever.
        
        Args:
            embedding_service: EmbeddingService instance (uses singleton if None)
            index_path: Path to pre-built index
            use_faiss: Whether to use FAISS for efficient search (if available)
        """
        self.embedding_service = embedding_service or get_embedding_service()
        self.index_path = index_path or os.getenv(
            "KNN_INDEX_PATH", 
            "./data/knowledge_base/train_index.pkl"
        )
        
        # Storage for examples
        self.examples: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        
        # Try to use FAISS for efficient search
        self.use_faiss = use_faiss
        if use_faiss:
            try:
                import faiss
                self.faiss = faiss
                print("FAISS available for efficient similarity search")
            except ImportError:
                self.faiss = None
                self.use_faiss = False
                print("FAISS not available, using numpy-based search")
        
        # Try to load existing index
        if os.path.exists(self.index_path):
            self.load_index(self.index_path)
    
    def build_index(
        self,
        examples: List[Dict[str, Any]],
        text_key: str = "question",
        include_options: bool = True,
        batch_size: int = 32,
        show_progress: bool = True
    ):
        """
        Build the KNN index from a list of examples.
        
        Args:
            examples: List of example dictionaries containing questions and answers
            text_key: Key in dictionary containing the text to embed
            include_options: Whether to include options in embedding
            batch_size: Batch size for embedding
            show_progress: Show progress during embedding
        """
        self.examples = examples
        
        # Prepare texts for embedding
        texts = []
        for ex in examples:
            question = ex.get(text_key, "")
            options = ex.get("options", {})
            
            if include_options and options:
                options_text = " ".join([f"{k}: {v}" for k, v in options.items()])
                full_text = f"{question} Options: {options_text}"
            else:
                full_text = question
            
            texts.append(full_text)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} examples...")
        self.embeddings = self.embedding_service.embed(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=True
        )
        
        # Build FAISS index if available
        if self.use_faiss and self.faiss is not None:
            self._build_faiss_index()
        
        print(f"Index built with {len(self.examples)} examples")
    
    def _build_faiss_index(self):
        """Build FAISS index for efficient similarity search."""
        if self.embeddings is None:
            return
            
        dim = self.embeddings.shape[1]
        
        # Use Inner Product for cosine similarity (embeddings are normalized)
        self.faiss_index = self.faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings.astype(np.float32))
        
        print(f"FAISS index built with dimension {dim}")
    
    def save_index(self, filepath: str = None):
        """
        Save the index to disk.
        
        Args:
            filepath: Path to save the index
        """
        filepath = filepath or self.index_path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'examples': self.examples,
            'embeddings': self.embeddings,
            'model_name': self.embedding_service.model_name,
            'embedding_dim': self.embedding_service.embedding_dim
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str = None):
        """
        Load the index from disk.
        
        Args:
            filepath: Path to the saved index
        """
        filepath = filepath or self.index_path
        
        if not os.path.exists(filepath):
            print(f"Index file not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.examples = save_dict['examples']
        self.embeddings = save_dict['embeddings']
        
        # Check for model mismatch
        saved_model = save_dict.get('model_name', 'unknown')
        current_model = self.embedding_service.model_name
        if saved_model != current_model:
            print(f"⚠️  WARNING: Embedding model mismatch!")
            print(f"   Index was built with: {saved_model}")
            print(f"   Current model: {current_model}")
            print(f"   This may cause poor similarity results.")
            print(f"   Consider rebuilding the index or installing the correct model.")
        
        # Rebuild FAISS index if available
        if self.use_faiss and self.faiss is not None:
            self._build_faiss_index()
        
        print(f"Index loaded: {len(self.examples)} examples, dim={self.embeddings.shape[1]}")
        return True
    
    def get_similar_examples(
        self,
        query: str,
        k: int = 3,
        options: Dict[str, str] = None,
        include_options_in_query: bool = True,
        min_similarity: float = 0.0,
        exclude_indices: List[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the k most similar examples to the query.
        
        Args:
            query: Query text (question)
            k: Number of examples to retrieve
            options: Query options (if multiple choice)
            include_options_in_query: Whether to include options in query embedding
            min_similarity: Minimum similarity threshold
            exclude_indices: Indices to exclude from results
            
        Returns:
            List of similar examples with similarity scores
        """
        if self.embeddings is None or len(self.examples) == 0:
            print("Warning: Index not built. Returning empty list.")
            return []
        
        # Prepare query text
        if include_options_in_query and options:
            options_text = " ".join([f"{k}: {v}" for k, v in options.items()])
            query_text = f"{query} Options: {options_text}"
        else:
            query_text = query
        
        # Embed query
        query_embedding = self.embedding_service.embed(query_text, normalize=True)
        
        # Find similar examples
        if self.use_faiss and self.faiss_index is not None:
            similarities, indices = self._search_faiss(query_embedding, k * 2)  # Get extra in case of filtering
        else:
            similarities, indices = self._search_numpy(query_embedding, k * 2)
        
        # Filter and format results
        results = []
        exclude_set = set(exclude_indices) if exclude_indices else set()
        
        for sim, idx in zip(similarities, indices):
            if idx in exclude_set:
                continue
            if sim < min_similarity:
                continue
                
            example = self.examples[idx].copy()
            example['similarity_score'] = float(sim)
            example['index'] = int(idx)
            results.append(example)
            
            if len(results) >= k:
                break
        
        return results
    
    def _search_faiss(
        self, 
        query_embedding: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS index."""
        query = query_embedding.reshape(1, -1).astype(np.float32)
        similarities, indices = self.faiss_index.search(query, min(k, len(self.examples)))
        return similarities[0], indices[0]
    
    def _search_numpy(
        self, 
        query_embedding: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search using numpy (fallback)."""
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        k = min(k, len(similarities))
        indices = np.argsort(similarities)[-k:][::-1]
        
        return similarities[indices], indices
    
    def format_examples_for_prompt(
        self,
        examples: List[Dict[str, Any]],
        include_reasoning: bool = True,
        include_answer: bool = True,
        max_examples: int = None
    ) -> str:
        """
        Format retrieved examples for use in a prompt.
        
        Args:
            examples: List of example dictionaries
            include_reasoning: Whether to include reasoning/explanation
            include_answer: Whether to include the answer
            max_examples: Maximum number of examples to include
            
        Returns:
            Formatted string for prompt
        """
        if max_examples:
            examples = examples[:max_examples]
        
        formatted_parts = []
        
        for i, ex in enumerate(examples, 1):
            parts = [f"Example {i}:"]
            
            # Question
            parts.append(f"Question: {ex.get('question', 'N/A')}")
            
            # Options
            options = ex.get('options', {})
            if options:
                options_str = "\n".join([f"  {k}: {v}" for k, v in options.items()])
                parts.append(f"Options:\n{options_str}")
            
            # Reasoning (if available and requested)
            if include_reasoning:
                reasoning = ex.get('reasoning', ex.get('cot_reasoning', ''))
                if reasoning:
                    parts.append(f"Reasoning: {reasoning}")
            
            # Answer
            if include_answer:
                answer = ex.get('answer', ex.get('answer_idx', 'N/A'))
                parts.append(f"Answer: {answer}")
            
            formatted_parts.append("\n".join(parts))
        
        return "\n\n---\n\n".join(formatted_parts)


def load_training_examples(
    filepath: str,
    max_examples: int = None
) -> List[Dict[str, Any]]:
    """
    Load training examples from a JSONL file.
    
    Args:
        filepath: Path to JSONL file
        max_examples: Maximum number of examples to load
        
    Returns:
        List of example dictionaries
    """
    examples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            
            try:
                example = json.loads(line.strip())
                examples.append(example)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}")
                continue
    
    print(f"Loaded {len(examples)} examples from {filepath}")
    return examples


# Singleton instance
_knn_retriever: Optional[KNNRetriever] = None


def get_knn_retriever() -> KNNRetriever:
    """Get or create a singleton KNNRetriever instance."""
    global _knn_retriever
    
    if _knn_retriever is None:
        _knn_retriever = KNNRetriever()
    
    return _knn_retriever

