"""Embedding Service - Vector embedding for medical text using HuggingFace models."""

import os
import numpy as np
from typing import List, Union, Optional
from pathlib import Path
import pickle

# Try to import sentence-transformers, fall back to simple implementation if not available
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Using fallback embedding.")


class EmbeddingService:
    """
    Service for generating text embeddings using medical-domain models.
    
    Recommended models for medical text:
    - 'pritamdeka/S-PubMedBert-MS-MARCO' - PubMedBERT fine-tuned for retrieval
    - 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    - 'sentence-transformers/all-MiniLM-L6-v2' - General purpose, fast
    - 'BAAI/bge-base-en-v1.5' - High quality general embeddings
    """
    
    # Default medical embedding model
    DEFAULT_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
    FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(
        self, 
        model_name: str = None,
        cache_dir: str = None,
        device: str = None
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the HuggingFace model to use
            cache_dir: Directory to cache the model
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", self.DEFAULT_MODEL)
        self.cache_dir = cache_dir or os.getenv("EMBEDDING_CACHE_DIR", "./cache/embeddings")
        self.device = device
        self.model = None
        self.embedding_dim = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        if not HAS_SENTENCE_TRANSFORMERS:
            print("sentence-transformers not available. Using TF-IDF fallback.")
            self._setup_fallback()
            return
            
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print(f"Trying fallback model: {self.FALLBACK_MODEL}")
            try:
                self.model = SentenceTransformer(
                    self.FALLBACK_MODEL,
                    cache_folder=self.cache_dir,
                    device=self.device
                )
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.model_name = self.FALLBACK_MODEL
            except Exception as e2:
                print(f"Fallback model also failed: {e2}")
                self._setup_fallback()
    
    def _setup_fallback(self):
        """Setup TF-IDF based fallback embedding."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.fallback_vectorizer = TfidfVectorizer(
                max_features=768,
                ngram_range=(1, 2),
                stop_words='english'
            )
            self.embedding_dim = 768
            self.model = None
            print("Using TF-IDF fallback embedding (dimension: 768)")
        except ImportError:
            raise RuntimeError("Neither sentence-transformers nor sklearn available for embeddings")
    
    def embed(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for input text(s).
        
        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            numpy array of embeddings, shape (n_texts, embedding_dim)
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        if self.model is not None:
            # Use sentence-transformers
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
        else:
            # Use fallback TF-IDF
            embeddings = self._fallback_embed(texts, normalize)
        
        if single_input:
            return embeddings[0]
        return embeddings
    
    def _fallback_embed(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Generate embeddings using TF-IDF fallback."""
        # Fit if not already fitted
        if not hasattr(self.fallback_vectorizer, 'vocabulary_'):
            self.fallback_vectorizer.fit(texts)
        
        # Transform
        tfidf_matrix = self.fallback_vectorizer.transform(texts).toarray()
        
        # Pad or truncate to embedding_dim
        if tfidf_matrix.shape[1] < self.embedding_dim:
            padding = np.zeros((tfidf_matrix.shape[0], self.embedding_dim - tfidf_matrix.shape[1]))
            embeddings = np.hstack([tfidf_matrix, padding])
        else:
            embeddings = tfidf_matrix[:, :self.embedding_dim]
        
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / norms
        
        return embeddings.astype(np.float32)
    
    def embed_question_with_options(
        self, 
        question: str, 
        options: dict = None,
        include_options: bool = True
    ) -> np.ndarray:
        """
        Embed a question with its options for better semantic matching.
        
        Args:
            question: The question text
            options: Dictionary of options (e.g., {"A": "option1", "B": "option2"})
            include_options: Whether to include options in the embedding
            
        Returns:
            numpy array of embedding
        """
        if include_options and options:
            # Combine question with options
            options_text = " ".join([f"{k}: {v}" for k, v in options.items()])
            full_text = f"{question} Options: {options_text}"
        else:
            full_text = question
            
        return self.embed(full_text)
    
    def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Normalize if not already
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray, 
        filepath: str,
        metadata: dict = None
    ):
        """
        Save embeddings to disk.
        
        Args:
            embeddings: numpy array of embeddings
            filepath: Path to save file
            metadata: Optional metadata to save alongside embeddings
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'embeddings': embeddings,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'metadata': metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Saved embeddings to {filepath}")
    
    @staticmethod
    def load_embeddings(filepath: str) -> tuple:
        """
        Load embeddings from disk.
        
        Args:
            filepath: Path to saved embeddings
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        return save_dict['embeddings'], save_dict.get('metadata', {})


# Singleton instance for reuse
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(model_name: str = None) -> EmbeddingService:
    """
    Get or create a singleton EmbeddingService instance.
    
    Args:
        model_name: Optional model name (only used if creating new instance)
        
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name=model_name)
    
    return _embedding_service

