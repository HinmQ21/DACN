"""Configuration management for the medical QA system."""

import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Config:
    """Configuration class for managing environment variables and settings."""
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
    # Model Configuration - Default
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
    
    # Model Configuration - Per Agent (optional overrides)
    COORDINATOR_MODEL: str = os.getenv("COORDINATOR_MODEL", "")  # Uses default if empty
    REASONING_MODEL: str = os.getenv("REASONING_MODEL", "")
    VALIDATOR_MODEL: str = os.getenv("VALIDATOR_MODEL", "")
    ANSWER_GENERATOR_MODEL: str = os.getenv("ANSWER_GENERATOR_MODEL", "")
    WEB_SEARCH_MODEL: str = os.getenv("WEB_SEARCH_MODEL", "")
    
    # Temperature Configuration - Per Agent (optional overrides)
    COORDINATOR_TEMPERATURE: float = float(os.getenv("COORDINATOR_TEMPERATURE", "0"))  # Uses default if 0
    REASONING_TEMPERATURE: float = float(os.getenv("REASONING_TEMPERATURE", "0"))
    VALIDATOR_TEMPERATURE: float = float(os.getenv("VALIDATOR_TEMPERATURE", "0"))
    ANSWER_GENERATOR_TEMPERATURE: float = float(os.getenv("ANSWER_GENERATOR_TEMPERATURE", "0"))
    WEB_SEARCH_TEMPERATURE: float = float(os.getenv("WEB_SEARCH_TEMPERATURE", "0"))
    
    # Benchmark Configuration
    MAX_SAMPLES: int = int(os.getenv("MAX_SAMPLES", "100"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "5"))
    
    # Search Configuration
    MAX_SEARCH_RESULTS: int = 5
    PUBMED_MAX_RESULTS: int = 5
    
    # =====================================================
    # Medprompt Configuration
    # =====================================================
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", 
        "pritamdeka/S-PubMedBert-MS-MARCO"
    )
    EMBEDDING_CACHE_DIR: str = os.getenv("EMBEDDING_CACHE_DIR", "./cache/embeddings")
    
    # Few-shot Selection Configuration
    FEW_SHOT_K: int = int(os.getenv("FEW_SHOT_K", "3"))  # Number of similar examples
    FEW_SHOT_MIN_SIMILARITY: float = float(os.getenv("FEW_SHOT_MIN_SIMILARITY", "0.3"))
    KNN_INDEX_PATH: str = os.getenv("KNN_INDEX_PATH", "./data/knowledge_base/medqa/train_index.pkl")
    ENABLE_FEW_SHOT: bool = os.getenv("ENABLE_FEW_SHOT", "true").lower() == "true"
    
    # Chain-of-Thought Configuration
    ENABLE_COT: bool = os.getenv("ENABLE_COT", "true").lower() == "true"
    COT_DETAILED: bool = os.getenv("COT_DETAILED", "true").lower() == "true"
    
    # Choice Shuffling Ensemble Configuration
    ENABLE_ENSEMBLE: bool = os.getenv("ENABLE_ENSEMBLE", "true").lower() == "true"
    ENSEMBLE_VARIANTS: int = int(os.getenv("ENSEMBLE_VARIANTS", "5"))  # Number of shuffled variants
    ENSEMBLE_CONFIDENCE_THRESHOLD: float = float(os.getenv("ENSEMBLE_CONFIDENCE_THRESHOLD", "0.6"))
    
    # Self-consistency Configuration (for high-stakes questions)
    ENABLE_SELF_CONSISTENCY: bool = os.getenv("ENABLE_SELF_CONSISTENCY", "false").lower() == "true"
    SELF_CONSISTENCY_SAMPLES: int = int(os.getenv("SELF_CONSISTENCY_SAMPLES", "3"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required")
        if not cls.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is required")
        return True
    
    @classmethod
    def get_llm_config(cls, agent_name: str = None) -> dict:
        """
        Get LLM configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent ('coordinator', 'reasoning', 'validator', 
                       'answer_generator', 'web_search'). If None, returns default config.
        
        Returns:
            Dictionary with model, temperature, and API key configuration
        """
        # Default configuration
        model = cls.GEMINI_MODEL
        temperature = cls.TEMPERATURE
        
        # Override with agent-specific configuration if available
        if agent_name:
            agent_name_lower = agent_name.lower()
            
            if agent_name_lower == 'coordinator' and cls.COORDINATOR_MODEL:
                model = cls.COORDINATOR_MODEL
                temperature = cls.COORDINATOR_TEMPERATURE or cls.TEMPERATURE
            elif agent_name_lower == 'reasoning' and cls.REASONING_MODEL:
                model = cls.REASONING_MODEL
                temperature = cls.REASONING_TEMPERATURE or cls.TEMPERATURE
            elif agent_name_lower == 'validator' and cls.VALIDATOR_MODEL:
                model = cls.VALIDATOR_MODEL
                temperature = cls.VALIDATOR_TEMPERATURE or cls.TEMPERATURE
            elif agent_name_lower == 'answer_generator' and cls.ANSWER_GENERATOR_MODEL:
                model = cls.ANSWER_GENERATOR_MODEL
                temperature = cls.ANSWER_GENERATOR_TEMPERATURE or cls.TEMPERATURE
            elif agent_name_lower == 'web_search' and cls.WEB_SEARCH_MODEL:
                model = cls.WEB_SEARCH_MODEL
                temperature = cls.WEB_SEARCH_TEMPERATURE or cls.TEMPERATURE
        
        return {
            "model": model,
            "temperature": temperature,
            "google_api_key": cls.GOOGLE_API_KEY
        }
    
    @classmethod
    def get_medprompt_config(cls) -> dict:
        """
        Get Medprompt-specific configuration.
        
        Returns:
            Dictionary with all Medprompt settings
        """
        return {
            # Embedding
            "embedding_model": cls.EMBEDDING_MODEL,
            "embedding_cache_dir": cls.EMBEDDING_CACHE_DIR,
            
            # Few-shot
            "enable_few_shot": cls.ENABLE_FEW_SHOT,
            "few_shot_k": cls.FEW_SHOT_K,
            "few_shot_min_similarity": cls.FEW_SHOT_MIN_SIMILARITY,
            "knn_index_path": cls.KNN_INDEX_PATH,
            
            # CoT
            "enable_cot": cls.ENABLE_COT,
            "cot_detailed": cls.COT_DETAILED,
            
            # Ensemble
            "enable_ensemble": cls.ENABLE_ENSEMBLE,
            "ensemble_variants": cls.ENSEMBLE_VARIANTS,
            "ensemble_confidence_threshold": cls.ENSEMBLE_CONFIDENCE_THRESHOLD,
            
            # Self-consistency
            "enable_self_consistency": cls.ENABLE_SELF_CONSISTENCY,
            "self_consistency_samples": cls.SELF_CONSISTENCY_SAMPLES,
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration for debugging."""
        print("=" * 60)
        print("Current Configuration")
        print("=" * 60)
        print(f"\n[LLM Settings]")
        print(f"  Default Model: {cls.GEMINI_MODEL}")
        print(f"  Temperature: {cls.TEMPERATURE}")
        
        print(f"\n[Medprompt Settings]")
        print(f"  Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"  Few-shot Enabled: {cls.ENABLE_FEW_SHOT}")
        print(f"  Few-shot K: {cls.FEW_SHOT_K}")
        print(f"  CoT Enabled: {cls.ENABLE_COT}")
        print(f"  Ensemble Enabled: {cls.ENABLE_ENSEMBLE}")
        print(f"  Ensemble Variants: {cls.ENSEMBLE_VARIANTS}")
        print("=" * 60)
