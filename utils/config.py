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

