"""
Configuration Service Module

This module provides classes and functions for loading and managing configuration settings 
from various sources (system configuration, testcase-specific configurations, etc.). 
"""

from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Literal, Optional
from src.utils.logger_mixin import LoggerMixin

ModelType = Literal["openai", "ollama", "lmstudio"]

@dataclass
class RetrievalConfig:
    top_k: int = 5

@dataclass
class VectorStoreConfig:
    chunk_size: int = 200
    chunk_overlap: int = 100

@dataclass
class LLMConfig:
    model_type: ModelType = "openai"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    response_mode: Literal["source_only", "detailed"] = "source_only"

@dataclass
class LangchainConfig:
    retrieval: RetrievalConfig
    vector_store: VectorStoreConfig
    llm: LLMConfig

class ConfigService(LoggerMixin):
    """Service for managing all configuration aspects of the application."""
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.system_config_path = Path(config_path)
        if not self.system_config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(self.system_config_path, 'r') as f:
            self.raw_config = yaml.safe_load(f)
            
    def _create_config_from_dict(self, config_dict: dict) -> LangchainConfig:
        """Create a LangchainConfig object from a dictionary."""
        return LangchainConfig(
            retrieval=RetrievalConfig(**config_dict.get('retrieval', {})),
            vector_store=VectorStoreConfig(**config_dict.get('vector_store', {})),
            llm=LLMConfig(**config_dict.get('llm', {}))
        )
    
    def load_langchain_config(self, testcase_path: Optional[Path] = None, 
                            provided_config: Optional[LangchainConfig] = None) -> LangchainConfig:
        """Load LangChain configuration in order of precedence:
        1. Provided config if exists
        2. Testcase config if exists, if not create it from system config
        3. System config if custom is specified
        4. Default config
        """
        self.logger.debug("Loading LangChain configuration...")
        
        # If config is provided directly, use it
        if provided_config:
            self.logger.info("Using provided configuration.")
            return provided_config
            
        # Try to load testcase config if path is provided
        if testcase_path:
            config_path = testcase_path / "config.yaml"
            if config_path.exists():
                self.logger.info("Loading testcase config from %s", config_path)
                with open(config_path, 'r') as f:
                    testcase_config = yaml.safe_load(f)
                    if testcase_config:
                        return self._create_config_from_dict(testcase_config)
            else:
                self.logger.info("No testcase config found, creating one...")
                # Get config data based on system config
                config = self._get_system_langchain_config()
                
                # Save to testcase directory
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, 'w') as f:
                    yaml.dump({
                        'retrieval': vars(config.retrieval),
                        'vector_store': vars(config.vector_store),
                        'llm': vars(config.llm)
                    }, f, default_flow_style=False)
                
                self.logger.info("Created new config file at %s", config_path)
                return config
        
        # If no testcase path provided, load from system config
        return self._get_system_langchain_config()
    
    def _get_system_langchain_config(self) -> LangchainConfig:
        """Get LangChain configuration from system config."""
        system_config = self.raw_config.get('config', {})
        langchain_config_name = system_config.get('langchain_config')
        
        # If custom config is specified and exists, use it
        if langchain_config_name and langchain_config_name != 'default':
            self.logger.info("Loading custom config: %s", langchain_config_name)
            custom_config = self.raw_config.get(langchain_config_name)
            if custom_config:
                return self._create_config_from_dict(custom_config)
        
        # Fall back to default config
        self.logger.info("Using default configuration.")
        return self._create_config_from_dict({
            'retrieval': {},
            'vector_store': {},
            'llm': {}
        })

def get_config_service(config_path: str = "config/system_config.yaml") -> ConfigService:
    """Factory function for ConfigService."""
    return ConfigService(config_path) 