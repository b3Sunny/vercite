"""
Centralized LangChain configuration service.
Manages LLM models, chains, and other LangChain components.
"""

from typing import Optional
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from src.prompts.rag_prompts import PREPROCESSING_PROMPT, PROCESSING_SOURCE_ONLY_PROMPT, PROCESSING_DETAILED_SOURCE_PROMPT
from src.services.vector_store import get_vector_store_service
from src.utils.path_utils import TestcasePaths
from config.config_loader import get_config_service, LangchainConfig
from src.utils.logger_mixin import LoggerMixin


class LangChainService(LoggerMixin):
    """Manages LangChain components and configurations"""
    
    def __init__(self, config: Optional[LangchainConfig] = None, paths: Optional[TestcasePaths] = None):
        self._paths = paths
        self._config_service = get_config_service()
        self.config = self._config_service.load_langchain_config(
            testcase_path=self._paths.base_dir if self._paths else None,
            provided_config=config
        )
        self.logger.info("LangChainService initialized with config: %s", self.config)
        self._llm = None
        self._preprocessing_chain = None
        self._source_chain = None
        self._detailed_source_chain = None
        self._vector_store = None
    
    @property
    def llm(self) -> BaseLanguageModel:
        """Get or create LLM instance based on configuration"""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm
    
    def _create_llm(self) -> BaseLanguageModel:
        """Create LLM instance based on configuration"""
        if self.config.llm.model_type == "openai":
            return ChatOpenAI(
                model=self.config.llm.model_name,
                temperature=self.config.llm.temperature
            )
        elif self.config.llm.model_type == "ollama":
            return ChatOllama(
                model=self.config.llm.model_name,
                temperature=self.config.llm.temperature,
                base_url="http://localhost:11434"
            )
        elif self.config.llm.model_type == "lmstudio":
            # same Class as OpenAI
            return ChatOpenAI( 
                model=self.config.llm.model_name,
                temperature=self.config.llm.temperature,
                base_url="http://localhost:1234/v1",
                api_key="not-needed" # LM Studio does not need an API key
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.llm.model_type}")
        
    def get_preprocessing_chain(self):
        """Get chain for claim preprocessing"""
        if self._preprocessing_chain is None:
            self._preprocessing_chain = (
                PREPROCESSING_PROMPT | 
                self.llm | 
                JsonOutputParser()
            )
        return self._preprocessing_chain
        
    def get_source_chain(self):
        """Get chain for source processing"""
        if self.config.llm.response_mode == "detailed":
            if self._detailed_source_chain is None:
                self._detailed_source_chain = (
                    PROCESSING_DETAILED_SOURCE_PROMPT | 
                    self.llm | 
                    JsonOutputParser()
                )
            return self._detailed_source_chain
        else:
            if self._source_chain is None:
                self._source_chain = (
                    PROCESSING_SOURCE_ONLY_PROMPT | 
                    self.llm | 
                    StrOutputParser()
                )
            return self._source_chain 

    def initialize_vector_store(self, collection_name: str = "langchain") -> None:
        """Initialize vector store if not already initialized."""
        self.logger.debug("Initializing vector store...")
        if not self._paths:
            self.logger.error("TestcasePaths must be set to initialize vector store")
            raise ValueError("TestcasePaths must be set to initialize vector store")
        
        if self._vector_store is None:
            self.logger.info("Creating vector store with collection name: %s", collection_name)
            self._vector_store = get_vector_store_service(
                "chroma", 
                self._paths,
                self.config.vector_store
            )
            self._vector_store.run(collection_name)
            self.logger.info("Vector store initialized successfully.")

    def get_retriever(self, **kwargs):
        """Get retriever from vector store."""
        self.logger.debug("Retrieving from vector store...")
        if self._vector_store is None:
            self.logger.error("Vector store not initialized. Call initialize_vector_store first.")
            raise ValueError("Vector store not initialized. Call initialize_vector_store first.")
        return self._vector_store.get_retriever(**kwargs)

def get_langchain_service(
    config: Optional[LangchainConfig] = None,
    paths: Optional[TestcasePaths] = None
) -> LangChainService:
    """Factory function for LangChainService."""
    return LangChainService(config, paths)