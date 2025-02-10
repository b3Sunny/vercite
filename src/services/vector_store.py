"""
Vector store service for managing embeddings.

Handles vector database operations and embeddings management.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Iterator
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
import time
from src.utils.logger_mixin import LoggerMixin
from src.utils.path_utils import TestcasePaths
from config.config_loader import VectorStoreConfig

class BaseVectorStore(ABC, LoggerMixin):
    """Abstract base class for vector store implementations."""
    
    @abstractmethod
    def initialize_store(self, collection_name: str) -> None:
        """Initialize or load existing vector store."""
        pass
    
    @abstractmethod
    def get_retriever(self, **kwargs):
        """Get retriever interface for the vector store."""
        pass

class ChromaVectorStore(BaseVectorStore):
    """ChromaDB implementation of vector store."""
    
    def __init__(self, paths: TestcasePaths, config: Optional[VectorStoreConfig] = None):
        self.paths = paths
        self.config = config or VectorStoreConfig()
        self.client = chromadb.PersistentClient(
            path=str(self.paths.db_path),
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
        self.vectordb: Optional[Chroma] = None
        self.rate_limit_tokens = 1_000_000  # 1M tokens per minute
        self.batch_size = 350  # Number of documents to process in each batch
    
    def _batch_documents(self, documents: List[Document]) -> Iterator[List[Document]]:
        """Yield batches of documents."""
        for i in range(0, len(documents), self.batch_size):
            yield documents[i:i + self.batch_size]

    def initialize_store(self, collection_name: str) -> None:
        """Initialize ChromaDB store with rate limiting."""
        if any(collection.name == collection_name for collection in self.client.list_collections()):
            self.logger.info(f"Collection {collection_name} already exists")
            self.vectordb = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
            )
            return

        self.logger.info(f"Creating new collection {collection_name}")
        print("\nCreating embeddings...")
        documents = []
        # sum of all pdf files in the referenced_papers_dir
        total_size = sum(f.stat().st_size for f in self.paths.referenced_papers_dir.glob("*.pdf"))
        processed_size = 0

        # Load documents with progress tracking
        for file in self.paths.referenced_papers_dir.glob("*.pdf"):
            try:
                self.logger.info(f"Loading PDF: {file.name} ({file.stat().st_size / 1024 / 1024:.2f} MB)")
                loader = PyPDFLoader(str(file))
                documents.extend(loader.load())
                processed_size += file.stat().st_size
                self.logger.info(f"Progress: {(processed_size / total_size) * 100:.2f}%")
            except Exception as e:
                self.logger.warning(f"Warning: Failed to load PDF {file}: {str(e)}")
                continue

        text_splitter = CharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        chunked_documents = text_splitter.split_documents(documents)
        self.logger.info(f"Split documents into {len(chunked_documents)} chunks")

        # Process documents in batches with rate limiting
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
        processed_chunks = []
        
        for batch in self._batch_documents(chunked_documents):
            try:
                self.logger.info(f"Processing batch of {len(batch)} chunks")
                print(f"\tProcessing batch of {len(batch)} chunks")
                # Create vectordb for first batch or add to existing
                if not processed_chunks:
                    self.vectordb = Chroma.from_documents(
                        documents=batch,
                        embedding=embedding_function,
                        collection_name=collection_name,
                        persist_directory=str(self.paths.db_path),
                        client=self.client
                    )
                else:
                    self.vectordb.add_documents(documents=batch)
                
                processed_chunks.extend(batch)
                self.logger.info(f"Processed {len(processed_chunks)}/{len(chunked_documents)} chunks")
                print(f"\tProcessed {len(processed_chunks)}/{len(chunked_documents)} chunks")
                
                # Implement rate limiting
                time.sleep(60)  # Wait for 1 minute after each batch to respect rate limit
                
            except Exception as e:
                self.logger.error(f"Error processing batch: {str(e)}")
                raise

        self.logger.info("Finished processing all documents")
        print("\nEmbeddings created successfully")
    
    def get_retriever(self, **kwargs):
        """Get ChromaDB retriever."""
        if not self.vectordb:
            raise ValueError("Vector store not initialized")
        return self.vectordb.as_retriever(**kwargs)

class VectorStoreService:
    """Service for managing vector store operations."""
    
    def __init__(self, store: BaseVectorStore):
        self.store = store
    
    def run(self, collection_name: str = "langchain") -> None:
        """Initialize vector store with default collection name."""
        self.store.initialize_store(collection_name)
    
    def get_retriever(self, **kwargs):
        """Get configured retriever."""
        return self.store.get_retriever(**kwargs)

def get_vector_store_service(store_type: str, paths: TestcasePaths, config: Optional[VectorStoreConfig] = None) -> VectorStoreService:
    """Factory function for vector store service."""
    stores = {
        "chroma": lambda: ChromaVectorStore(paths, config),
        # Add more implementations as needed
    }
    
    store_class = stores.get(store_type)
    if not store_class:
        raise ValueError(f"Unsupported vector store type: {store_type}")
        
    return VectorStoreService(store_class()) 