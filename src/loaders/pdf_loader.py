"""
PDF loading and text extraction.
For detailed documentation, see: [pdf_loader.md](../documentation/pdf_loader.md)

Handles PDF document loading and text extraction.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
from langchain_community.document_loaders import PyPDFLoader

class BasePDFLoader(ABC):
    """Abstract base class for PDF loaders."""
    
    @abstractmethod
    def load(self, pdf_path: Path) -> Optional[str]:
        """Load and combine PDF content."""
        pass

class PyPDFLoaderWrapper(BasePDFLoader):
    """Wrapper for LangChain's PyPDFLoader."""
    
    def load(self, pdf_path: Path) -> Optional[str]:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            return "\n".join(page.page_content for page in pages)
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {str(e)}")
            return None

# Add more implementations as needed:
# class PDFPlumberLoader(BasePDFLoader):
#     def load(self, pdf_path: Path) -> Optional[str]:
#         ...

# class PyMuPDFLoader(BasePDFLoader):
#     def load(self, pdf_path: Path) -> Optional[str]:
#         ...

class PDFLoader:
    """Handles PDF loading and validation operations."""
    
    def __init__(self, loader: Optional[BasePDFLoader] = None):
        """
        Initialize PDFLoader with specific loader implementation.
        
        Args:
            loader: PDF loader implementation to use. Defaults to PyPDFLoaderWrapper.
        """
        self._loader = loader or PyPDFLoaderWrapper()
    
    @staticmethod
    def validate_pdf(pdf_path: Optional[Path], base_dir: str) -> bool:
        """
        Validate PDF file existence.
        
        Args:
            pdf_path: Path to the PDF file
            base_dir: Base directory for error messaging
            
        Returns:
            bool: True if PDF exists, False otherwise
        """
        if not pdf_path:
            print(f"Error: No PDF file found in {base_dir}")
            return False
        if not pdf_path.exists():
            print(f"Error: PDF file not found at {pdf_path}")
            return False
        return True

    def load_pdf_content(self, pdf_path: Path, base_dir: str) -> Optional[str]:
        """
        Validate and load PDF content.
        
        Args:
            pdf_path: Path to the PDF file
            base_dir: Base directory for error messaging
            
        Returns:
            Optional[str]: Combined text content of the PDF, or None if validation/loading fails
        """
        if not self.validate_pdf(pdf_path, base_dir):
            return None
            
        return self._loader.load(pdf_path)

# Factory function to create PDFLoader with specific implementation
def get_pdf_loader(loader_type: str = "pypdf") -> PDFLoader:
    """
    Create PDFLoader with specified loader implementation.
    
    Args:
        loader_type: Type of loader to use ("pypdf", "plumber", "mupdf", etc.)
        
    Returns:
        PDFLoader: Configured PDFLoader instance
    """
    loaders = {
        "pypdf": PyPDFLoaderWrapper,
        # Add more implementations as they become available:
        # "plumber": PDFPlumberLoader,
        # "mupdf": PyMuPDFLoader,
    }
    
    loader_class = loaders.get(loader_type, PyPDFLoaderWrapper)
    return PDFLoader(loader_class())
