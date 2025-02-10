"""
Utility functions for arXiv-related operations.
Handles arXiv ID validation and pattern matching.
"""

import re
from typing import Optional
import arxiv
import os
from pathlib import Path
import time

class ArxivUtils:
    """Utility class for arXiv-related operations."""
    
    # Patterns for finding arXiv IDs in text
    CITATION_PATTERNS = [
        r'ArXiv,?\s*vol\.\s*abs/(\d+\.\d+)',  # Matches "ArXiv, vol. abs/2212.05901"
        r'arXiv:(\d+\.\d+)',                   # Matches "arXiv:2212.05901"
        r'arXiv\s+e-prints,?\s*p\.\s*arXiv:(\d+\.\d+)',  # Matches "arXiv e-prints, p. arXiv:2110.04366"
        r'arxiv\.org/[a-z]+/(\d+\.\d+)'       # Matches URLs like arxiv.org/abs/2110.04366
    ]
    
    # Patterns for validating arXiv IDs
    VALID_PATTERNS = [
        # New ID format (2007-present): YYMM.NNNNN
        r'^\d{4}\.\d{4,5}(v\d+)?$',
        # Old ID format: subject/YYMMNNN
        r'^[a-z-]+/\d{7}(v\d+)?$'
    ]
    
    @classmethod
    def extract_arxiv_id(cls, text: str) -> Optional[str]:
        """Extract arXiv ID from text using citation patterns."""
        for pattern in cls.CITATION_PATTERNS:
            if match := re.search(pattern, text, re.IGNORECASE):
                return match.group(1)
        return None
    
    @classmethod
    def is_valid_arxiv_id(cls, arxiv_id: str) -> bool:
        """
        Validate if the given string is a valid arXiv ID.
        Supports both new (YYMM.NNNNN) and legacy (subject/YYMMNNN) formats.
        """
        return any(re.match(pattern, arxiv_id) for pattern in cls.VALID_PATTERNS)

    @staticmethod
    def download_paper(arxiv_id: str, output_dir: str, ref_num: str = "main") -> bool:
        """Download paper from arXiv using its ID."""
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            filename = f"[{ref_num}]_{arxiv_id}.pdf"
            output_path = os.path.join(output_dir, filename)
            
            # Check if file already exists
            if os.path.exists(output_path):
                print(f"Already exists: {filename}")
                return True
                
            # Search for the paper
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(client.results(search))
            
            # Download the paper
            paper.download_pdf(filename=output_path)
            print(f"Downloaded: {filename}")
            
            # Sleep to respect arXiv's API rate limits
            time.sleep(3)
            return True
        except Exception as e:
            print(f"Error downloading {arxiv_id}: {e}")
            return False