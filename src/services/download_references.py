"""
Paper download management for arXiv references.

Handles selective downloading of cited papers from arXiv with rate limiting.
"""

from typing import Set, List
from src.utils.arxiv_utils import ArxivUtils
from src.models.models import Reference
from src.utils.path_utils import TestcasePaths
from src.utils.logger_mixin import LoggerMixin

class ReferenceDownloader(LoggerMixin):
    """Manages downloading of cited arXiv papers."""
    
    def __init__(self, paths: TestcasePaths, claims_data, references_data):
        self.paths = paths
        self._claims_data = claims_data
        self._references_data = references_data
        
    def _get_required_citations(self) -> Set[str]:
        """Extract all unique citation numbers from claims."""
        try:
            citations = {ref for claim in self._claims_data for ref in claim.references}
            self.logger.info(f"Found {len(citations)} citations in claims")
            return citations

        except Exception as e:
            self.logger.error(f"Error reading claims file: {e}")
            return set()

    def _get_cited_references(self, required_citations: Set[str]) -> List[Reference]:
        """Get references that are actually cited in claims."""
        try:
            cited_refs = [ref for ref in self._references_data if ref.number in required_citations]
            self.logger.info(f"Found {len(cited_refs)} references that match citations")
            return cited_refs
            
        except Exception as e:
            self.logger.error(f"Error loading references: {e}")
            return []

    def _download_papers(self, cited_references: List[Reference]) -> None:
        """Download papers for cited references."""
        output_dir = self.paths.referenced_papers_dir
        self.logger.info(f"Starting download of {len(cited_references)} papers to: {output_dir}")
        
        for ref in cited_references:
            if ref.arxiv_id:
                ArxivUtils.download_paper(ref.arxiv_id, output_dir, ref.number)

    def process(self) -> bool:
        """Main method to handle the download process."""
        try:
            required_citations = self._get_required_citations()
            cited_references = self._get_cited_references(required_citations)
            
            self.logger.info(f"Found {len(cited_references)} cited arXiv references")
            self._download_papers(cited_references)
            print(f"\nDocuments saved to: {self.paths.referenced_papers_dir}")
            return True
            

        except Exception as e:
            self.logger.error(f"Error during download process: {e}")
            return False


class ReferenceDownloadService(LoggerMixin):
    """Service layer for reference download operations"""
    
    def __init__(self, paths: TestcasePaths, claims_data, references_data):
        self.paths = paths
        self._downloader = ReferenceDownloader(paths, claims_data, references_data)
    
    def download(self) -> bool:
        """Execute the reference download process"""
        try:
            return self._downloader.process()
        except Exception as e:
            self.logger.error(f"Error during download: {e}")
            return False


def get_download_service(paths: TestcasePaths, claims_data, references_data) -> ReferenceDownloadService:
    """Factory function for ReferenceDownloadService"""
    return ReferenceDownloadService(paths, claims_data, references_data)
    
