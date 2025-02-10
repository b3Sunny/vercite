"""
Simple script to download papers and extract claims into a directory.
Usage: python extract_claims.py <directory_name> or <arxiv_id1> [arxiv_id2 arxiv_id3 ...]

may be not up to date
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))


from src.utils.arxiv_utils import ArxivUtils
from src.services.paper_extractor import PaperDataExtractor
from src.loaders.pdf_loader import get_pdf_loader
from src.utils.json_manager import JSONManager
from src.utils.logger_mixin import LoggerMixin
from dataclasses import dataclass
from typing import List
from src.models.models import SimpleClaim


@dataclass
class SimplePaths:
    """Minimal path structure for claim extraction"""
    base_dir: Path
    pdf_dir: Path
    claims_json_path: Path

    @classmethod
    def create(cls, directory_name: str) -> 'SimplePaths':
        base = Path("data") / directory_name
        return cls(
            base_dir=base,
            pdf_dir=base / "pdfs",
            claims_json_path=base / "extracted_claims.json",
        )


class SimpleClaimExtractor(LoggerMixin):
    """Downloads papers and extracts claims with minimal setup."""
    
    def __init__(self, directory_name: str):
        self.paths = SimplePaths.create(directory_name)
        self.pdf_loader = get_pdf_loader()
        self.logger.info(f"Initialized claim extractor for directory: {directory_name}")
        
    def extract_multiple(self, arxiv_ids: List[str]) -> bool:
        """Download multiple papers and extract claims."""
        try:
            # Create directories
            self.paths.base_dir.mkdir(parents=True, exist_ok=True)
            self.paths.pdf_dir.mkdir(parents=True, exist_ok=True)
            
            all_claims = []
            
            for arxiv_id in arxiv_ids:
                self.logger.info(f"\nProcessing paper {arxiv_id}...")
                
                # Download paper
                success = ArxivUtils.download_paper(arxiv_id, self.paths.pdf_dir)
                if not success:
                    self.logger.error(f"Failed to download paper: {arxiv_id}")
                    continue
                
                # Find PDF file
                pdf_files = list(self.paths.pdf_dir.glob(f"*{arxiv_id}*.pdf"))
                if not pdf_files:
                    self.logger.error(f"No PDF file found for {arxiv_id}")
                    continue
                pdf_path = pdf_files[0]
                
                # Extract data
                self.logger.info(f"Extracting claims from {pdf_path.name}...")
                extractor = PaperDataExtractor(self.paths, self.pdf_loader)
                text = self.pdf_loader.load_pdf_content(pdf_path, self.paths.base_dir)
                claims = extractor.extract_claims_directly(text, None)
                
                # Konvertiere zu SimpleClaim
                for claim in claims:
                    simple_claim = SimpleClaim(
                        original_claim=PaperDataExtractor.remove_citations(claim.original_claim),
                    )
                    all_claims.append(simple_claim)
                
                self.logger.info(f"Extracted {len(claims)} claims from {arxiv_id}")
            
            # Save all results
            if all_claims:
                JSONManager.dump_to_json(self.paths.claims_json_path, all_claims)
                
                print(f"\nResults saved in {self.paths.base_dir}:")
                print(f"- PDFs: {self.paths.pdf_dir}")
                print(f"- Total {len(all_claims)} claims from {len(arxiv_ids)} papers: {self.paths.claims_json_path.name}")
                return True
            else:
                self.logger.error("No claims were extracted from any papers")
                return False
            
        except Exception as e:
            self.logger.error(f"Error during extraction: {str(e)}", exc_info=True)
            return False

    def process_local_pdfs(self) -> bool:
        """Process all PDF files in the pdf directory."""
        try:
            # Create directories if they don't exist
            self.paths.base_dir.mkdir(parents=True, exist_ok=True)
            self.paths.pdf_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all PDF files
            pdf_files = list(self.paths.pdf_dir.glob("*.pdf"))
            
            if not pdf_files:
                self.logger.error(f"No PDF files found in {self.paths.pdf_dir}")
                return False
                
            self.logger.info(f"Found {len(pdf_files)} PDF files to process")
            all_claims = []
            
            for pdf_path in pdf_files:
                self.logger.info(f"\nProcessing {pdf_path.name}...")
                
                # Extract data
                extractor = PaperDataExtractor(self.paths, self.pdf_loader)
                text = self.pdf_loader.load_pdf_content(pdf_path, self.paths.base_dir)
                
                if not text:
                    self.logger.error(f"Could not extract text from {pdf_path.name}")
                    continue
                    
                claims = extractor.extract_claims_directly(text, None)
                
                # Convert to SimpleClaim
                for claim in claims:
                    simple_claim = SimpleClaim(
                        original_claim=PaperDataExtractor.remove_citations(claim.original_claim),
                    )
                    all_claims.append(simple_claim)
                
                self.logger.info(f"Extracted {len(claims)} claims from {pdf_path.name}")
            
            # Save all results
            if all_claims:
                JSONManager.dump_to_json(self.paths.claims_json_path, all_claims)
                
                print(f"\nResults saved in {self.paths.base_dir}:")
                print(f"- PDFs processed: {len(pdf_files)}")
                print(f"- Total {len(all_claims)} claims extracted: {self.paths.claims_json_path.name}")
                return True
            else:
                self.logger.error("No claims were extracted from any PDFs")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during local PDF processing: {str(e)}", exc_info=True)
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Extract claims from ArXiv papers or local PDF files'
    )
    parser.add_argument('directory', help='Name of the directory to store results')
    parser.add_argument('arxiv_ids', nargs='*', help='Optional: One or more ArXiv IDs of papers to process. If not provided, processes PDFs from directory')
    
    args = parser.parse_args()
    
    extractor = SimpleClaimExtractor(args.directory)
    if args.arxiv_ids:
        if not extractor.extract_multiple(args.arxiv_ids):
            sys.exit(1)
    else:
        if not extractor.process_local_pdfs():
            sys.exit(1)


if __name__ == "__main__":
    main() 