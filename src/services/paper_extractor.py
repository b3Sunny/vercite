"""
PDF data extraction for claims and references.

Extracts claims and arXiv references from academic papers in PDF format.
"""

from typing import List, Tuple, Optional
import re
from src.models.models import Reference, DataEntry
from src.utils.path_utils import TestcasePaths
from src.utils.json_manager import JSONManager
from src.loaders.pdf_loader import PDFLoader, get_pdf_loader
from src.utils.arxiv_utils import ArxivUtils

class PaperDataExtractor:
    """Extracts and processes data from academic papers."""
    
    def __init__(self, paths: TestcasePaths, pdf_loader: Optional[PDFLoader] = None):
        self.paths = paths
        self.pdf_loader = pdf_loader or get_pdf_loader() 

    def _process_paper(self) -> Tuple[bool, List[DataEntry], List[Reference]]:
        """Main method to extract claims and references from the paper."""
        if not (text := self.pdf_loader.load_pdf_content(self.paths.pdf_path, self.paths.base_dir)):
            return False, [], []
        
        print(f"\nProcessing paper: {self.paths.pdf_path}")
        claims, references = self._extract_paper_data(text)
        
        print(f"Total claims with citations: {len(claims)}")
        print(f"Total references: {len(references)}")
        
        self._save_results(claims, references)
        return True, claims, references

    def _extract_paper_data(self, text: str) -> Tuple[List[DataEntry], List[Reference]]:
        """Extract both claims and references from the paper."""
        main_content, references_section = self._split_content(text)
        
        references = self._extract_references(references_section)
        arxiv_ref_numbers = {ref.number for ref in references}
        
        claims = []
        if arxiv_ref_numbers:
            claims = self._extract_claims(main_content, arxiv_ref_numbers)
            
        return claims, references

    def _split_content(self, text: str) -> Tuple[str, str]:
        """Split text into main content and references section."""
        split_pattern = r'(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\s*\n'
        parts = re.split(split_pattern, text, maxsplit=1)
        return (parts[0], parts[1]) if len(parts) > 1 else (text, "")

    def _extract_references(self, text: str) -> List[Reference]:
        """Extract reference entries with arXiv IDs."""
        references = []
        # Match reference entries like "[1] Author..."
        reference_pattern = r'\[(\d+)\](.*?)(?=\[\d+\]|\Z)'
        
        # Extract references from text
        for match in re.finditer(reference_pattern, text, re.DOTALL):
            ref_num = match.group(1)
            ref_text = ' '.join(match.group(2).strip().split())
            
            # create reference if arxiv id is found
            if ref_text and (arxiv_id := self._extract_arxiv_id(ref_text)):
                reference = Reference(
                    number=ref_num,
                    text=ref_text,
                    arxiv_id=arxiv_id,
                    pdf_link=f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                )
                references.append(reference)
        
        return references

    def _extract_arxiv_id(self, text: str) -> Optional[str]:
        """Extract arXiv ID from reference text."""
        return ArxivUtils.extract_arxiv_id(text)

    def _extract_claims(self, text: str, valid_citations: set) -> List[DataEntry]:
        """Extract sentences containing citations to arXiv papers."""
        text = self._preprocess_text(text)
        sentences = self._split_into_sentences(text)
        return self._process_claims(sentences, valid_citations)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better sentence extraction."""
        # Handle hyphenated line breaks
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        # Remove line breaks and extra spaces
        return ' '.join(text.split())

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling bullet points."""
        sentences = []
        bullet_parts = re.split(r'(?:•|\*|\-|\d+\.)\s+', text)
        
        for part in bullet_parts:
            if not part.strip():
                continue
                
            # Split each part by sentence boundaries
            parts = re.split(r'([.!?])\s+(?=[A-Z])', part)
            
            for i in range(0, len(parts), 2):
                if i < len(parts):
                    sentence = parts[i]
                    if i + 1 < len(parts):
                        sentence += parts[i + 1]
                    
                    sentence = sentence.strip()
                    if sentence:
                        if not sentence.endswith(('.', '!', '?')):
                            sentence += '.'
                        sentences.append(sentence)

        return sentences

    def _find_citations_in_text(self, text: str) -> set:
        """Extract all citation numbers from text."""
        citation_pattern = r'\[(\d+)\]'
        citations = set()
        for match in re.finditer(citation_pattern, text):
            citations.add(match.group(1))
        return citations

    def _process_claims(self, sentences: List[str], valid_citations: set) -> List[DataEntry]:
        """
        Process sentences to extract claims with citations.
        Args:
            sentences: List of sentences to process
            valid_citations: Set of valid citation numbers. If empty, accepts all citations.
        """
        claims = []
        
        for i, sentence in enumerate(sentences):
            # Skip sentences with grouped citations
            if re.search(r'\[\d+,\s*\d+.*?\]', sentence):
                continue
                
            # Find citations in this sentence
            citations_in_sentence = self._find_citations_in_text(sentence)
            
            # If valid_citations is empty or citation is valid, include it
            valid_citations_in_sentence = [num for num in citations_in_sentence 
                                         if not valid_citations or num in valid_citations]
            
            if valid_citations_in_sentence:
                claim = DataEntry(
                    sentence,
                    self._get_context(sentences, i - 1),
                    self._get_context(sentences, i + 1),
                    valid_citations_in_sentence,
                    [],
                    True,
                    None
                )
                claims.append(claim)
        
        return claims

    @staticmethod
    def remove_citations(text: str) -> str:
        """Remove citation brackets and normalize whitespace.
        
        Args:
            text: Input text containing citations like [1] or [1, 2, 3]
            
        Returns:
            Text with citations removed and normalized whitespace
        """
        # Remove citations in brackets and normalize whitespace
        cleaned = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        return ' '.join(cleaned.split())

    def _get_context(self, sentences: List[str], index: int) -> List[str]:
        """Get context sentence at index, removing citations."""
        if 0 <= index < len(sentences):
            return [self.remove_citations(sentences[index])]
        return []

    def _save_results(self, claims: List[DataEntry], references: List[Reference]) -> None:
        """Save extracted data to JSON files."""
        JSONManager.dump_to_json(self.paths.claims_json_path, claims)
        JSONManager.dump_to_json(self.paths.references_json_path, references)
    
    def extract_claims_directly(self, text: str, valid_citations: Optional[set] = None) -> List[DataEntry]:
        """
        Public method to extract claims directly from text (For Script (extract_claims.py))
        Args:
            text: The text to extract claims from
            valid_citations: Optional set of valid citation numbers. 
                           If None, accepts all citations found in text.
        """
        main_content, _ = self._split_content(text)
        
        # If no valid_citations provided, use all citations from text
        if valid_citations is None:
            valid_citations = self._find_citations_in_text(main_content)
            
        return self._extract_claims(main_content, valid_citations)

class PaperExtractionService:
    """Service layer for paper data extraction operations"""
    
    def __init__(self, paths: TestcasePaths, pdf_loader: PDFLoader):
        self.paths = paths
        self._extractor = PaperDataExtractor(paths, pdf_loader)
    
    def extract(self) -> Tuple[bool, List[DataEntry], List[Reference]]:
        """Execute the paper data extraction"""
        try:
            return self._extractor._process_paper()
        except Exception as e:
            print(f"Error during extraction: {e}")
            return False, [], []

def get_extraction_service(paths: TestcasePaths, pdf_loader: PDFLoader) -> PaperExtractionService:
    """Factory function for PaperExtractionService"""
    return PaperExtractionService(paths, pdf_loader)