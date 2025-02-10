"""
Centralized logging service with markdown formatting.

Provides structured logging with table of contents and timestamp management.
"""

from datetime import datetime
from src.utils.doc_utils import format_docs
from src.utils.path_utils import TestcasePaths
from src.utils.logger_mixin import LoggerMixin

class MarkdownLogger(LoggerMixin):
    def __init__(self, paths: TestcasePaths):
        self.paths = paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.paths.get_markdown_log_file(timestamp)
        self.claims_toc = []
        self.logger.info(f"Initialized MarkdownLogger with log file: {self.log_file}")
    
    def log(self, content: str, mode: str = 'a') -> None:
        """Write content to log file"""
        with open(self.log_file, mode, encoding='utf-8') as f:
            f.write(f"{content}\n")
    
    def start_logging(self) -> None:
        """Initialize log file with header"""
        self.log(f"# Claims Processing Log\n", mode='w')
        self.log(f"Processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log("## Table of Contents\n")
        self.log("\n## Processing Details\n")
    
    def add_toc_entry(self, claim_number: str, claim_text: str):
        """Add entry to table of contents"""
        self.claims_toc.append({
            'number': claim_number,
            'text': claim_text[:50] + '...' if len(claim_text) > 50 else claim_text
        })
    
    def write_toc(self):
        """Write table of contents to the log file"""
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get the log filename without extension
        log_filename = self.log_file.stem
        
        # Create TOC entries with the new format
        toc_entries = [
            f"[[{log_filename}###Claim {entry['number']}|Claim {entry['number']}]]"
            for entry in self.claims_toc
        ]
        
        toc = "\n".join(toc_entries)
        content = content.replace("## Table of Contents\n", f"## Table of Contents\n\n{toc}\n")
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def end_logging(self) -> None:
        """Add completion message and write TOC"""
        self.log("\n## Processing Completed")
        self.log(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.write_toc()
    def log_claim_start(self, claim_number: str, claim_text: str, original_source: str = None) -> None:
        """Log the start of claim processing with optional source."""
        self.log(f"\n### Claim {claim_number}")
        self.log(f"\n#### Claim Text")
        self.log(f"{claim_text}")
        if original_source:
            self.log(f"\n#### Original Source")
            self.log(f"{original_source}")

    def log_retrieved_docs(self, docs) -> None:
        """Log retrieved documents."""
        self.log("\n#### Retrieved Documents")
        self.log(format_docs(docs))
        
    def log_detailed_analysis(self, query_used_for_retrieval: str, result: dict) -> None:
        """Log detailed analysis results."""
        self.log("\n#### Detailed Analysis")
        self.log(f"\n##### Query Used for Retrieval")
        self.log(f"{query_used_for_retrieval}")
        for key in ['predicted_source', 'page', 'reason', 'relevant_text', 'confidence']:
            if key in result:
                self.log(f"\n##### {key.title()}")
                self.log(f"{result[key]}")
        
    def log_results(self, calculated_source: str, correctness: int) -> None:
        """Log processing results."""
        self.log(f"\n#### Results")
        self.log(f"\n##### Calculated Source")
        self.log(f"{calculated_source}")
        self.log(f"\n##### Correctness")
        self.log(f"{correctness}")

    def log_error(self, error_msg: str) -> None:
        """Log error message."""
        self.log(f"\n#### ERROR")
        self.log(f"{error_msg}")
        
    def log_evaluation_results(self, accuracy: float, correct_claims: int, total_claims: int) -> None:
        """Log evaluation results."""
        self.log("\n## Evaluation Results")
        self.log(f"\nAccuracy: {accuracy:.2%}")
        self.log(f"\nCorrect claims: {correct_claims}/{total_claims}")

