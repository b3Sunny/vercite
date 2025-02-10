"""
Path management utilities for test case directory structure.

Manages directory structure and ensures consistent path handling across the application.
"""

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional
from functools import cached_property

@dataclass
class TestcasePaths:
    """Holds all paths for a testcase using Path objects for better path handling"""
    # Class constants for file names
    _CLAIMS_FILE = "extracted_claims.json"
    _REFERENCES_FILE = "extracted_references.json"
    _NEGATIVE_CLAIMS_FILE = "extracted_claims.json"
    _PREPROCESSED_CLAIMS_FILE = "preprocessed_claims.json"
    _RESULTS_FILE = "results.json"
    _EVALUATION_METRICS_FILE = "evaluation_metrics.json"
    _CONFIG_FILE = "system_config.yaml"
    base_dir: Path
    db_path: Path
    logs_dir: Path
    results_dir: Path
    referenced_papers_dir: Path
    extracted_data_dir: Path
    current_results_dir: Optional[Path] = None
    timestamp_dir: Optional[Path] = None
    current_results_path: Optional[Path] = None

    @classmethod
    def create(cls, testcase_name: str) -> 'TestcasePaths':
        """Builder method to create TestcasePaths with all necessary directories"""
        base = Path("data") / testcase_name
        
        paths = cls(
            base_dir=base,
            db_path=base / "db",
            logs_dir=base / "logs",
            results_dir=base / "results",
            referenced_papers_dir=base / "referenced_papers",
            extracted_data_dir=base / "extracted_data"
        )
        
        paths._create_directories()
        return paths

    def create_timestamped_results_dir(self) -> Path:
        """Create and set the current results directory with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_results_dir = self.results_dir / timestamp
        self.timestamp_dir = timestamp
        self.current_results_dir.mkdir(parents=True, exist_ok=True)
        self.set_current_results_path(timestamp)
        return self.current_results_dir

    def _create_directories(self) -> None:
        """Create all necessary directories"""
        for path in self._get_required_directories():
            path.mkdir(parents=True, exist_ok=True)

    def _get_required_directories(self) -> list[Path]:
        """Get list of directories that should exist"""
        return [
            self.db_path,
            self.logs_dir, 
            self.results_dir,
            self.referenced_papers_dir,
            self.extracted_data_dir
        ]

    @cached_property
    def pdf_path(self) -> Optional[Path]:
        """Get the path to the PDF file if it exists"""
        pdf_files = list(self.base_dir.glob("*.pdf"))
        return pdf_files[0] if pdf_files else None

    @cached_property
    def claims_json_path(self) -> Path:
        """Path to the extracted claims JSON file"""
        return self.extracted_data_dir / self._CLAIMS_FILE
    
    @cached_property
    def preprocessed_claims_path(self) -> Path:
        """Path to the preprocessed claims JSON file"""
        return self.extracted_data_dir / self._PREPROCESSED_CLAIMS_FILE

    @cached_property
    def references_json_path(self) -> Path:
        """Path to the extracted references JSON file"""
        return self.extracted_data_dir / self._REFERENCES_FILE
    
    def set_current_results_path(self, current_results_dir: Path) -> None:
        """Path to the results JSON file"""
        self.current_results_path = self.results_dir / current_results_dir / self._RESULTS_FILE
    
    def evaluation_metrics_path(self, current_results_dir: Path) -> Path:
        """Path to the evaluation metrics JSON file"""
        return self.results_dir / current_results_dir / self._EVALUATION_METRICS_FILE

    def negative_claims_path(self, field_of_study: str) -> Path:
        """Get the path to the extracted negative claims JSON file"""
        return Path("data") / "negative_claims" / field_of_study / self._NEGATIVE_CLAIMS_FILE

    def get_result_file(self, filename: str) -> Path:
        """Get path for a result file in the current results directory"""
        if self.current_results_dir is None:
            raise ValueError("Current results directory not set. Call create_results_dir() first.")
        return self.current_results_dir / filename

    def exists(self, path_property: str) -> bool:
        """Check if a specific path exists"""
        path = getattr(self, path_property)
        return path.exists() if path else False

    @classmethod
    def exists(cls, testcase_name: str) -> bool:
        """Check if a testcase with the given name already exists"""
        base = Path("data") / testcase_name
        return base.exists()

    def get_claims_path(self, field_of_study: str) -> Path:
        """Get the path to the extracted claims JSON file for the specified field of study."""
        return Path("data") / "negative_claims" / field_of_study / "extracted_claims.json"

    def get_markdown_log_file(self, timestamp: str) -> Path:
        """Get path for markdown log file with timestamp"""
        return self.logs_dir / f"log_{timestamp}.md"

    @cached_property
    def config_path(self) -> Path:
        """Path to the testcase's system configuration file"""
        return self.base_dir / self._CONFIG_FILE

def get_testcase_paths(testcase_name: str) -> TestcasePaths:
    """Factory function to create TestcasePaths instance"""
    return TestcasePaths.create(testcase_name)
