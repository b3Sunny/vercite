"""
Test case management and initialization service.

Handles creation of new test cases and initial paper downloads.
"""

import os
from src.utils.path_utils import TestcasePaths, get_testcase_paths
from src.services.download_references import get_download_service
from src.loaders.pdf_loader import get_pdf_loader
from src.services.paper_extractor import get_extraction_service
from pathlib import Path
from src.utils.arxiv_utils import ArxivUtils
from src.utils.logger_mixin import LoggerMixin
from src.utils.json_manager import JSONManager
from src.services.claim_preprocessor import get_preprocess_service
from src.services.langchain import get_langchain_service
from src.services.claim_processor import get_processor_service
import asyncio
from src.services.evaluator import get_evaluator_service
from src.services.langchain import LangChainService

class TestcaseCreator(LoggerMixin):
    """Handles the core logic of testcase creation."""
    
    def __init__(self, testcase_name: str):
        self.testcase_name = testcase_name
        self.paths = None  # Initialize paths as None
    
    def create(self, arxiv_id: str) -> bool:
        """Create a new testcase with the specified arXiv paper."""

        # Validate arXiv ID
        if not ArxivUtils.is_valid_arxiv_id(arxiv_id):
            self.logger.error(f"Invalid arXiv ID format: {arxiv_id}")
            return False
            
        # Check if testcase already exists
        if TestcasePaths.exists(self.testcase_name):
            self.logger.error(f"Testcase '{self.testcase_name}' already exists")
            return False
            
        # Try downloading the paper first without creating directories
        temp_dir = Path("data") / "_temp_download"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        success = ArxivUtils.download_paper(arxiv_id, temp_dir)
        
        if not success:
            self.logger.error("Failed to download paper")
            temp_dir.rmdir()  # Clean up temp directory
            return False
            
        # If download was successful, initialize paths and move the paper
        self.paths = get_testcase_paths(self.testcase_name)
        
        # Move the downloaded paper from temp to actual directory
        downloaded_pdf = next(temp_dir.glob("*.pdf"))
        downloaded_pdf.rename(self.paths.base_dir / downloaded_pdf.name)
        temp_dir.rmdir()  # Clean up temp directory
        
        return True

class TestcaseService(LoggerMixin):
    """Service layer for testcase management operations."""
    
    def __init__(self, testcase_name: str):
        self._creator = TestcaseCreator(testcase_name)
        self.paths = None
        self.pdf_loader = get_pdf_loader()
        self._claims_cache = None
        self._preprocessed_claims_cache = None
        self._references_cache = None
        self._langchain_service = None
        self.current_results_dir = None
        
        # Initialize paths if testcase already exists
        if TestcasePaths.exists(testcase_name):
            self.paths = get_testcase_paths(testcase_name)
            
            if os.path.exists(self.paths.claims_json_path):
                self._claims_cache = JSONManager.load_claims(self.paths.claims_json_path)
            if os.path.exists(self.paths.references_json_path):
                self._references_cache = JSONManager.load_references(self.paths.references_json_path)
            if os.path.exists(self.paths.preprocessed_claims_path):
                self._preprocessed_claims_cache = JSONManager.load_preprocessed_claims(self.paths.preprocessed_claims_path)

            self.logger.info(f"Using existing testcase at: {self.paths.base_dir}")

    @property
    def langchain_service(self) -> LangChainService:
        """Lazy initialization of LangChainService"""
        if self._langchain_service is None:
            self._langchain_service = get_langchain_service(paths=self.paths)
        return self._langchain_service

    async def run(self, arxiv_id: str = None) -> bool:
        """
        Run the testcase workflow. If arxiv_id is provided, create new testcase.
        If testcase exists and no arxiv_id is provided, continue with existing testcase.
        
        Args:
            arxiv_id: Optional arXiv ID for new testcase creation
        """
        try:
            if self.paths is None:
                if not arxiv_id:
                    self.logger.error("No testcase exists. Please provide an arXiv ID.")
                    return False
                    
                if not self.create_testcase(arxiv_id):
                    return False
            
            workflow_steps = {
                "extraction": self.extract,
                "reference_download": self.download_references,
                "claim_preprocessing": self.preprocess_claims,
                "claim_processing": self.process_claims,
                "evaluation": lambda: self.evaluate_results(self.paths.timestamp_dir)
            }
            
            for step_name, step_func in workflow_steps.items():
                self.logger.info(f"Starting {step_name}")
                if step_name == "claim_processing":
                    success = await step_func()
                else: 
                    # if step_func is a coroutine, run it asynchronously, otherwise run it synchronously
                    success = await step_func() if asyncio.iscoroutinefunction(step_func) else step_func() 
                
                if not success:
                    self.logger.error(f"Failed during {step_name}")
                    return False
                self.logger.info(f"Completed {step_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error in testcase processing: {str(e)}")
            return False
    
    def create_testcase(self, arxiv_id: str) -> bool:
        """Create a new testcase with the specified arXiv paper."""
        
        print(f"\nStarting testcase creation for: {arxiv_id}\n")
        try:
            success = self._creator.create(arxiv_id)
            
            if success:
                # Update paths after successful creation
                self.paths = self._creator.paths
                # Reset langchain service to use new paths
                self._langchain_service = None
                print(f"\nSuccessfully created testcase at: {self.paths.base_dir}\n")
                self.logger.info(f"Successfully created testcase at: {self.paths.base_dir}")
            else:
                self.logger.error(f"Failed to create testcase for arXiv ID: {arxiv_id}")

            
            return success
            
        except Exception as e:
            self.logger.error(f"Error creating testcase: {e}")
            return False
        
    def extract(self) -> bool:
        """Extract claims and references from the testcase."""
        print(f"\nStarting extraction for: {self.paths.base_dir}")
        if self.paths is None:
            self.logger.error("No testcase found. Please create testcase first.")
            return False
            
        extraction_service = get_extraction_service(self.paths, self.pdf_loader) 
        success, claims, references = extraction_service.extract()
        
        if success:
            # Update cache directly with extracted data
            self._claims_cache = claims
            self._references_cache = references
            print(f"\nExtraction completed for: {self.paths.base_dir}\n")

        return success

    def download_references(self) -> bool:
        """Download referenced papers for the testcase."""
        print(f"\nStarting reference download for: {self.paths.base_dir}\n")
        try:
            download_service = get_download_service(
                self.paths, 
                claims_data=self._claims_cache,
                references_data=self._references_cache
            )

            success = download_service.download()
            print(f"\nReference download completed for: {self.paths.base_dir}\n")
            return success


        except Exception as e:
            self.logger.error(f"Error downloading references: {e}")
            return False

    async def preprocess_claims(self) -> bool:
        """Preprocess claims to generate search queries."""
        print(f"\nStarting claim preprocess for: {self.paths.base_dir}\n")
        preprocess_service = get_preprocess_service(
            self.paths, 
            self._claims_cache,
            self.langchain_service  # Uses property getter
        )
        success, preprocessed_claims = await preprocess_service.run()
        if success:
            self._preprocessed_claims_cache = preprocessed_claims
            print(f"\nClaim preprocess completed for: {self.paths.base_dir}\n")
        return success

    async def process_claims(self) -> bool:
        """Process claims using RAG-based verification."""
        print(f"\nStarting claim processing for: {self.paths.base_dir}")
        try:
            processor_service = get_processor_service(
                self.paths,
                self._preprocessed_claims_cache,
                self.langchain_service,  # Uses property getter
            )
            success = await processor_service.run()
            print(f"\nClaim processing completed for: {self.paths.base_dir}\n")
            return success

        except Exception as e:
            self.logger.error(f"Error processing claims: {e}")
            return False

    def evaluate_results(self, target_results_dir: Path) -> bool:
        """Evaluate processing results from specified directory."""
        print(f"\nStarting evaluation for: {self.paths.base_dir}\n")
        self.paths.set_current_results_path(target_results_dir)
        
        if not self.paths.current_results_path.exists():
            self.logger.error(f"No results.json found in {self.paths.current_results_path}")
            return False

        self.logger.info(f"Evaluating results from: {self.paths.current_results_path}")
        
        # Load results using JSONManager
        results = JSONManager.load_results(self.paths.current_results_path)
        
        evaluator = get_evaluator_service(
            results,
            self.paths.current_results_dir,
            self.langchain_service.config
        )

        metrics = evaluator.run()
        metrics_path = self.paths.evaluation_metrics_path(target_results_dir)
        JSONManager.dump_to_json(metrics_path, metrics)
        print(f"\nEvaluation completed for: {self.paths.base_dir}\n")

        return True

def get_testcase_service(testcase_name: str) -> TestcaseService:
    """Factory function for TestcaseService."""
    return TestcaseService(testcase_name)