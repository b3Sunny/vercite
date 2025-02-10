"""
Core claim processing implementation using RAG (Verification)
"""

from typing import List
from src.models.models import DataEntry, Result
from src.utils.path_utils import TestcasePaths
from src.services.langchain import LangChainService
from src.utils.doc_utils import format_docs, extract_source_ids
from src.utils.logger_mixin import LoggerMixin
from src.utils.json_manager import JSONManager
from dataclasses import replace
from src.utils.logger_md import MarkdownLogger
from src.services.evaluator import RankingMetricsCalculator

class ClaimProcessor(LoggerMixin):
    """Processes claims using RAG-based verification."""
    
    def __init__(self, paths: TestcasePaths, claims_data: List[DataEntry], 
                 chain, retriever, langchain_service: LangChainService):
        self.paths = paths
        self._claims_data = claims_data
        self.chain = chain
        self.retriever = retriever
        self.mdlogger = MarkdownLogger(self.paths)
        self.langchain_service = langchain_service

    async def process_all_claims(self) -> bool:
        """Process all claims using RAG."""
        try:
            processed_claims = []
            total_claims = len(self._claims_data)
            
            self.logger.info(f"Processing {total_claims} claims")
            self.mdlogger.start_logging()
            
            # Process each claim
            for idx, claim in enumerate(self._claims_data):
                processed_claim = await self._process_single_claim(claim, total_claims, idx)
                processed_claims.append(processed_claim)
                
                # Save intermediate results
                self._save_results(processed_claims)
                    
            self.logger.info(f"Processing completed. Results saved to: {self.paths.current_results_dir}")
            self.mdlogger.end_logging()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing claims: {e}")
            return False

    async def _process_single_claim(self, claim: DataEntry, total_claims: int, idx: int) -> DataEntry:
        """Process a single claim using RAG."""
        claim_number = f"{idx + 1}/{total_claims}"
        print(f"\nProcessing claim {claim_number}")
        self.logger.info(f"Processing claim {claim_number}")
        self.mdlogger.log_claim_start(claim_number, claim.original_claim)
        self.mdlogger.add_toc_entry(claim_number, claim.original_claim)


        results = []
        
        if not claim.is_positive:
            # Process negative claim using original_claim directly
            query_used_for_retrieval = claim.original_claim
            retrieved_docs = await self.retriever.ainvoke(query_used_for_retrieval)
            self.mdlogger.log_retrieved_docs(retrieved_docs)
            
            # len of page content
            self.logger.info(f"Page content length: {len(retrieved_docs[0].page_content)}")
            # Check if any retrieved document exceeds chunk_size
            # TODO: 30000 hard coded, means character count pro doc
            if all(len(doc.page_content) < 30000 for doc in retrieved_docs):
                result = await self.chain.ainvoke({
                    "claim": query_used_for_retrieval,
                    "docs": format_docs(retrieved_docs),
                })
                
                # Handle different response modes
                if self.langchain_service.config.llm.response_mode == "detailed":
                    self.mdlogger.log_detailed_analysis(query_used_for_retrieval, result)
                    result_entry = Result(
                        "",
                        query_used_for_retrieval,
                        extract_source_ids(retrieved_docs),
                        result["source"],
                        0
                    )
                else:
                    result_entry = Result(
                        "",
                        query_used_for_retrieval,
                        extract_source_ids(retrieved_docs),
                        result.strip(),
                        0
                    )
                results.append(result_entry)
            else:
                self.logger.warning("Retrieved documents exceed the chunk size limit. Claim not processed.")
            
        else:
            # Process positive claim using queries
            for query in claim.doc_retrieval_queries:
                retrieved_docs = await self.retriever.ainvoke(query.main_query)
                self.mdlogger.log_retrieved_docs(retrieved_docs)

                # Check if any retrieved document exceeds chunk_size
                # TODO: 3000 hard coded, means character count pro doc
                if all(len(doc.page_content) < 30000 for doc in retrieved_docs):
                    result = await self.chain.ainvoke({
                        "claim": query.main_query,
                        "docs": format_docs(retrieved_docs),
                    })

                    source_ids = extract_source_ids(retrieved_docs)
                    # Calculate ranking metrics for positive claims
                    
                    target_doc = int(query.related_to_reference)
                    ranking_metrics = RankingMetricsCalculator.calculate_all_metrics(
                        source_ids, 
                        target_doc,
                        self.langchain_service.config.retrieval.top_k
                    )

                    # Handle different response modes
                    if self.langchain_service.config.llm.response_mode == "detailed":
                        self.mdlogger.log_detailed_analysis(query.main_query, result)
                        result_entry = Result(
                            query.related_to_reference,
                            query.main_query,
                            source_ids,
                            result["source"],  # Use source from JSON response
                            0,
                            ranking_metrics=ranking_metrics,
                        )
                    else:
                        result_entry = Result(
                            query.related_to_reference,
                            query.main_query,
                            source_ids,
                            result.strip(),
                            0,
                            ranking_metrics=ranking_metrics
                        )
                    results.append(result_entry)

        processed_claim = replace(claim, results=results)
        return processed_claim

    def _save_results(self, processed_claims: List[DataEntry]) -> None:
        """Save processed claims to results file using JSONManager."""
        try:
            if self.paths.current_results_dir is None:
                self.paths.create_timestamped_results_dir()
            
            # Convert DataEntry objects to dictionaries
            claims_dict = [claim.to_dict() for claim in processed_claims]
            
            JSONManager.dump_to_json(self.paths.current_results_path, claims_dict)
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise

class ClaimProcessorService(LoggerMixin):
    """Service layer for claim processing operations."""
    
    def __init__(self, paths: TestcasePaths, claims_data: List[DataEntry], 
                 langchain_service: LangChainService):
        self.paths = paths
        self.langchain_service = langchain_service
        self._init_services()
        self._processor = ClaimProcessor(
            paths, 
            claims_data, 
            self.chain,
            self.retriever,
            self.langchain_service
        )
    
    def _init_services(self) -> None:
        """Initialize required services."""
        # Initialize vector store through LangChain service
        self.langchain_service.initialize_vector_store("langchain")
        
        # Get components with configured top_k from retrieval config
        self.retriever = self.langchain_service.get_retriever(
            search_kwargs={"k": self.langchain_service.config.retrieval.top_k}
        )
        self.chain = self.langchain_service.get_source_chain()
    
    async def run(self) -> bool:
        """Execute the claim processing"""
        try:
            # Process claims
            if not await self._processor.process_all_claims():
                return False

            return True
            
        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            return False

def get_processor_service(paths: TestcasePaths, claims_data: List[DataEntry], 
                         langchain_service: LangChainService) -> ClaimProcessorService:
    """Factory function for ClaimProcessorService"""
    return ClaimProcessorService(paths, claims_data, langchain_service)