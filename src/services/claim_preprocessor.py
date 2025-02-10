"""
Claim preprocessing and query generation.

Handles preprocessing of claims and generation of negative claims.
"""

from typing import List, Tuple
import json
import os
from src.utils.logger_mixin import LoggerMixin
from src.utils.path_utils import TestcasePaths
from src.models.models import DataEntry, DocRetrievalQuery
from src.services.langchain import LangChainService
from src.utils.json_manager import JSONManager
import random

# TODO: Filter out claims that have more than 10 citations citations (Tables could produce a 
# lot of citations)
class ClaimPreprocessor(LoggerMixin):
    def __init__(self, paths: TestcasePaths, claims_data: List[DataEntry], langchain_service: LangChainService):
        self.paths = paths
        self._claims_data = claims_data
        self.chain = langchain_service.get_preprocessing_chain()
    
    def select_field_of_study(self, excluded_field: str = None) -> str:
        """Prompt the user to select a field of study for negative claims."""
        # Get list of available fields of study
        fields = [
            d for d in os.listdir("data/negative_claims") 
            if os.path.isdir(os.path.join("data/negative_claims", d))
        ]

        # Check if any fields are available
        if not fields:
            raise ValueError("No fields available for selection.")

        # Display options to the user
        print("\nField of Study Selection for Negative Claims")
        print("-------------------------------------------------")
        if excluded_field:
            print(f"The positive claims are based on the field of study of the processed main paper: '{excluded_field.replace('_', ' ').title()}'.")
            print("For best results, negative claims should come from a different field of study.")
            print("These claims should not be verified by the system.")
        else:
            print("You will select a field of study to be used for negative claims.")
        print("\nAvailable fields of study:")
        print("-------------------------------------------------")

        # Display fields
        for i, field in enumerate(fields, 1):
            print(f"{i}. {field.replace('_', ' ').title()}")

        # Prompt for input with basic validation
        while True:
            try:
                choice = int(input("\nEnter the number of the field: ")) - 1
                if 0 <= choice < len(fields):
                    selected_field = fields[choice]
                    print(f"You selected: {selected_field.replace('_', ' ').title()}")
                    return selected_field
                else:
                    print("Invalid selection. Please enter a valid number from the list.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    async def generate_positive_queries(self) -> List[DataEntry]:
        """Generate queries for positive claims."""
        processed_claims = []
        self.logger.info(f"Processing {len(self._claims_data)} positive claims...")
        
        for i, claim in enumerate(self._claims_data, 1):
            processed_claim = DataEntry(
                claim.original_claim,
                claim.context_before,
                claim.context_after,
                claim.references,
                [],
                True,
                None
            )
            
            print(f"Processing claim {i}/{len(self._claims_data)}")
            self.logger.info(f"Processing claim {i}/{len(self._claims_data)}")
            
            # Generate queries for each reference
            for reference in claim.references:
                try:
                    llm_generated_queries = await self.chain.ainvoke({
                        "claim": claim.original_claim,
                        "context_before": "\n".join(claim.context_before),
                        "context_after": "\n".join(claim.context_after),
                        "reference": reference
                    })
                    
                    doc_retrieval_query = DocRetrievalQuery(
                        reference,
                        llm_generated_queries["main_query"],
                        llm_generated_queries["rewritten_queries"]
                    )
                    processed_claim.doc_retrieval_queries.append(doc_retrieval_query)
                    
                    self.logger.info(f"Generated queries for reference [{reference}]")
                    print(f"\tGenerated queries for reference [{reference}]")
                except Exception as e:
                    self.logger.error(f"Error generating queries for reference [{reference}]: {e}")

                    continue

            processed_claims.append(processed_claim)
            
        return processed_claims

    async def generate_negative_queries(self, field_of_study: str) -> List[DataEntry]:
        """Generate negative claims based on selected field of study."""
        self.logger.info(f"Loading negative claims from field: {field_of_study}")
        claims_path = self.paths.negative_claims_path(field_of_study)
        
        try:
            # Load raw JSON data first
            negative_pool = JSONManager.load_from_json(claims_path)
            
            # Shuffle the negative pool to ensure randomness
            random.shuffle(negative_pool)
            
            # Limit the number of negative claims to the number of positive claims
            limit = len(self._claims_data)
            negative_claims = []
            
            for claim_data in negative_pool[:limit]:  # Select only as many as positive claims
                # Create DataEntry with minimal required fields
                negative_claim = DataEntry(
                    claim_data.get("original_claim", ""),
                    [],
                    [],
                    [],
                    [],
                    False,
                    None
                )
                negative_claims.append(negative_claim)
            
            return negative_claims
        except Exception as e:
            self.logger.error(f"Error loading negative claims from {claims_path}: {e}")
            raise

    async def process_all_claims(self) -> Tuple[bool, List[DataEntry]]:
        """Process both positive and negative claims."""
        try:
            # Process positive claims
            processed_claims = await self.generate_positive_queries()

            # Process negative claims
            selected_field = self.select_field_of_study("computer_science")
            synthetic_claims = await self.generate_negative_queries(selected_field)

            # Combine and save results
            all_claims = processed_claims + synthetic_claims
            print(f"Processed claims: positive: {len(processed_claims)}, negative: {len(synthetic_claims)}")
            claims_data = [claim.to_dict() for claim in all_claims]
            
            output_path = os.path.join(self.paths.extracted_data_dir, "preprocessed_claims.json") #TODO: use JSONManager
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(claims_data, f, indent=2, ensure_ascii=False)
                
            print(f"Processed claims saved to: {output_path}")
            self.logger.info(f"Processed claims saved to: {output_path}")
            self.logger.info(f"Total claims: {len(all_claims)} ({len(processed_claims)} positive, {len(synthetic_claims)} negative)")
            return True, all_claims

        except Exception as e:
            self.logger.error(f"Error preprocessing claims: {e}")
            raise

class ClaimPreprocessService(LoggerMixin):
    def __init__(self, paths: TestcasePaths, claims_data: List[DataEntry], langchain_service: LangChainService):
        self.paths = paths
        self._preprocessor = ClaimPreprocessor(paths, claims_data, langchain_service)
    
    async def run(self) -> Tuple[bool, List[DataEntry]]:
        """Execute the claim preprocessing"""
        try:
            return await self._preprocessor.process_all_claims()
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            return False, []

def get_preprocess_service(paths: TestcasePaths, claims_data: List[DataEntry], 
                          langchain_service: LangChainService) -> ClaimPreprocessService:
    """Factory function for ClaimPreprocessService"""
    return ClaimPreprocessService(paths, claims_data, langchain_service)