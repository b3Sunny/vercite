"""
Handles all JSON operations including reading, writing, and converting between JSON and model objects.
"""

import json
from pathlib import Path
from typing import List, TypeVar, Type, Union, Dict, Optional
from dataclasses import asdict

from src.models.models import DataEntry, Reference, DocRetrievalQuery, Result, RankingMetrics
from src.utils.logger_mixin import LoggerMixin

T = TypeVar('T')

class JSONManager(LoggerMixin):
    @staticmethod
    def dump_to_json(file_path: Union[str, Path], data: any, create_dir: bool = True) -> None:
        """Write data to a JSON file.
        
        Args:
            file_path: Path to the JSON file
            data: Data to write to the JSON file
            create_dir: Whether to create the directory if it doesn't exist
        """
        try:

            file_path = Path(file_path)
            if create_dir:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
            with open(file_path, 'w', encoding='utf-8') as f:
                if hasattr(data, '__iter__') and not isinstance(data, dict):
                    data = [asdict(item) if hasattr(item, '__dataclass_fields__') else item for item in data]
                elif hasattr(data, '__dataclass_fields__'):
                    data = asdict(data)
                json.dump(data, f, indent=2, ensure_ascii=False)
            JSONManager().logger.info(f"Successfully wrote data to {file_path}")
            print(f"Saved to: {file_path}")
        except Exception as e:
            JSONManager().logger.error(f"Error writing to {file_path}: {str(e)}")
            raise

    @staticmethod
    def load_from_json(file_path: Union[str, Path], model_class: Type[T] = None) -> Union[T, List[T], Dict, List]:
        """
        Load data from a JSON file and optionally convert to model objects.
        
        Args:
            file_path: Path to JSON file
            model_class: Optional dataclass type to convert the JSON data into
        """
        file_path = Path(file_path)
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            JSONManager().logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if model_class is None:
                JSONManager().logger.info(f"Successfully loaded raw JSON data from {file_path}")
                return data
                
            if isinstance(data, list):
                result = [model_class(**item) for item in data]
            else:
                result = model_class(**data)
                
            JSONManager().logger.info(f"Successfully loaded and converted data from {file_path} to {model_class.__name__}")
            return result
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format in {file_path}: {str(e)}"
            JSONManager().logger.error(error_msg)
            raise json.JSONDecodeError(error_msg, e.doc, e.pos)
        except Exception as e:
            error_msg = f"Error loading {file_path}: {str(e)}"
            JSONManager().logger.error(error_msg)
            raise

    @staticmethod
    def load_claims(file_path: Union[str, Path]) -> List[DataEntry]:
        """Load claims from JSON file."""
        return JSONManager.load_from_json(file_path, DataEntry)
    
    @staticmethod
    def load_preprocessed_claims(file_path: Union[str, Path]) -> List[DataEntry]:
        """
        Load preprocessed claims from JSON file with support for DocRetrievalQuery structure.
        
        Args:
            file_path: Path to the JSON file containing claims
        
        Returns:
            List[DataEntry]: List of parsed DataEntry objects
        """
        def _convert_to_doc_retrieval_queries(queries_data: List[Dict]) -> List[DocRetrievalQuery]:
            return [
                DocRetrievalQuery(
                    related_to_reference=q["related_to_reference"],
                    main_query=q["main_query"],
                    rewritten_queries=q["rewritten_queries"]
                )
                for q in queries_data
            ]

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return [
            DataEntry(
                original_claim=entry["original_claim"],
                context_before=entry["context_before"],
                context_after=entry["context_after"],
                references=entry["references"],
                doc_retrieval_queries=_convert_to_doc_retrieval_queries(entry["doc_retrieval_queries"]),
                is_positive=entry["is_positive"]
            )
            for entry in data
        ]

    @staticmethod
    def load_references(file_path: Union[str, Path]) -> List[Reference]:
        """Load references from JSON file."""
        return JSONManager.load_from_json(file_path, Reference)

    @staticmethod
    def load_negative_claims(directory: Union[str, Path]) -> List[DataEntry]:
        """Load claims from JSON files in the specified directory."""
        claims = []
        directory = Path(directory)
        
        for file in directory.glob("*.json"):
            claims.extend(JSONManager.load_claims(file))
        
        return claims

    @staticmethod
    def load_results(file_path: Union[str, Path]) -> List[DataEntry]:
        """Load results from JSON file and convert to DataEntry objects with nested Result objects."""
        def _convert_to_doc_retrieval_queries(queries_data: List[Dict]) -> List[DocRetrievalQuery]:
            return [
                DocRetrievalQuery(
                    related_to_reference=q["related_to_reference"],
                    main_query=q["main_query"],
                    rewritten_queries=q["rewritten_queries"]
                )
                for q in queries_data
            ]

        def _convert_to_results(results_data: Optional[List[Dict]]) -> Optional[List[Result]]:
            if not results_data:
                return None
            return [
                Result(
                    original_reference=r["original_reference"],
                    query_used_for_retrieval=r["query_used_for_retrieval"],
                    retrieved_docs_from_sources=r["retrieved_docs_from_sources"],
                    predicted_reference=r["predicted_reference"],
                    ranking_metrics=RankingMetrics(**r["ranking_metrics"]) if r.get("ranking_metrics") else None
                )
                for r in results_data
            ]

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return [
            DataEntry(
                original_claim=entry["original_claim"],
                context_before=entry["context_before"],
                context_after=entry["context_after"],
                references=entry["references"],
                doc_retrieval_queries=_convert_to_doc_retrieval_queries(entry["doc_retrieval_queries"]),
                is_positive=entry["is_positive"],
                results=_convert_to_results(entry.get("results"))
            )
            for entry in data
        ]