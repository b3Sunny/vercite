"""
Data models for JSON management using dataclasses.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from config.config_loader import LangchainConfig

@dataclass
class DocRetrievalQuery:
    """ Format for query generation"""
    related_to_reference: str
    main_query: str
    rewritten_queries: List[str]

    def to_dict(self):
        return {
            "related_to_reference": self.related_to_reference,
            "main_query": self.main_query,
            "rewritten_queries": self.rewritten_queries
        }

@dataclass
class RankingMetrics:
    """ Ranking metrics for a document retrieval"""
    precision_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    mrr: float = 0.0
    hit_rate_at_k: float = 0.0

@dataclass
class Result:
    """ Result for a claim"""
    original_reference: str
    query_used_for_retrieval: str
    retrieved_docs_from_sources: List[int]
    predicted_reference: str
    prediction_validated_by_human: int = 0 # 0: not verified, 1: verified (correct), 2: verified (incorrect)
    ranking_metrics: Optional[RankingMetrics] = None

    def to_dict(self) -> Dict:
        return {
            "original_reference": self.original_reference,
            "query_used_for_retrieval": self.query_used_for_retrieval,
            "retrieved_docs_from_sources": self.retrieved_docs_from_sources,
            "predicted_reference": self.predicted_reference,
            "prediction_validated_by_human": self.prediction_validated_by_human,
            "ranking_metrics": asdict(self.ranking_metrics) if self.ranking_metrics else None
        }

@dataclass
class DetailedResult:
    """ Format for detailed prompt"""
    source: str
    page: str
    reason: str
    relevant_text: str
    confidence: float


@dataclass
class DataEntry:
    """ Data entry for a claim"""
    original_claim: str 
    context_before: List[str]  
    context_after: List[str]  
    references: List[str]  
    doc_retrieval_queries: List[DocRetrievalQuery]
    is_positive: bool 

    results: Optional[List[Result]] = None

    def to_dict(self) -> Dict:
        return {
            "original_claim": self.original_claim,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "references": self.references,
            "doc_retrieval_queries": [query.to_dict() for query in self.doc_retrieval_queries],
            "is_positive": self.is_positive,
            "results": [result.to_dict() for result in self.results] if self.results else None
        }

@dataclass
class Reference:
    """ ArXiv reference"""
    number: str
    text: str
    arxiv_id: str
    pdf_link: str

@dataclass
class ProcessingMetrics:
    """ Processing metrics for a testcase"""
    population: int
    skipped_claim: int
    positive_population: int
    negative_population: int
    langchain_config: LangchainConfig

@dataclass
class ConfusionMatrix:
    """ Confusion matrix for verification"""
    true_positive: int
    false_negative: int
    false_positive: int
    true_negative: int

@dataclass
class BaseMetrics:
    """ System metrics (Verification)"""
    confusion_matrix: ConfusionMatrix
    accuracy: float
    precision: float
    recall: float
    f1_score: float

@dataclass
class TestcaseRankingMetrics:
    """Aggregated ranking metrics across all positives in a testcase"""
    # Average metrics
    avg_precision_at_k: float = 0.0
    avg_ndcg_at_k: float = 0.0
    avg_mrr: float = 0.0
    avg_hit_rate_at_k: float = 0.0
    
    # Best/worst case
    max_precision_at_k: float = 0.0
    min_precision_at_k: float = 1.0
    max_ndcg_at_k: float = 0.0
    min_ndcg_at_k: float = 1.0
    
    # Distribution info
    num_perfect_retrievals: int = 0 
    num_failed_retrievals: int = 0   
    total_queries: int = 0
    
    # Weighted combined score (0-100)
    overall_score: float = 0.0
    
    # Interpretation
    performance_summary: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass 
class EvaluationMetrics:
    """Combined evaluation metrics object"""
    processing_metrics: ProcessingMetrics
    base_metrics: BaseMetrics
    testcase_ranking_metrics: TestcaseRankingMetrics

@dataclass
class SimpleClaim:
    """Simple claim structure with just the essential text and source for negatives."""
    original_claim: str

    def to_dict(self):
        return {
            "original_claim": self.original_claim,
        }
    
""" Factory methods for creating empty models"""
def create_empty_processing_metrics() -> ProcessingMetrics:
   return ProcessingMetrics(0, 0, 0, 0, None)
def create_empty_confusion_matrix() -> ConfusionMatrix:
   return ConfusionMatrix(0, 0, 0, 0)
def create_empty_base_metrics() -> BaseMetrics:
   return BaseMetrics(create_empty_confusion_matrix(), 0, 0, 0, 0)
def create_empty_ranking_metrics() -> RankingMetrics:
   return RankingMetrics(0, 0, 0, 0)
def create_empty_testcase_ranking_metrics() -> TestcaseRankingMetrics:
   return TestcaseRankingMetrics()