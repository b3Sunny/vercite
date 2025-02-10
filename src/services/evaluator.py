"""
Evaluation metrics and analysis for claim processing.
"""

import matplotlib
matplotlib.use('Agg')

from typing import List, Dict, Optional
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from src.utils.logger_mixin import LoggerMixin
from src.models.models import EvaluationMetrics, Result, DataEntry, BaseMetrics, ConfusionMatrix, ProcessingMetrics, RankingMetrics, TestcaseRankingMetrics
from src.models.models import create_empty_processing_metrics, create_empty_confusion_matrix, create_empty_base_metrics, create_empty_testcase_ranking_metrics
from pathlib import Path
from config.config_loader import LangchainConfig

class ResultProcessor(LoggerMixin):
    """Core processor for evaluation metrics calculation."""
    
    def __init__(self, results: List[DataEntry], output_dir: Path, langchain_config: LangchainConfig):
        self.results = results
        self.output_dir = output_dir
        self.skipped_claims = 0
        self.retrieval_stats = []
        self.results_positive_claims: List[Result] = []
        self.results_negative_claims: List[Result] = []
        self.results_all_valid_claims: List[Result] = [] # combined positive and negative claims
        self.processing_metrics: ProcessingMetrics = create_empty_processing_metrics()
        self.confusion_matrix: ConfusionMatrix = create_empty_confusion_matrix()
        self.base_metrics: BaseMetrics = create_empty_base_metrics()
        self.testcase_ranking_metrics: TestcaseRankingMetrics = create_empty_testcase_ranking_metrics()
        self.langchain_config = langchain_config

    def process(self) -> EvaluationMetrics:
        """Main processing method returning EvaluationMetrics object."""
        
        try:    
            # split results into positive and negative claims + count invalid results
            self.processing_metrics = self._prepare_claim_results(self.results)
            self.confusion_matrix = self._calculate_confusion_metrics(
                self.results_positive_claims, self.results_negative_claims
            )
            self.base_metrics = self._calculate_base_metrics(self.confusion_matrix)
            
            # Calculate ranking metrics only for positive claims
            self.testcase_ranking_metrics = RankingMetricsAggregator.aggregate_metrics(
                self.results_positive_claims
            )
            
            # Combine all metrics
            evaluation_metrics = EvaluationMetrics(
                processing_metrics=self.processing_metrics,
                base_metrics=self.base_metrics,
                testcase_ranking_metrics=self.testcase_ranking_metrics
            )

        except Exception as e:
            self.logger.error(f"Error during processing evaluation metrics: {e}")
            raise

        return evaluation_metrics
        
    def _evaluate_doc_retrieval(self, citation: str, retrieved_sources: List[int]) -> Dict:
        """
        Evaluate retrieval performance for a single citation.
        
        Formulas:
        - found: bool(citation ∈ retrieved_sources)
        - occurrences: count(citation in retrieved_sources)
        - retrieval_ratio: occurrences / k, where k = len(retrieved_sources)
        """
        try:
            citation = int(citation)
            k = len(retrieved_sources)  # Dynamic k from retrieved sources
            occurrences = retrieved_sources.count(citation)
            
            return {
                "found": occurrences > 0,
                "occurrences": occurrences,
                "retrieval_ratio": occurrences / k if k > 0 else 0
            }
        except ValueError:
            self.logger.warning(f"Invalid citation format: {citation}")
            return {
                "found": False,
                "occurrences": 0,
                "retrieval_ratio": 0
            }

    def _prepare_claim_results(self, claims: List[DataEntry]) -> ProcessingMetrics:
        """
        Process and separate claims into positive and negative lists.
        Counts skipped claims (those without results).
        """
        for claim in claims:
            if not claim.results:
                self.logger.warning(f"Skipping claim without results: {claim.original_claim[:100]}...")
                self.skipped_claims += 1
                continue
            
            for result in claim.results:
                # Separate into positive and negative claims
                if claim.is_positive:
                    self.results_positive_claims.append(result)
                else:
                    self.results_negative_claims.append(result)
                
        # After processing all results for this claim
        self.results_all_valid_claims = self.results_positive_claims + self.results_negative_claims

        # processing metrics
        return ProcessingMetrics(
            population=len(self.results_all_valid_claims),
            skipped_claim=self.skipped_claims,
            positive_population=len(self.results_positive_claims),
            negative_population=len(self.results_negative_claims),
            langchain_config=self.langchain_config
        )

    def _calculate_confusion_metrics(self, positive_results: List[Result], negative_results: List[Result]) -> ConfusionMatrix:
        """Calculate confusion matrix metrics."""
        y_true = []
        y_pred = []

        # Process positive claims
        for result in positive_results:
            if result.predicted_reference.startswith(f"[{result.original_reference}]"):
                # Correct source identified
                y_true.append(1) # true
                y_pred.append(1) # positive
            else:
                # Wrong source or none
                y_true.append(1) # false
                y_pred.append(0) # negative

        # Process negative claims
        for result in negative_results:
            # It's a negative claim
            if result.predicted_reference == "none":
                # Correctly identified as not verifiable
                y_true.append(0) # true
                y_pred.append(0) # negative
            else:
                # Incorrectly identified as verifiable
                y_true.append(0) # false
                y_pred.append(1) # positive

        # Calculate confusion matrix using sklearn
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Create our ConfusionMatrix object
        return ConfusionMatrix(
            true_positive=int(tp),
            false_negative=int(fn),
            false_positive=int(fp),
            true_negative=int(tn)
        )

    def _calculate_base_metrics(self, confusion_matrix: ConfusionMatrix) -> BaseMetrics:
        """Calculate system metrics (Verification)."""
        try:
            tp = confusion_matrix.true_positive
            fn = confusion_matrix.false_negative
            fp = confusion_matrix.false_positive
            tn = confusion_matrix.true_negative

            positive_population = tp + fn
            negative_population = fp + tn
            total_population = positive_population + negative_population

            predicted_positive = tp + fp
            predicted_negative = fn + tn
            correct_prediction = tp + tn
            incorrect_prediction = fn + fp
            
            # Calculate basic metrics
            accuracy = correct_prediction / total_population
            precision = tp / predicted_positive if predicted_positive > 0 else 0
            recall = tp / positive_population if positive_population > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print("Verification performence:")
            print(f"\tAccuracy: {accuracy*100:.1f} %")
            print(f"\tPrecision: {precision*100:.1f} %")
            print(f"\tRecall: {recall*100:.1f} %")
            print(f"\tF1 Score: {f1*100:.1f} %")

            self.base_metrics = BaseMetrics(
                confusion_matrix=confusion_matrix,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1
            )
        except Exception as e:
            self.logger.error(f"Error during calculation of base metrics: {e}")
            raise

        return self.base_metrics

    def _create_confusion_matrix_plot(self) -> None:
        """Create and save confusion matrix visualization.""" # TODO: not used, can be deleted
        _, confusion_matrix_data = self._calculate_metrics()
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_data, 
                   annot=True, 
                   fmt='d',
                   xticklabels=['Negative (none)', 'Positive (match)'],
                   yticklabels=['Negative (synthetic)', 'Positive (real)'])
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()

class RankingMetricsCalculator:
    """Calculator for ranking-based metrics: Precision@k, NDCG@k, MRR, etc."""

    @staticmethod
    def calculate_precision_at_k(retrieved_docs: List[int], target_doc: int, k: Optional[int] = None) -> float:
        """Calculate Precision for a single query."""
        if k is None:
            k = len(retrieved_docs)
        k = min(k, len(retrieved_docs))
        
        relevant_docs = sum(1 for doc in retrieved_docs[:k] if doc == target_doc)
        return relevant_docs / k if k > 0 else 0.0

    @staticmethod
    def calculate_ndcg_at_k(retrieved_docs: List[int], target_doc: int, k: Optional[int] = None) -> float:
        """Calculate NDCG for a single query."""
        if k is None:
            k = len(retrieved_docs)
        k = min(k, len(retrieved_docs))
        
        # Create relevance scores (1 for match, 0 for no match)
        rel = [1 if doc == target_doc else 0 for doc in retrieved_docs[:k]]
        
        # Calculate DCG
        dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel))
        
        # Calculate IDCG (ideal DCG)
        ideal_rel = sorted(rel, reverse=True)
        idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_rel))
        
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_mrr(retrieved_docs: List[int], target_doc: int) -> float:
        """Calculate Mean Reciprocal Rank (MRR) for a single query."""
        try:
            rank = retrieved_docs.index(target_doc) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0

    @staticmethod
    def calculate_hit_rate_at_k(retrieved_docs: List[int], target_doc: int, k: Optional[int] = None) -> float:
        """Calculate Hit Rate for a single query."""
        if k is None:
            k = len(retrieved_docs)
        k = min(k, len(retrieved_docs))
        
        return 1.0 if target_doc in retrieved_docs[:k] else 0.0

    @staticmethod
    def calculate_all_metrics(retrieved_docs: List[int], target_doc: int, k: Optional[int] = None) -> RankingMetrics:
        """Calculate all ranking metrics for a single query."""
        return RankingMetrics(
            precision_at_k=RankingMetricsCalculator.calculate_precision_at_k(retrieved_docs, target_doc, k),
            ndcg_at_k=RankingMetricsCalculator.calculate_ndcg_at_k(retrieved_docs, target_doc, k),
            mrr=RankingMetricsCalculator.calculate_mrr(retrieved_docs, target_doc),
            hit_rate_at_k=RankingMetricsCalculator.calculate_hit_rate_at_k(retrieved_docs, target_doc, k)
        )

class RankingMetricsAggregator:
    """Aggregates ranking metrics across all claims in a testcase."""
    
    @staticmethod
    def _calculate_overall_score(metrics: Dict[str, float]) -> float:
        """
        Calculate weighted overall score (0-100).
        
        Weights:
        - NDCG: 40% (best measure of ranking quality)
        - MRR: 30% (important for first relevant result)
        - Precision: 20% (general retrieval quality)
        - Hit Rate: 10% (basic success measure)
        """
        weights = {
            'ndcg': 0.4,
            'mrr': 0.3,
            'precision': 0.2,
            'hit_rate': 0.1
        }
        
        weighted_sum = (
            metrics['avg_ndcg_at_k'] * weights['ndcg'] +
            metrics['avg_mrr'] * weights['mrr'] +
            metrics['avg_precision_at_k'] * weights['precision'] +
            metrics['avg_hit_rate_at_k'] * weights['hit_rate']
        )
        print(f"\tOverall score: {weighted_sum*100:.1f} %")
        return weighted_sum * 100  # Convert to 0-100 scale

    @staticmethod
    def _generate_performance_summary(metrics: Dict[str, float], score: float) -> str:
        """Generate human-readable performance summary."""
        summary_parts = []
        
        # Overall performance category
        if score >= 90:
            summary_parts.append("Excellent retrieval performance.")
        elif score >= 80:
            summary_parts.append("Very good retrieval performance.")
        elif score >= 70:
            summary_parts.append("Good retrieval performance.")
        elif score >= 60:
            summary_parts.append("Fair retrieval performance.")
        else:
            summary_parts.append("Poor retrieval performance.")

        # Specific metric insights
        if metrics['avg_ndcg_at_k'] > 0.8:
            summary_parts.append("Ranking quality is very high.")
        elif metrics['avg_ndcg_at_k'] < 0.4:
            summary_parts.append("Significant issues with ranking quality.")

        if metrics['avg_mrr'] > 0.8:
            summary_parts.append("Relevant documents consistently appear at top positions.")
        elif metrics['avg_mrr'] < 0.3:
            summary_parts.append("Relevant documents often appear too low in results.")

        perfect_ratio = metrics['num_perfect_retrievals'] / metrics['total_queries']
        if perfect_ratio > 0.7:
            summary_parts.append("High proportion of perfect retrievals.")
        
        failed_ratio = metrics['num_failed_retrievals'] / metrics['total_queries']
        if failed_ratio > 0.3:
            summary_parts.append("Concerning number of failed retrievals.")

        return " ".join(summary_parts)

    @staticmethod
    def aggregate_metrics(positive_claims: List[Result]) -> TestcaseRankingMetrics:
        """
        Aggregate ranking metrics across all positive claims.
        
        Args:
            positive_claims: List of Result objects from positive claims only
        """
        metrics_list = []
        perfect_retrievals = 0
        failed_retrievals = 0
        
        # Collect metrics from positive claims
        for result in positive_claims:
            if result.ranking_metrics:
                metrics_list.append(result.ranking_metrics)
                
                # Count perfect and failed retrievals
                if result.ranking_metrics.precision_at_k == 1.0:
                    perfect_retrievals += 1
                if all(getattr(result.ranking_metrics, metric) == 0.0 
                      for metric in ['precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate_at_k']):
                    failed_retrievals += 1
            else:
                print(f"Warning: No ranking metrics for result: {result.query_used_for_retrieval}")

        total_queries = len(metrics_list)
        if total_queries == 0:
            print("Warning: No valid ranking metrics found in positive claims")
            return TestcaseRankingMetrics()

        # Calculate averages
        avg_metrics = {
            'avg_precision_at_k': sum(m.precision_at_k for m in metrics_list) / total_queries,
            'avg_ndcg_at_k': sum(m.ndcg_at_k for m in metrics_list) / total_queries,
            'avg_mrr': sum(m.mrr for m in metrics_list) / total_queries,
            'avg_hit_rate_at_k': sum(m.hit_rate_at_k for m in metrics_list) / total_queries,
            'num_perfect_retrievals': perfect_retrievals,
            'num_failed_retrievals': failed_retrievals,
            'total_queries': total_queries
        }

        # Debug logging
        print("Ranking performence:")
        print(f"\tAverage Precision: {avg_metrics['avg_precision_at_k']*100:.1f} %")
        print(f"\tAverage NDCG: {avg_metrics['avg_ndcg_at_k']*100:.1f} %")
        print(f"\tAverage MRR: {avg_metrics['avg_mrr']*100:.1f} %")
        print(f"\tAverage Hit Rate: {avg_metrics['avg_hit_rate_at_k']*100:.1f} %")
        print(f"\tPerfect retrievals: {perfect_retrievals}, Failed retrievals: {failed_retrievals}")

        # Calculate min/max
        avg_metrics.update({
            'max_precision_at_k': max(m.precision_at_k for m in metrics_list),
            'min_precision_at_k': min(m.precision_at_k for m in metrics_list),
            'max_ndcg_at_k': max(m.ndcg_at_k for m in metrics_list),
            'min_ndcg_at_k': min(m.ndcg_at_k for m in metrics_list),
        })

        # Calculate overall score
        overall_score = RankingMetricsAggregator._calculate_overall_score(avg_metrics)

        # Generate performance summary
        performance_summary = RankingMetricsAggregator._generate_performance_summary(
            avg_metrics, overall_score
        )
        print(f"\tPerformance summary: {performance_summary}\n")

        return TestcaseRankingMetrics(
            **avg_metrics,
            overall_score=overall_score,
            performance_summary=performance_summary
        )

class EvaluatorService(LoggerMixin):
    """Service layer for evaluation operations."""
    
    def __init__(self, results: List[DataEntry], output_dir: Path, langchain_config: LangchainConfig):
        self.output_dir = output_dir
        self._processor = ResultProcessor(results, output_dir, langchain_config)
        
    
    def run(self) -> EvaluationMetrics:
        """Execute the evaluation process."""
        try:
            self.logger.info("Starting evaluation process...")
            metrics = self._processor.process()
            self.logger.info(f"Evaluation completed. Results saved to: {self.output_dir}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise

def get_evaluator_service(results: List[DataEntry], output_dir: Path, langchain_config: LangchainConfig) -> EvaluatorService:
    """Factory function for EvaluatorService."""
    return EvaluatorService(results, output_dir, langchain_config)
    
