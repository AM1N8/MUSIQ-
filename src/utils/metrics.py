import numpy as np
from typing import List, Dict
import torch

class RecommendationMetrics:
    """Calculate various recommendation quality metrics"""
    
    @staticmethod
    def ndcg_at_k(recommendations: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K"""
        dcg = 0.0
        idcg = 0.0
        
        for i in range(min(k, len(recommendations))):
            if recommendations[i] in ground_truth:
                dcg += 1.0 / np.log2(i + 2)
        
        for i in range(min(k, len(ground_truth))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def ctr(clicks: List[bool]) -> float:
        """Calculate Click-Through Rate"""
        return sum(clicks) / len(clicks) if clicks else 0.0
    
    @staticmethod
    def diversity(recommendations: np.ndarray, item_embeddings: np.ndarray) -> float:
        """Calculate recommendation diversity using cosine distance"""
        if len(recommendations) < 2:
            return 0.0
        
        rec_embeddings = item_embeddings[recommendations]
        similarities = []
        
        for i in range(len(rec_embeddings)):
            for j in range(i + 1, len(rec_embeddings)):
                sim = np.dot(rec_embeddings[i], rec_embeddings[j])
                sim /= (np.linalg.norm(rec_embeddings[i]) * np.linalg.norm(rec_embeddings[j]) + 1e-8)
                similarities.append(sim)
        
        return 1.0 - np.mean(similarities) if similarities else 0.0
    
    @staticmethod
    def catalog_coverage(recommendations: np.ndarray, catalog_size: int) -> float:
        """Calculate percentage of catalog covered"""
        unique_items = len(np.unique(recommendations))
        return unique_items / catalog_size if catalog_size > 0 else 0.0

