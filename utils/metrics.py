"""Metrics calculation for benchmark evaluation."""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        
    Returns:
        Dictionary containing metrics
    """
    # Calculate accuracy
    accuracy = accuracy_score(ground_truths, predictions)
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truths, 
        predictions, 
        average='weighted',
        zero_division=0
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }


def extract_answer_from_text(text: str, options: List[str] = None) -> str:
    """
    Extract answer from generated text.
    
    Args:
        text: Generated answer text
        options: List of possible options (e.g., ['A', 'B', 'C', 'D'])
        
    Returns:
        Extracted answer
    """
    text = text.strip().upper()
    
    if options:
        for option in options:
            if option.upper() in text[:10]:  # Check first 10 chars
                return option.upper()
    
    # Try to extract single letter answer
    if len(text) > 0 and text[0] in ['A', 'B', 'C', 'D', 'E']:
        return text[0]
    
    # For yes/no questions
    if 'YES' in text[:20]:
        return 'yes'
    if 'NO' in text[:20]:
        return 'no'
    
    return text.split()[0] if text else ""


def calculate_confidence_score(validator_result: Dict[str, Any]) -> float:
    """
    Calculate confidence score based on validator results.
    
    Args:
        validator_result: Dictionary containing validation results
        
    Returns:
        Confidence score between 0 and 1
    """
    score = 0.0
    
    # Check consistency
    if validator_result.get('is_consistent', False):
        score += 0.4
    
    # Check evidence quality
    evidence_quality = validator_result.get('evidence_quality', 0)
    score += evidence_quality * 0.3
    
    # Check reasoning quality
    reasoning_quality = validator_result.get('reasoning_quality', 0)
    score += reasoning_quality * 0.3
    
    return min(score, 1.0)

