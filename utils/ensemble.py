"""Ensemble Module - Choice shuffling and voting mechanisms for robust predictions."""

import random
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import numpy as np


class ChoiceShuffler:
    """
    Implements choice shuffling for multiple-choice questions.
    Creates variants with different option orderings to reduce position bias.
    """
    
    @staticmethod
    def shuffle_options(
        options: Dict[str, str],
        seed: int = None
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Shuffle the options while maintaining A, B, C, D labels.
        
        Args:
            options: Original options dict (e.g., {"A": "option1", "B": "option2", ...})
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (shuffled_options, mapping) where mapping shows original -> new label
        """
        if seed is not None:
            random.seed(seed)
        
        # Get original labels and values
        labels = list(options.keys())
        values = list(options.values())
        
        # Create shuffled indices
        indices = list(range(len(values)))
        random.shuffle(indices)
        
        # Create new options with shuffled values
        shuffled_options = {}
        mapping = {}  # original_label -> new_label
        reverse_mapping = {}  # new_label -> original_label
        
        for new_idx, orig_idx in enumerate(indices):
            new_label = labels[new_idx]
            orig_label = labels[orig_idx]
            shuffled_options[new_label] = values[orig_idx]
            reverse_mapping[new_label] = orig_label
            mapping[orig_label] = new_label
        
        return shuffled_options, reverse_mapping
    
    @staticmethod
    def create_shuffled_variants(
        question: str,
        options: Dict[str, str],
        k: int = 5,
        base_seed: int = None
    ) -> List[Dict[str, Any]]:
        """
        Create k shuffled variants of a multiple-choice question.
        
        Args:
            question: The question text
            options: Original options dictionary
            k: Number of variants to create
            base_seed: Base random seed for reproducibility
            
        Returns:
            List of variant dictionaries with shuffled options and mappings
        """
        import math
        
        variants = []
        
        # Calculate maximum possible unique permutations
        num_options = len(options)
        max_permutations = math.factorial(num_options)
        
        # Limit k to max possible permutations
        k = min(k, max_permutations)
        
        # First variant is always the original
        variants.append({
            'question': question,
            'options': options.copy(),
            'reverse_mapping': {label: label for label in options.keys()},
            'is_original': True,
            'variant_id': 0
        })
        
        # Create k-1 shuffled variants with retry logic
        max_attempts = k * 10  # Try up to 10x to find unique variants
        attempt = 0
        variant_id = 1
        
        while len(variants) < k and attempt < max_attempts:
            seed = (base_seed + attempt) if base_seed is not None else None
            shuffled_options, reverse_mapping = ChoiceShuffler.shuffle_options(
                options, seed=seed
            )
            
            # Check if this variant is different from existing ones
            is_duplicate = any(
                shuffled_options == v['options'] for v in variants
            )
            
            if not is_duplicate:
                variants.append({
                    'question': question,
                    'options': shuffled_options,
                    'reverse_mapping': reverse_mapping,
                    'is_original': False,
                    'variant_id': variant_id
                })
                variant_id += 1
            
            attempt += 1
        
        return variants
    
    @staticmethod
    def map_answer_to_original(
        predicted_answer: str,
        reverse_mapping: Dict[str, str]
    ) -> str:
        """
        Map a predicted answer back to the original option label.
        
        Args:
            predicted_answer: The predicted answer label (e.g., "A", "B")
            reverse_mapping: Mapping from shuffled to original labels
            
        Returns:
            Original answer label
        """
        # Extract just the letter if answer contains more text
        letter_match = re.match(r'^([A-Ea-e])', predicted_answer.strip())
        if letter_match:
            letter = letter_match.group(1).upper()
            return reverse_mapping.get(letter, letter)
        
        return predicted_answer


class EnsembleVoter:
    """
    Implements voting mechanisms for ensemble predictions.
    """
    
    @staticmethod
    def majority_vote(
        predictions: List[str],
        weights: List[float] = None
    ) -> Tuple[str, float]:
        """
        Perform weighted majority voting on predictions.
        
        Args:
            predictions: List of predicted answers
            weights: Optional weights for each prediction
            
        Returns:
            Tuple of (winning_answer, confidence_score)
        """
        if not predictions:
            return "", 0.0
        
        # Clean predictions (extract just the letter)
        cleaned_predictions = []
        for pred in predictions:
            match = re.match(r'^([A-Ea-e])', str(pred).strip())
            if match:
                cleaned_predictions.append(match.group(1).upper())
            else:
                cleaned_predictions.append(str(pred).strip().upper())
        
        if weights is None:
            weights = [1.0] * len(cleaned_predictions)
        
        # Weighted vote counting
        vote_weights = {}
        for pred, weight in zip(cleaned_predictions, weights):
            vote_weights[pred] = vote_weights.get(pred, 0) + weight
        
        # Find winner
        total_weight = sum(weights)
        winner = max(vote_weights.keys(), key=lambda x: vote_weights[x])
        confidence = vote_weights[winner] / total_weight if total_weight > 0 else 0.0
        
        return winner, confidence
    
    @staticmethod
    def consistency_score(predictions: List[str]) -> float:
        """
        Calculate consistency score based on agreement among predictions.
        
        Args:
            predictions: List of predicted answers
            
        Returns:
            Consistency score between 0 and 1
        """
        if not predictions:
            return 0.0
        
        # Clean predictions
        cleaned = []
        for pred in predictions:
            match = re.match(r'^([A-Ea-e])', str(pred).strip())
            if match:
                cleaned.append(match.group(1).upper())
            else:
                cleaned.append(str(pred).strip())
        
        # Count most common
        counter = Counter(cleaned)
        most_common_count = counter.most_common(1)[0][1]
        
        return most_common_count / len(predictions)
    
    @staticmethod
    def confidence_weighted_vote(
        predictions: List[str],
        confidences: List[float]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Perform confidence-weighted voting.
        
        Args:
            predictions: List of predicted answers
            confidences: Confidence scores for each prediction
            
        Returns:
            Tuple of (winner, final_confidence, vote_distribution)
        """
        if not predictions:
            return "", 0.0, {}
        
        # Clean predictions
        cleaned_predictions = []
        for pred in predictions:
            match = re.match(r'^([A-Ea-e])', str(pred).strip())
            if match:
                cleaned_predictions.append(match.group(1).upper())
            else:
                cleaned_predictions.append(str(pred).strip().upper())
        
        # Weighted aggregation
        vote_weights = {}
        for pred, conf in zip(cleaned_predictions, confidences):
            vote_weights[pred] = vote_weights.get(pred, 0) + conf
        
        # Normalize
        total = sum(vote_weights.values())
        if total > 0:
            vote_distribution = {k: v / total for k, v in vote_weights.items()}
        else:
            vote_distribution = {k: 1.0 / len(vote_weights) for k in vote_weights.keys()}
        
        # Find winner
        winner = max(vote_distribution.keys(), key=lambda x: vote_distribution[x])
        final_confidence = vote_distribution[winner]
        
        return winner, final_confidence, vote_distribution


class EnsembleManager:
    """
    High-level manager for ensemble-based prediction with choice shuffling.
    """
    
    def __init__(
        self,
        num_variants: int = 5,
        use_choice_shuffling: bool = True,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize the ensemble manager.
        
        Args:
            num_variants: Number of shuffled variants to create
            use_choice_shuffling: Whether to use choice shuffling
            confidence_threshold: Minimum confidence threshold for reliable predictions
        """
        self.num_variants = num_variants
        self.use_choice_shuffling = use_choice_shuffling
        self.confidence_threshold = confidence_threshold
        self.shuffler = ChoiceShuffler()
        self.voter = EnsembleVoter()
    
    def prepare_variants(
        self,
        question: str,
        options: Dict[str, str],
        base_seed: int = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare question variants for ensemble prediction.
        
        Args:
            question: Question text
            options: Options dictionary
            base_seed: Random seed for reproducibility
            
        Returns:
            List of question variants
        """
        if not self.use_choice_shuffling or not options:
            return [{
                'question': question,
                'options': options,
                'reverse_mapping': {k: k for k in options.keys()} if options else {},
                'is_original': True,
                'variant_id': 0
            }]
        
        return self.shuffler.create_shuffled_variants(
            question, options, k=self.num_variants, base_seed=base_seed
        )
    
    def aggregate_predictions(
        self,
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate predictions from multiple variants.
        
        Args:
            predictions: List of prediction results, each containing:
                - 'answer': predicted answer
                - 'reverse_mapping': mapping to original labels
                - 'confidence': optional confidence score
                
        Returns:
            Aggregated result dictionary
        """
        if not predictions:
            return {
                'answer': '',
                'confidence': 0.0,
                'consistency': 0.0,
                'all_predictions': [],
                'vote_distribution': {}
            }
        
        # Map predictions back to original labels
        original_answers = []
        confidences = []
        
        for pred in predictions:
            answer = pred.get('answer', '')
            reverse_mapping = pred.get('reverse_mapping', {})
            
            # Map to original
            original_answer = self.shuffler.map_answer_to_original(
                answer, reverse_mapping
            )
            original_answers.append(original_answer)
            
            # Collect confidence
            conf = pred.get('confidence', 1.0)
            confidences.append(conf if conf is not None else 1.0)
        
        # Perform voting
        if all(c == confidences[0] for c in confidences):
            # Equal weights - use simple majority
            final_answer, confidence = self.voter.majority_vote(original_answers)
            vote_distribution = dict(Counter(original_answers))
            total = sum(vote_distribution.values())
            vote_distribution = {k: v / total for k, v in vote_distribution.items()}
        else:
            # Weighted voting
            final_answer, confidence, vote_distribution = self.voter.confidence_weighted_vote(
                original_answers, confidences
            )
        
        # Calculate consistency
        consistency = self.voter.consistency_score(original_answers)
        
        return {
            'answer': final_answer,
            'confidence': confidence,
            'consistency': consistency,
            'all_predictions': original_answers,
            'vote_distribution': vote_distribution,
            'is_reliable': confidence >= self.confidence_threshold
        }
    
    def is_multiple_choice(self, options: Any) -> bool:
        """Check if the question has multiple choice options."""
        if options is None:
            return False
        if isinstance(options, dict) and len(options) > 1:
            return True
        if isinstance(options, list) and len(options) > 1:
            return True
        return False


def format_options_text(options: Dict[str, str]) -> str:
    """
    Format options dictionary as text for prompts.
    
    Args:
        options: Options dictionary
        
    Returns:
        Formatted string
    """
    if not options:
        return ""
    
    return "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])

