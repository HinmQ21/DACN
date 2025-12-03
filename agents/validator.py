"""Validator Agent - Kiểm tra tính nhất quán và Choice Shuffling Ensemble."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, List, Optional, Callable
from utils.config import Config
from utils.ensemble import EnsembleManager, ChoiceShuffler, EnsembleVoter
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor


class ValidatorAgent:
    """
    Agent kiểm chứng tính nhất quán và độ tin cậy.
    Tích hợp Choice Shuffling Ensemble từ Medprompt.
    """
    
    def __init__(self, enable_ensemble: bool = None):
        """
        Initialize Validator Agent.
        
        Args:
            enable_ensemble: Whether to enable choice shuffling ensemble (uses config if None)
        """
        # Use validator-specific configuration
        self.llm = ChatGoogleGenerativeAI(**Config.get_llm_config('validator'))
        
        # Ensemble configuration
        medprompt_config = Config.get_medprompt_config()
        self.enable_ensemble = enable_ensemble if enable_ensemble is not None else medprompt_config['enable_ensemble']
        self.num_variants = medprompt_config['ensemble_variants']
        self.confidence_threshold = medprompt_config['ensemble_confidence_threshold']
        
        # Initialize ensemble manager
        self.ensemble_manager = EnsembleManager(
            num_variants=self.num_variants,
            use_choice_shuffling=self.enable_ensemble,
            confidence_threshold=self.confidence_threshold
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional validator in a medical system.
Your task is to check and evaluate:

1. CONSISTENCY:
   - Are the web search and reasoning results consistent?
   - Are there any contradictions between information sources?

2. EVIDENCE QUALITY:
   - Reliability of information sources (0-1)
   - Timeliness of information
   - Relevance to the question

3. REASONING QUALITY:
   - Is the reasoning logic rigorous? (0-1)
   - Are there any missing steps?

Return evaluation in JSON format:
{{
    "is_consistent": true/false,
    "consistency_explanation": "explanation",
    "evidence_quality": 0.0-1.0,
    "evidence_issues": ["issue 1", "issue 2"],
    "reasoning_quality": 0.0-1.0,
    "reasoning_issues": ["issue 1", "issue 2"],
    "conflicts": ["conflict 1", "conflict 2"],
    "overall_confidence": 0.0-1.0,
    "recommendation": "proceed/revise/reject"
}}"""),
            ("human", """Question: {question}

Web Search Results:
{web_search_result}

Reasoning Results:
{reasoning_result}

Please evaluate and validate.""")
        ])
    
    def validate(
        self, 
        question: str,
        web_search_result: Dict[str, Any],
        reasoning_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Kiểm chứng kết quả từ web search và reasoning.
        
        Args:
            question: Câu hỏi gốc
            web_search_result: Kết quả từ web search agent
            reasoning_result: Kết quả từ reasoning agent
            
        Returns:
            Dictionary chứa kết quả kiểm chứng
        """
        # Format inputs for validation
        web_summary = web_search_result.get('synthesis', 'No web search results')
        reasoning_summary = reasoning_result.get('raw_output', 'No reasoning results')
        
        validation_input = {
            "question": question,
            "web_search_result": web_summary,
            "reasoning_result": reasoning_summary
        }
        
        result = (self.prompt | self.llm).invoke(validation_input)
        result_text = result.content if hasattr(result, 'content') else str(result)
        
        # Parse JSON response
        try:
            # Extract JSON from markdown code block if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            validation = json.loads(result_text)
        except json.JSONDecodeError:
            # Fallback validation
            validation = {
                "is_consistent": True,
                "consistency_explanation": "Unable to parse validation result",
                "evidence_quality": 0.7,
                "evidence_issues": [],
                "reasoning_quality": 0.7,
                "reasoning_issues": [],
                "conflicts": [],
                "overall_confidence": 0.7,
                "recommendation": "proceed"
            }
        
        return validation
    
    def is_multiple_choice(self, options: Any) -> bool:
        """Check if the question has valid multiple choice options."""
        return self.ensemble_manager.is_multiple_choice(options)
    
    def create_shuffled_variants(
        self, 
        question: str, 
        options: Dict[str, str],
        k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Create shuffled variants of a multiple-choice question.
        
        Args:
            question: The question text
            options: Answer options
            k: Number of variants (uses config if None)
            
        Returns:
            List of question variants with shuffled options
        """
        if k is not None:
            self.ensemble_manager.num_variants = k
        
        return self.ensemble_manager.prepare_variants(question, options)
    
    def validate_with_ensemble(
        self,
        question: str,
        options: Dict[str, str],
        reasoning_func: Callable,
        web_search_result: Dict[str, Any] = None,
        few_shot_examples: str = None,
        k: int = None
    ) -> Dict[str, Any]:
        """
        Validate answer using Choice Shuffling Ensemble.
        
        Args:
            question: The question text
            options: Answer options
            reasoning_func: Function to call for reasoning (takes question, options, context)
            web_search_result: Web search results for context
            few_shot_examples: Few-shot examples for CoT
            k: Number of variants to use
            
        Returns:
            Ensemble validation result
        """
        if not self.is_multiple_choice(options):
            # Not a multiple choice question - use standard validation
            reasoning_result = reasoning_func(question, options, web_search_result, few_shot_examples)
            validation = self.validate(question, web_search_result or {}, reasoning_result)
            return {
                'answer': reasoning_result.get('conclusion', ''),
                'confidence': validation.get('overall_confidence', 0.7),
                'validation': validation,
                'ensemble_used': False
            }
        
        if not self.enable_ensemble:
            # Ensemble disabled - use single prediction
            reasoning_result = reasoning_func(question, options, web_search_result, few_shot_examples)
            validation = self.validate(question, web_search_result or {}, reasoning_result)
            return {
                'answer': reasoning_result.get('conclusion', ''),
                'confidence': validation.get('overall_confidence', 0.7),
                'validation': validation,
                'ensemble_used': False
            }
        
        # Create shuffled variants
        variants = self.create_shuffled_variants(question, options, k)
        
        # Run reasoning on each variant
        predictions = []
        
        for variant in variants:
            try:
                # Run reasoning with shuffled options
                result = reasoning_func(
                    variant['question'],
                    variant['options'],
                    web_search_result,
                    few_shot_examples
                )
                
                # Extract answer
                answer = self._extract_answer_from_result(result)
                
                predictions.append({
                    'answer': answer,
                    'reverse_mapping': variant['reverse_mapping'],
                    'confidence': 1.0,  # Equal weight for each prediction
                    'variant_id': variant['variant_id'],
                    'raw_result': result
                })
                
            except Exception as e:
                print(f"[Validator] Error processing variant {variant['variant_id']}: {e}")
                continue
        
        if not predictions:
            return {
                'answer': '',
                'confidence': 0.0,
                'consistency': 0.0,
                'ensemble_used': True,
                'error': 'All variants failed'
            }
        
        # Aggregate predictions using voting
        aggregated = self.ensemble_manager.aggregate_predictions(predictions)
        
        # Add validation result
        if predictions:
            # Validate using the first (original) prediction
            original_result = predictions[0].get('raw_result', {})
            validation = self.validate(question, web_search_result or {}, original_result)
            aggregated['validation'] = validation
        
        aggregated['ensemble_used'] = True
        aggregated['num_variants'] = len(predictions)
        
        return aggregated
    
    def validate_with_ensemble_parallel(
        self,
        question: str,
        options: Dict[str, str],
        reasoning_func: Callable,
        web_search_result: Dict[str, Any] = None,
        few_shot_examples: str = None,
        k: int = None,
        max_workers: int = 3
    ) -> Dict[str, Any]:
        """
        Validate with ensemble using parallel execution.
        
        Args:
            question: The question
            options: Answer options
            reasoning_func: Reasoning function to call
            web_search_result: Web search results
            few_shot_examples: Few-shot examples
            k: Number of variants
            max_workers: Maximum parallel workers
            
        Returns:
            Ensemble validation result
        """
        if not self.is_multiple_choice(options) or not self.enable_ensemble:
            return self.validate_with_ensemble(
                question, options, reasoning_func, 
                web_search_result, few_shot_examples, k
            )
        
        # Create shuffled variants
        variants = self.create_shuffled_variants(question, options, k)
        
        def process_variant(variant):
            """Process a single variant."""
            try:
                result = reasoning_func(
                    variant['question'],
                    variant['options'],
                    web_search_result,
                    few_shot_examples
                )
                
                answer = self._extract_answer_from_result(result)
                
                return {
                    'answer': answer,
                    'reverse_mapping': variant['reverse_mapping'],
                    'confidence': 1.0,
                    'variant_id': variant['variant_id'],
                    'raw_result': result
                }
            except Exception as e:
                print(f"[Validator] Error in variant {variant['variant_id']}: {e}")
                return None
        
        # Run in parallel
        predictions = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_variant, v) for v in variants]
            for future in futures:
                result = future.result()
                if result:
                    predictions.append(result)
        
        if not predictions:
            return {
                'answer': '',
                'confidence': 0.0,
                'consistency': 0.0,
                'ensemble_used': True,
                'error': 'All variants failed'
            }
        
        # Aggregate
        aggregated = self.ensemble_manager.aggregate_predictions(predictions)
        
        # Validate
        if predictions:
            original_result = predictions[0].get('raw_result', {})
            validation = self.validate(question, web_search_result or {}, original_result)
            aggregated['validation'] = validation
        
        aggregated['ensemble_used'] = True
        aggregated['num_variants'] = len(predictions)
        
        return aggregated
    
    def _extract_answer_from_result(self, result: Dict[str, Any]) -> str:
        """
        Extract the answer letter from reasoning result.
        
        Args:
            result: Reasoning result dictionary
            
        Returns:
            Answer letter (A, B, C, D, etc.)
        """
        import re
        
        # Try conclusion first
        conclusion = result.get('conclusion', '')
        
        # Look for answer pattern
        patterns = [
            r'\b([A-E])\b(?:\s*[-:)]|\s+is\s+correct|\s+is\s+the\s+answer)',
            r'(?:answer|correct|option)(?:\s+is)?[:\s]+([A-E])\b',
            r'^([A-E])\b',
            r'\b([A-E])\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, conclusion, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Try raw output
        raw_output = result.get('raw_output', '')
        for pattern in patterns:
            match = re.search(pattern, raw_output, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return conclusion[:1].upper() if conclusion else ''
    
    def calculate_ensemble_metrics(
        self, 
        predictions: List[str]
    ) -> Dict[str, float]:
        """
        Calculate metrics for ensemble predictions.
        
        Args:
            predictions: List of predicted answers
            
        Returns:
            Dictionary of metrics
        """
        if not predictions:
            return {
                'consistency': 0.0,
                'entropy': 0.0,
                'agreement_ratio': 0.0
            }
        
        # Consistency
        consistency = EnsembleVoter.consistency_score(predictions)
        
        # Calculate entropy
        from collections import Counter
        import math
        
        counter = Counter(predictions)
        total = len(predictions)
        entropy = 0.0
        
        for count in counter.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        # Maximum possible entropy
        max_entropy = math.log2(len(counter)) if len(counter) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Agreement ratio (how much the top answer dominates)
        top_count = counter.most_common(1)[0][1]
        agreement_ratio = top_count / total
        
        return {
            'consistency': consistency,
            'entropy': normalized_entropy,
            'agreement_ratio': agreement_ratio
        }
