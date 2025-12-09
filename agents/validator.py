"""Validator Agent - Kiểm tra tính nhất quán và Choice Shuffling Ensemble."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, List, Optional, Callable
from utils.config import Config
from utils.ensemble import EnsembleManager, ChoiceShuffler, EnsembleVoter
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import Counter


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
    
    def calculate_evidence_score(self, web_search_result: Dict[str, Any]) -> float:
        """
        Tính điểm objective dựa trên chất lượng nguồn thông tin.
        
        Args:
            web_search_result: Kết quả từ web search
            
        Returns:
            Evidence score từ 0.0 đến 1.0
        """
        total_sources = web_search_result.get('total_sources', 0)
        pubmed_results = web_search_result.get('pubmed_results', [])
        pubmed_count = len(pubmed_results)
        
        # Điểm theo số lượng nguồn (max 10 sources = 1.0)
        source_score = min(total_sources / 10.0, 1.0)
        
        # Bonus cho PubMed (nguồn y tế đáng tin cậy)
        pubmed_bonus = min(pubmed_count / 5.0, 0.3)  # Max +0.3 bonus
        
        # Penalty nếu không có nguồn
        if total_sources == 0:
            return 0.0
        
        evidence_score = min(source_score + pubmed_bonus, 1.0)
        return evidence_score
    
    def calculate_reasoning_score(self, reasoning_result: Dict[str, Any]) -> float:
        """
        Tính điểm objective dựa trên chất lượng suy luận.
        
        Args:
            reasoning_result: Kết quả từ reasoning agent
            
        Returns:
            Reasoning score từ 0.0 đến 1.0
        """
        reasoning_type = reasoning_result.get('reasoning_type', 'standard')
        structured = reasoning_result.get('structured', {})
        
        # Base score theo loại reasoning
        type_scores = {
            'cot': 0.8,              # Chain-of-Thought có cấu trúc
            'self_consistency': 0.9,  # Self-consistency cao nhất
            'standard': 0.6           # Standard thấp hơn
        }
        base_score = type_scores.get(reasoning_type, 0.5)
        
        # Bonus cho có conclusion rõ ràng
        conclusion = structured.get('conclusion', '')
        conclusion_bonus = 0.1 if len(conclusion) > 20 else 0.0
        
        # Bonus cho có analysis chi tiết
        analysis = structured.get('analysis', '')
        analysis_bonus = 0.1 if len(analysis) > 50 else 0.0
        
        reasoning_score = min(base_score + conclusion_bonus + analysis_bonus, 1.0)
        return reasoning_score
    
    def calculate_consistency_score(
        self,
        web_search_result: Dict[str, Any],
        reasoning_result: Dict[str, Any],
        answer: str = ""
    ) -> float:
        """
        Tính điểm consistency giữa các nguồn thông tin.
        
        Args:
            web_search_result: Kết quả từ web search
            reasoning_result: Kết quả từ reasoning
            answer: Đáp án đã chọn (optional)
            
        Returns:
            Consistency score từ 0.0 đến 1.0
        """
        web_synthesis = web_search_result.get('synthesis', '').lower()
        reasoning_conclusion = reasoning_result.get('conclusion', '').lower()
        answer_lower = str(answer).lower()
        
        # Check answer có xuất hiện trong reasoning
        answer_in_reasoning = answer_lower in reasoning_conclusion if answer else 0.5
        
        # Check answer có được support bởi web evidence
        answer_in_web = answer_lower in web_synthesis if answer else 0.5
        
        # Length-based confidence (dài = nhiều chi tiết)
        web_length_score = min(len(web_synthesis) / 1000.0, 0.3)
        reasoning_length_score = min(len(reasoning_conclusion) / 500.0, 0.3)
        
        consistency_score = (
            (0.4 if answer_in_reasoning else 0.0) +
            (0.4 if answer_in_web else 0.0) +
            web_length_score +
            reasoning_length_score
        )
        
        return min(consistency_score, 1.0)
    
    def apply_self_consistency_bonus(
        self,
        base_confidence: float,
        reasoning_result: Dict[str, Any]
    ) -> float:
        """
        Áp dụng bonus nếu dùng self-consistency với agreement cao.
        
        Args:
            base_confidence: Confidence cơ bản
            reasoning_result: Kết quả reasoning
            
        Returns:
            Confidence sau khi thêm bonus
        """
        if reasoning_result.get('reasoning_type') != 'self_consistency':
            return base_confidence
        
        all_conclusions = reasoning_result.get('all_conclusions', [])
        num_samples = reasoning_result.get('num_samples', 0)
        
        if num_samples < 2:
            return base_confidence
        
        # Tính agreement ratio
        counter = Counter(all_conclusions)
        most_common_count = counter.most_common(1)[0][1] if counter else 0
        agreement_ratio = most_common_count / num_samples if num_samples > 0 else 0
        
        # Bonus từ 0% - 15% tùy agreement
        bonus = agreement_ratio * 0.15
        
        return min(base_confidence + bonus, 1.0)
    
    def calculate_objective_confidence(
        self,
        web_search_result: Dict[str, Any],
        reasoning_result: Dict[str, Any],
        llm_validation: Dict[str, Any],
        answer: str = ""
    ) -> Dict[str, Any]:
        """
        Tính confidence hybrid: kết hợp objective metrics và LLM assessment.
        
        Args:
            web_search_result: Kết quả web search
            reasoning_result: Kết quả reasoning
            llm_validation: Kết quả validation từ LLM
            answer: Đáp án (optional)
            
        Returns:
            Dictionary với overall_confidence và breakdown
        """
        # 1. Evidence score (objective)
        evidence_score = self.calculate_evidence_score(web_search_result)
        
        # 2. Reasoning score (objective)
        reasoning_score = self.calculate_reasoning_score(reasoning_result)
        
        # 3. Consistency score (objective)
        consistency_score = self.calculate_consistency_score(
            web_search_result, reasoning_result, answer
        )
        
        # 4. LLM validator score (giảm weight từ 100% → 30%)
        llm_score = llm_validation.get('overall_confidence', 0.7)
        
        # 5. Weighted average
        weights = {
            'llm': 0.30,
            'evidence': 0.25,
            'reasoning': 0.25,
            'consistency': 0.20
        }
        
        objective_confidence = (
            llm_score * weights['llm'] +
            evidence_score * weights['evidence'] +
            reasoning_score * weights['reasoning'] +
            consistency_score * weights['consistency']
        )
        
        # 6. Apply self-consistency bonus
        final_confidence = self.apply_self_consistency_bonus(
            objective_confidence, reasoning_result
        )
        
        return {
            'overall_confidence': final_confidence,
            'objective_confidence': objective_confidence,
            'breakdown': {
                'llm_score': llm_score,
                'evidence_score': evidence_score,
                'reasoning_score': reasoning_score,
                'consistency_score': consistency_score,
                'self_consistency_bonus': final_confidence - objective_confidence,
                'weights_used': weights
            }
        }
    
    def validate_with_objective_confidence(
        self,
        question: str,
        web_search_result: Dict[str, Any],
        reasoning_result: Dict[str, Any],
        answer: str = ""
    ) -> Dict[str, Any]:
        """
        Validate với hybrid confidence calculation.
        
        Args:
            question: Câu hỏi
            web_search_result: Kết quả web search
            reasoning_result: Kết quả reasoning
            answer: Đáp án (optional)
            
        Returns:
            Validation result với hybrid confidence
        """
        # Get LLM validation
        llm_validation = self.validate(question, web_search_result, reasoning_result)
        
        # Calculate objective confidence
        confidence_result = self.calculate_objective_confidence(
            web_search_result,
            reasoning_result,
            llm_validation,
            answer
        )
        
        # Merge results
        llm_validation['overall_confidence'] = confidence_result['overall_confidence']
        llm_validation['objective_confidence'] = confidence_result['objective_confidence']
        llm_validation['confidence_breakdown'] = confidence_result['breakdown']
        
        return llm_validation
    
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
        
        # Look for answer pattern (from most specific to general)
        patterns = [
            # JSON format: "answer": "B"
            r'"answer"\s*:\s*"([A-Ea-e])"',
            # Answer: B or ANSWER: B
            r'(?:^|\n)\s*[Aa]nswer\s*:\s*([A-Ea-e])(?:\s|\.|\n|$)',
            # The answer is B / The correct answer is B
            r'(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-Ea-e])(?:\s|\.|\n|$)',
            # Final answer: B
            r'[Ff]inal\s+[Aa]nswer\s*:\s*([A-Ea-e])(?:\s|\.|\n|$)',
            # Option B is correct
            r'[Oo]ption\s+([A-Ea-e])\s+is\s+(?:the\s+)?correct',
            # Conclusion: B
            r'[Cc]onclusion\s*:\s*([A-Ea-e])(?:\s|\.|\n|$)',
            # **B** or *B* (markdown)
            r'\*+([A-Ea-e])\*+',
            # (B) or B.
            r'\(([A-Ea-e])\)|(?:^|\s)([A-Ea-e])\.(?:\s|$)',
            # Standalone letter at start
            r'^([A-Ea-e])(?:\s|$)',
            # Any letter A-E surrounded by word boundaries
            r'\b([A-Ea-e])\b(?:\s*[-:)]|\s+is)',
        ]
        
        # Try patterns on conclusion
        for pattern in patterns:
            match = re.search(pattern, conclusion, re.IGNORECASE | re.MULTILINE)
            if match:
                # Get the first non-None group
                answer = next((g for g in match.groups() if g), None)
                if answer:
                    return answer.upper()
        
        # Try raw output
        raw_output = result.get('raw_output', '')
        for pattern in patterns:
            match = re.search(pattern, raw_output, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = next((g for g in match.groups() if g), None)
                if answer:
                    return answer.upper()
        
        # Last resort: find any single letter A-E
        all_text = conclusion + ' ' + raw_output
        letter_match = re.search(r'\b([A-Ea-e])\b', all_text)
        if letter_match:
            return letter_match.group(1).upper()
        
        return ''
    
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
