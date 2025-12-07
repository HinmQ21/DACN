"""Reflexion Agent - Self-correction mechanism for improving answer quality."""

import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, List, Optional
from utils.config import Config


class ReflexionAgent:
    """
    Agent tự phê bình và sửa câu trả lời (Reflexion pattern).
    
    Cơ chế hoạt động:
    1. Critique Phase: Đánh giá câu trả lời hiện tại
    2. Reflection Phase: Xác định điểm yếu/không nhất quán
    3. Correction Phase: Sinh câu trả lời cải thiện (nếu cần)
    4. Verification Phase: Xác nhận cải thiện tốt hơn
    """
    
    def __init__(self, enable_reflexion: bool = None):
        """
        Initialize Reflexion Agent.
        
        Args:
            enable_reflexion: Whether to enable reflexion (uses config if None)
        """
        # Use reflexion-specific configuration
        self.llm = ChatGoogleGenerativeAI(**Config.get_llm_config('reflexion'))
        
        # Reflexion configuration
        reflexion_config = Config.get_reflexion_config()
        self.enable_reflexion = enable_reflexion if enable_reflexion is not None else reflexion_config['enable_reflexion']
        self.max_iterations = reflexion_config['max_iterations']
        self.confidence_threshold = reflexion_config['confidence_threshold']
        
        # Critique prompt - đánh giá câu trả lời
        self.critique_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a critical medical expert reviewing an AI-generated answer.
Your task is to thoroughly critique the answer and identify any issues.

Evaluate the following aspects:
1. LOGICAL CONSISTENCY: Is the reasoning sound and coherent?
2. MEDICAL ACCURACY: Are the medical facts correct?
3. EVIDENCE SUPPORT: Is the answer well-supported by the provided evidence?
4. COMPLETENESS: Does the answer address all aspects of the question?
5. ANSWER ALIGNMENT: Does the final answer match the reasoning?

Be rigorous and identify any:
- Logical fallacies or reasoning gaps
- Potential medical inaccuracies
- Missing considerations
- Contradictions between reasoning and answer
- Weak or unsupported claims

Return your critique in JSON format:
{{
    "is_satisfactory": true/false,
    "confidence_assessment": 0.0-1.0,
    "issues_found": [
        {{
            "type": "logic|accuracy|evidence|completeness|alignment",
            "severity": "minor|moderate|critical",
            "description": "detailed description of the issue"
        }}
    ],
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1", "weakness 2"],
    "correction_needed": true/false,
    "correction_suggestions": ["suggestion 1", "suggestion 2"],
    "overall_assessment": "brief overall assessment"
}}"""),
            ("human", """Question: {question}

Options:
{options_text}

Generated Answer: {answer}

Reasoning Used:
{reasoning}

Validation Result:
{validation}

Web Search Evidence:
{web_evidence}

Please provide your thorough critique.""")
        ])
        
        # Correction prompt - sửa câu trả lời
        self.correction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical expert tasked with correcting and improving an answer.

Based on the critique provided, you must:
1. Address all identified issues
2. Strengthen weak points
3. Provide more rigorous reasoning
4. Ensure answer aligns with evidence

Important guidelines:
- Maintain medical accuracy
- Use step-by-step reasoning
- Cite evidence when available
- Be explicit about your reasoning
- Choose the most defensible answer

Return your corrected answer in JSON format:
{{
    "corrected_answer": "A/B/C/D/E",
    "improved_reasoning": "detailed step-by-step reasoning",
    "issues_addressed": ["issue 1 addressed", "issue 2 addressed"],
    "confidence": 0.0-1.0,
    "explanation": "brief explanation of the answer"
}}"""),
            ("human", """Question: {question}

Options:
{options_text}

Original Answer: {original_answer}

Original Reasoning:
{original_reasoning}

Critique:
{critique}

Web Search Evidence:
{web_evidence}

Please provide a corrected and improved answer addressing the critique.""")
        ])
        
        # Verification prompt - xác nhận cải thiện
        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical expert verifying whether a corrected answer is indeed better.

Compare the original and corrected answers objectively.
Consider:
1. Quality of reasoning
2. Medical accuracy
3. Evidence alignment
4. Logical consistency
5. Confidence level

Return your verification in JSON format:
{{
    "is_improvement": true/false,
    "original_score": 0.0-1.0,
    "corrected_score": 0.0-1.0,
    "comparison_notes": "detailed comparison",
    "final_recommendation": "use_original|use_corrected",
    "reasoning": "why this recommendation"
}}"""),
            ("human", """Question: {question}

Original Answer: {original_answer}
Original Reasoning: {original_reasoning}
Original Confidence: {original_confidence}

Corrected Answer: {corrected_answer}
Corrected Reasoning: {corrected_reasoning}
Corrected Confidence: {corrected_confidence}

Please verify which answer is better.""")
        ])
    
    def should_reflect(
        self,
        answer_result: Dict[str, Any],
        validation_result: Dict[str, Any]
    ) -> bool:
        """
        Determine if reflexion should be triggered.
        
        When reflexion is enabled, it ALWAYS runs to ensure answer quality.
        
        Args:
            answer_result: Result from AnswerGenerator
            validation_result: Result from Validator
            
        Returns:
            True if reflexion should be performed
        """
        if not self.enable_reflexion:
            return False
        
        # Always perform reflexion when enabled
        print("[Reflexion] Triggered: reflexion enabled - always perform self-correction")
        return True
    
    def critique(
        self,
        question: str,
        options: Dict[str, str],
        answer_result: Dict[str, Any],
        reasoning_result: Dict[str, Any],
        validation_result: Dict[str, Any],
        web_search_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate critique for the current answer.
        
        Args:
            question: The original question
            options: Answer options
            answer_result: Result from AnswerGenerator
            reasoning_result: Result from ReasoningAgent
            validation_result: Result from Validator
            web_search_result: Optional web search results
            
        Returns:
            Critique result dictionary
        """
        # Format options
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()]) if options else "No options provided"
        
        # Format inputs
        critique_input = {
            "question": question,
            "options_text": options_text,
            "answer": answer_result.get('answer', ''),
            "reasoning": reasoning_result.get('raw_output', 'No reasoning available'),
            "validation": json.dumps(validation_result, indent=2) if validation_result else "No validation",
            "web_evidence": web_search_result.get('synthesis', 'No web evidence') if web_search_result else "No web evidence"
        }
        
        # Generate critique
        chain = self.critique_prompt | self.llm | StrOutputParser()
        result = chain.invoke(critique_input)
        
        # Parse critique result
        return self._parse_critique_result(result)
    
    def correct(
        self,
        question: str,
        options: Dict[str, str],
        original_answer: Dict[str, Any],
        original_reasoning: Dict[str, Any],
        critique: Dict[str, Any],
        web_search_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate corrected answer based on critique.
        
        Args:
            question: The original question
            options: Answer options
            original_answer: Original answer result
            original_reasoning: Original reasoning result
            critique: Critique result
            web_search_result: Optional web search results
            
        Returns:
            Corrected answer dictionary
        """
        # Format options
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()]) if options else "No options provided"
        
        # Format critique for prompt
        critique_text = self._format_critique_for_prompt(critique)
        
        # Build correction input
        correction_input = {
            "question": question,
            "options_text": options_text,
            "original_answer": original_answer.get('answer', ''),
            "original_reasoning": original_reasoning.get('raw_output', ''),
            "critique": critique_text,
            "web_evidence": web_search_result.get('synthesis', 'No web evidence') if web_search_result else "No web evidence"
        }
        
        # Generate correction
        chain = self.correction_prompt | self.llm | StrOutputParser()
        result = chain.invoke(correction_input)
        
        # Parse correction result
        return self._parse_correction_result(result)
    
    def verify_improvement(
        self,
        question: str,
        original_answer: Dict[str, Any],
        original_reasoning: Dict[str, Any],
        corrected_answer: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify if the corrected answer is actually better.
        
        Args:
            question: The original question
            original_answer: Original answer result
            original_reasoning: Original reasoning result
            corrected_answer: Corrected answer result
            
        Returns:
            Verification result dictionary
        """
        verification_input = {
            "question": question,
            "original_answer": original_answer.get('answer', ''),
            "original_reasoning": original_reasoning.get('raw_output', ''),
            "original_confidence": original_answer.get('confidence', 0.0),
            "corrected_answer": corrected_answer.get('corrected_answer', ''),
            "corrected_reasoning": corrected_answer.get('improved_reasoning', ''),
            "corrected_confidence": corrected_answer.get('confidence', 0.0)
        }
        
        # Generate verification
        chain = self.verification_prompt | self.llm | StrOutputParser()
        result = chain.invoke(verification_input)
        
        # Parse verification result
        return self._parse_verification_result(result)
    
    def reflect_and_correct(
        self,
        question: str,
        options: Dict[str, str],
        answer_result: Dict[str, Any],
        reasoning_result: Dict[str, Any],
        validation_result: Dict[str, Any],
        web_search_result: Dict[str, Any] = None,
        iteration: int = 0
    ) -> Dict[str, Any]:
        """
        Complete reflexion cycle: critique → correct → verify.
        
        Args:
            question: The original question
            options: Answer options
            answer_result: Current answer result
            reasoning_result: Reasoning result
            validation_result: Validation result
            web_search_result: Optional web search results
            iteration: Current iteration number
            
        Returns:
            Reflexion result with potentially corrected answer
        """
        print(f"[Reflexion] Starting iteration {iteration + 1}/{self.max_iterations}")
        
        # Check if we should reflect
        if not self.should_reflect(answer_result, validation_result) and iteration == 0:
            print("[Reflexion] Skipped: reflexion is disabled")
            return {
                'performed': False,
                'reason': 'Reflexion is disabled',
                'final_answer': answer_result,
                'iterations': 0
            }
        
        # Check max iterations
        if iteration >= self.max_iterations:
            print(f"[Reflexion] Max iterations ({self.max_iterations}) reached")
            return {
                'performed': True,
                'reason': 'Max iterations reached',
                'final_answer': answer_result,
                'iterations': iteration
            }
        
        # Step 1: Critique
        print("[Reflexion] Phase 1: Critique")
        critique = self.critique(
            question=question,
            options=options,
            answer_result=answer_result,
            reasoning_result=reasoning_result,
            validation_result=validation_result,
            web_search_result=web_search_result
        )
        
        # Check if correction is needed
        if critique.get('is_satisfactory', True) and not critique.get('correction_needed', False):
            print("[Reflexion] Critique: Answer is satisfactory, no correction needed")
            return {
                'performed': True,
                'reason': 'Critique found answer satisfactory',
                'critique': critique,
                'final_answer': answer_result,
                'iterations': iteration + 1
            }
        
        # Step 2: Correct
        print("[Reflexion] Phase 2: Correction")
        corrected = self.correct(
            question=question,
            options=options,
            original_answer=answer_result,
            original_reasoning=reasoning_result,
            critique=critique,
            web_search_result=web_search_result
        )
        
        # Step 3: Verify improvement
        print("[Reflexion] Phase 3: Verification")
        verification = self.verify_improvement(
            question=question,
            original_answer=answer_result,
            original_reasoning=reasoning_result,
            corrected_answer=corrected
        )
        
        # Decide which answer to use
        if verification.get('is_improvement', False) and verification.get('final_recommendation') == 'use_corrected':
            print(f"[Reflexion] Improvement verified: {answer_result.get('answer')} → {corrected.get('corrected_answer')}")
            
            # Build improved answer result
            improved_answer = {
                'answer': corrected.get('corrected_answer', answer_result.get('answer')),
                'explanation': corrected.get('explanation', answer_result.get('explanation', '')),
                'confidence': corrected.get('confidence', answer_result.get('confidence', 0.0)),
                'sources_count': answer_result.get('sources_count', 0),
                'parsed_successfully': True,
                'reflexion_applied': True
            }
            
            # Check if we need another iteration (if still below threshold)
            if corrected.get('confidence', 1.0) < self.confidence_threshold and iteration + 1 < self.max_iterations:
                print("[Reflexion] Confidence still below threshold, continuing...")
                return self.reflect_and_correct(
                    question=question,
                    options=options,
                    answer_result=improved_answer,
                    reasoning_result={'raw_output': corrected.get('improved_reasoning', '')},
                    validation_result=validation_result,
                    web_search_result=web_search_result,
                    iteration=iteration + 1
                )
            
            return {
                'performed': True,
                'reason': 'Correction improved answer',
                'critique': critique,
                'correction': corrected,
                'verification': verification,
                'original_answer': answer_result.get('answer'),
                'original_confidence': answer_result.get('confidence', 0.0),
                'final_answer': improved_answer,
                'iterations': iteration + 1
            }
        else:
            print("[Reflexion] Correction did not improve answer, keeping original")
            return {
                'performed': True,
                'reason': 'Correction did not improve answer',
                'critique': critique,
                'correction': corrected,
                'verification': verification,
                'final_answer': answer_result,
                'iterations': iteration + 1
            }
    
    def _parse_critique_result(self, result: str) -> Dict[str, Any]:
        """Parse critique LLM output."""
        try:
            # Extract JSON from markdown code block if present
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            return json.loads(result)
        except json.JSONDecodeError:
            print("[Reflexion] Failed to parse critique JSON, using defaults")
            return {
                'is_satisfactory': True,
                'confidence_assessment': 0.7,
                'issues_found': [],
                'strengths': [],
                'weaknesses': [],
                'correction_needed': False,
                'correction_suggestions': [],
                'overall_assessment': 'Unable to parse critique'
            }
    
    def _parse_correction_result(self, result: str) -> Dict[str, Any]:
        """Parse correction LLM output."""
        try:
            # Extract JSON from markdown code block if present
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(result)
            
            # Normalize answer to uppercase letter
            if 'corrected_answer' in parsed:
                answer = parsed['corrected_answer']
                # Extract just the letter if there's more text
                match = re.match(r'^([A-Ea-e])', str(answer).strip())
                if match:
                    parsed['corrected_answer'] = match.group(1).upper()
            
            return parsed
        except json.JSONDecodeError:
            print("[Reflexion] Failed to parse correction JSON")
            return {
                'corrected_answer': '',
                'improved_reasoning': '',
                'issues_addressed': [],
                'confidence': 0.5,
                'explanation': 'Unable to parse correction'
            }
    
    def _parse_verification_result(self, result: str) -> Dict[str, Any]:
        """Parse verification LLM output."""
        try:
            # Extract JSON from markdown code block if present
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            return json.loads(result)
        except json.JSONDecodeError:
            print("[Reflexion] Failed to parse verification JSON")
            return {
                'is_improvement': False,
                'original_score': 0.5,
                'corrected_score': 0.5,
                'comparison_notes': 'Unable to parse verification',
                'final_recommendation': 'use_original',
                'reasoning': 'Parse error - defaulting to original'
            }
    
    def _format_critique_for_prompt(self, critique: Dict[str, Any]) -> str:
        """Format critique dict as readable text for prompt."""
        parts = []
        
        # Overall assessment
        if critique.get('overall_assessment'):
            parts.append(f"Overall Assessment: {critique['overall_assessment']}")
        
        # Issues found
        issues = critique.get('issues_found', [])
        if issues:
            parts.append("\nIssues Found:")
            for i, issue in enumerate(issues, 1):
                severity = issue.get('severity', 'unknown')
                issue_type = issue.get('type', 'unknown')
                desc = issue.get('description', '')
                parts.append(f"  {i}. [{severity.upper()}] ({issue_type}): {desc}")
        
        # Weaknesses
        weaknesses = critique.get('weaknesses', [])
        if weaknesses:
            parts.append("\nWeaknesses:")
            for w in weaknesses:
                parts.append(f"  - {w}")
        
        # Correction suggestions
        suggestions = critique.get('correction_suggestions', [])
        if suggestions:
            parts.append("\nCorrection Suggestions:")
            for s in suggestions:
                parts.append(f"  - {s}")
        
        return "\n".join(parts) if parts else "No specific critique provided"

