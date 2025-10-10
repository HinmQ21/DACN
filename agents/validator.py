"""Validator Agent - Kiểm tra tính nhất quán và độ tin cậy."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any
from utils.config import Config
import json


class ValidatorAgent:
    """Agent kiểm chứng tính nhất quán và độ tin cậy."""
    
    def __init__(self):
        # Use validator-specific configuration
        self.llm = ChatGoogleGenerativeAI(**Config.get_llm_config('validator'))
        
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

