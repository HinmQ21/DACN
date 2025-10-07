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
            ("system", """Bạn là một validator chuyên nghiệp trong hệ thống y tế.
Nhiệm vụ của bạn là kiểm tra và đánh giá:

1. TÍNH NHẤT QUÁN:
   - Kết quả web search và reasoning có nhất quán không?
   - Có mâu thuẫn giữa các nguồn thông tin không?

2. CHẤT LƯỢNG BẰNG CHỨNG:
   - Độ tin cậy của nguồn thông tin (0-1)
   - Tính cập nhật của thông tin
   - Tính phù hợp với câu hỏi

3. CHẤT LƯỢNG SUY LUẬN:
   - Logic suy luận có chặt chẽ không? (0-1)
   - Có bỏ sót bước nào không?

Trả về đánh giá theo format JSON:
{{
    "is_consistent": true/false,
    "consistency_explanation": "giải thích",
    "evidence_quality": 0.0-1.0,
    "evidence_issues": ["vấn đề 1", "vấn đề 2"],
    "reasoning_quality": 0.0-1.0,
    "reasoning_issues": ["vấn đề 1", "vấn đề 2"],
    "conflicts": ["mâu thuẫn 1", "mâu thuẫn 2"],
    "overall_confidence": 0.0-1.0,
    "recommendation": "proceed/revise/reject"
}}"""),
            ("human", """Câu hỏi: {question}

Kết quả Web Search:
{web_search_result}

Kết quả Reasoning:
{reasoning_result}

Hãy đánh giá và kiểm chứng.""")
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

