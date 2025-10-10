"""Coordinator Agent - Phân tích câu hỏi và điều phối workflow."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any
from utils.config import Config


class CoordinatorAgent:
    """Agent điều phối - phân tích câu hỏi và quyết định chiến lược."""
    
    def __init__(self):
        # Use coordinator-specific configuration
        self.llm = ChatGoogleGenerativeAI(**Config.get_llm_config('coordinator'))
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a coordinator agent in a multi-agent medical system.
Your task is to analyze medical questions and determine the search strategy.

Analyze the following factors:
1. Question type (diagnosis, treatment, physiology, pharmacology, etc.)
2. Question complexity
3. Key terms for searching
4. Whether web search is needed or reasoning alone is sufficient
5. Priority level between web search and reasoning

Return a concise analysis in JSON format:
{{
    "question_type": "question type",
    "complexity": "low/medium/high",
    "key_terms": ["keyword 1", "keyword 2"],
    "needs_web_search": true/false,
    "needs_reasoning": true/false,
    "search_priority": "high/medium/low",
    "reasoning_priority": "high/medium/low"
}}"""),
            ("human", "{question}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def analyze(self, question: str) -> Dict[str, Any]:
        """
        Phân tích câu hỏi và quyết định chiến lược.
        
        Args:
            question: Câu hỏi cần phân tích
            
        Returns:
            Dictionary chứa phân tích
        """
        result = self.chain.invoke({"question": question})
        
        # Parse JSON response
        import json
        try:
            # Extract JSON from markdown code block if present
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(result)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            analysis = {
                "question_type": "general",
                "complexity": "medium",
                "key_terms": [],
                "needs_web_search": True,
                "needs_reasoning": True,
                "search_priority": "medium",
                "reasoning_priority": "medium"
            }
        
        return analysis

