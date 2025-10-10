"""Answer Generator Agent - Tổng hợp và tạo câu trả lời cuối cùng."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, List, Optional
from utils.config import Config


class AnswerGeneratorAgent:
    """Agent tạo câu trả lời cuối cùng."""
    
    def __init__(self):
        # Use answer_generator-specific configuration
        self.llm = ChatGoogleGenerativeAI(**Config.get_llm_config('answer_generator'))
        
        # Template for multiple choice questions
        self.mc_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialist physician providing accurate answers to medical questions.

Based on:
- Web search results
- Logical reasoning
- Validator assessment

Please synthesize and provide the MOST ACCURATE answer.

For multiple choice questions, respond in this format:
ANSWER: [A/B/C/D/E]
EXPLANATION: [brief explanation of why this answer is chosen]
SOURCE: [summary of supporting evidence]"""),
            ("human", """Question: {question}

{options_text}

Information from Web Search:
{web_info}

Reasoning:
{reasoning_info}

Validator Assessment:
{validation_info}

Please provide the final answer.""")
        ])
        
        # Template for yes/no questions
        self.yesno_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialist physician providing accurate answers to medical questions.

For yes/no questions, respond in this format:
ANSWER: yes/no
EXPLANATION: [detailed explanation]
CONFIDENCE: [high/medium/low]"""),
            ("human", """Question: {question}

Information from Web Search:
{web_info}

Reasoning:
{reasoning_info}

Validator Assessment:
{validation_info}

Please provide the final answer.""")
        ])
    
    def generate(
        self,
        question: str,
        web_search_result: Dict[str, Any],
        reasoning_result: Dict[str, Any],
        validation_result: Dict[str, Any],
        options: Optional[List[str]] = None,
        question_type: str = "multiple_choice"
    ) -> Dict[str, Any]:
        """
        Tạo câu trả lời cuối cùng.
        
        Args:
            question: Câu hỏi gốc
            web_search_result: Kết quả từ web search
            reasoning_result: Kết quả từ reasoning
            validation_result: Kết quả từ validator
            options: Các lựa chọn (nếu là multiple choice)
            question_type: Loại câu hỏi ("multiple_choice" hoặc "yes_no")
            
        Returns:
            Dictionary chứa câu trả lời và thông tin bổ sung
        """
        # Prepare information
        web_info = web_search_result.get('synthesis', 'No information from web search')
        reasoning_info = reasoning_result.get('raw_output', 'No reasoning available')
        
        validation_summary = (
            f"Consistency: {validation_result.get('is_consistent', 'Unknown')}\n"
            f"Evidence Quality: {validation_result.get('evidence_quality', 0)}\n"
            f"Reasoning Quality: {validation_result.get('reasoning_quality', 0)}\n"
            f"Confidence: {validation_result.get('overall_confidence', 0)}"
        )
        
        # Choose prompt based on question type
        if question_type == "multiple_choice" and options:
            options_text = "Options:\n" + "\n".join([f"{opt}" for opt in options])
            prompt = self.mc_prompt
        else:
            options_text = ""
            prompt = self.yesno_prompt
        
        # Generate answer
        answer_input = {
            "question": question,
            "options_text": options_text,
            "web_info": web_info,
            "reasoning_info": reasoning_info,
            "validation_info": validation_summary
        }
        
        result = (prompt | self.llm | StrOutputParser()).invoke(answer_input)
        
        # Extract answer
        answer = ""
        explanation = ""
        
        lines = result.split('\n')
        for i, line in enumerate(lines):
            line_upper = line.strip().upper()
            if line_upper.startswith('ANSWER:'):
                answer = line.split(':', 1)[1].strip() if ':' in line else ''
            elif line_upper.startswith('EXPLANATION:'):
                # Get all following lines until next section
                explanation = line.split(':', 1)[1].strip() if ':' in line else ''
                for j in range(i+1, len(lines)):
                    if ':' in lines[j] and any(keyword in lines[j].upper() for keyword in ['SOURCE', 'CONFIDENCE']):
                        break
                    explanation += ' ' + lines[j].strip()
        
        return {
            'answer': answer,
            'explanation': explanation,
            'full_response': result,
            'confidence': validation_result.get('overall_confidence', 0.0),
            'sources_count': web_search_result.get('total_sources', 0)
        }

