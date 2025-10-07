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
        
        # Template cho câu hỏi multiple choice
        self.mc_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là một bác sĩ chuyên khoa đưa ra câu trả lời chính xác cho câu hỏi y tế.

Dựa trên:
- Kết quả tìm kiếm web
- Suy luận logic
- Đánh giá từ validator

Hãy tổng hợp và đưa ra câu trả lời CHÍNH XÁC NHẤT.

Với câu hỏi multiple choice, trả lời theo format:
ĐÁP ÁN: [A/B/C/D/E]
GIẢI THÍCH: [giải thích ngắn gọn tại sao chọn đáp án này]
NGUỒN: [tóm tắt bằng chứng hỗ trợ]"""),
            ("human", """Câu hỏi: {question}

{options_text}

Thông tin từ Web Search:
{web_info}

Suy luận:
{reasoning_info}

Đánh giá Validator:
{validation_info}

Hãy đưa ra câu trả lời cuối cùng.""")
        ])
        
        # Template cho câu hỏi yes/no
        self.yesno_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là một bác sĩ chuyên khoa đưa ra câu trả lời chính xác cho câu hỏi y tế.

Với câu hỏi yes/no, trả lời theo format:
ĐÁP ÁN: yes/no
GIẢI THÍCH: [giải thích chi tiết]
ĐỘ TIN CẬY: [high/medium/low]"""),
            ("human", """Câu hỏi: {question}

Thông tin từ Web Search:
{web_info}

Suy luận:
{reasoning_info}

Đánh giá Validator:
{validation_info}

Hãy đưa ra câu trả lời cuối cùng.""")
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
            options_text = "Các lựa chọn:\n" + "\n".join([f"{opt}" for opt in options])
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
            if line_upper.startswith('ĐÁP ÁN:') or line_upper.startswith('DAP AN:'):
                answer = line.split(':', 1)[1].strip() if ':' in line else ''
            elif line_upper.startswith('GIẢI THÍCH:') or line_upper.startswith('GIAI THICH:'):
                # Get all following lines until next section
                explanation = line.split(':', 1)[1].strip() if ':' in line else ''
                for j in range(i+1, len(lines)):
                    if ':' in lines[j] and any(keyword in lines[j].upper() for keyword in ['NGUỒN', 'ĐỘ TIN CẬY', 'NGUON', 'DO TIN CAY']):
                        break
                    explanation += ' ' + lines[j].strip()
        
        return {
            'answer': answer,
            'explanation': explanation,
            'full_response': result,
            'confidence': validation_result.get('overall_confidence', 0.0),
            'sources_count': web_search_result.get('total_sources', 0)
        }

