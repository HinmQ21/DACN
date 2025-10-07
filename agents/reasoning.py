"""Reasoning Agent - Suy luận logic dựa trên kiến thức y tế."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any
from utils.config import Config


class ReasoningAgent:
    """Agent suy luận y tế dựa trên kiến thức."""
    
    def __init__(self):
        # Use reasoning-specific configuration
        self.llm = ChatGoogleGenerativeAI(**Config.get_llm_config('reasoning'))
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là một bác sĩ chuyên khoa với kiến thức y tế sâu rộng.
Nhiệm vụ của bạn là suy luận logic để trả lời câu hỏi y tế dựa trên:
1. Kiến thức y khoa cơ bản
2. Sinh lý học và sinh hóa
3. Dược lý học
4. Chẩn đoán lâm sàng
5. Cơ chế bệnh sinh

Hãy sử dụng phương pháp suy luận từng bước (step-by-step reasoning):
1. Phân tích câu hỏi
2. Xác định kiến thức nền tảng liên quan
3. Áp dụng logic y khoa
4. Đưa ra kết luận

Trả lời theo format:
PHÂN TÍCH: [phân tích câu hỏi]
KIẾN THỨC: [kiến thức y khoa liên quan]
SUY LUẬN: [quá trình suy luận từng bước]
KẾT LUẬN: [kết luận cuối cùng]"""),
            ("human", "{question}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def reason(self, question: str, context: str = None) -> Dict[str, Any]:
        """
        Thực hiện suy luận logic.
        
        Args:
            question: Câu hỏi cần suy luận
            context: Ngữ cảnh bổ sung (nếu có)
            
        Returns:
            Dictionary chứa kết quả suy luận
        """
        full_question = question
        if context:
            full_question = f"Câu hỏi: {question}\n\nNgữ cảnh bổ sung: {context}"
        
        reasoning_result = self.chain.invoke({"question": full_question})
        
        # Parse structured output
        sections = {
            'analysis': '',
            'knowledge': '',
            'reasoning_steps': '',
            'conclusion': ''
        }
        
        lines = reasoning_result.split('\n')
        current_section = None
        
        for line in lines:
            line_upper = line.strip().upper()
            if line_upper.startswith('PHÂN TÍCH:'):
                current_section = 'analysis'
                sections['analysis'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif line_upper.startswith('KIẾN THỨC:'):
                current_section = 'knowledge'
                sections['knowledge'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif line_upper.startswith('SUY LUẬN:'):
                current_section = 'reasoning_steps'
                sections['reasoning_steps'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif line_upper.startswith('KẾT LUẬN:'):
                current_section = 'conclusion'
                sections['conclusion'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif current_section and line.strip():
                sections[current_section] += ' ' + line.strip()
        
        return {
            'raw_output': reasoning_result,
            'structured': sections,
            'conclusion': sections['conclusion']
        }

