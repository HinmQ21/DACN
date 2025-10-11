"""Answer Generator Agent - Tổng hợp và tạo câu trả lời cuối cùng."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from utils.config import Config


# Pydantic models for structured output
class MultipleChoiceAnswer(BaseModel):
    """Structured output for multiple choice questions."""
    answer: str = Field(description="The letter of the correct answer (A, B, C, D, or E)")
    explanation: str = Field(description="Brief explanation of why this answer is chosen")
    source: str = Field(description="Summary of supporting evidence")


class YesNoAnswer(BaseModel):
    """Structured output for yes/no questions."""
    answer: str = Field(description="Either 'yes' or 'no'")
    explanation: str = Field(description="Detailed explanation for the answer")
    confidence: str = Field(description="Confidence level: high, medium, or low")


class AnswerGeneratorAgent:
    """Agent tạo câu trả lời cuối cùng."""
    
    def __init__(self):
        # Use answer_generator-specific configuration
        self.llm = ChatGoogleGenerativeAI(**Config.get_llm_config('answer_generator'))
        
        # Create parsers for structured output
        self.mc_parser = PydanticOutputParser(pydantic_object=MultipleChoiceAnswer)
        self.yesno_parser = PydanticOutputParser(pydantic_object=YesNoAnswer)
        
        # Template for multiple choice questions
        self.mc_prompt = PromptTemplate(
            template="""You are a specialist physician providing accurate answers to medical questions.

Based on:
- Web search results
- Logical reasoning
- Validator assessment

Please synthesize and provide the MOST ACCURATE answer.

Question: {question}

{options_text}

Information from Web Search:
{web_info}

Reasoning:
{reasoning_info}

Validator Assessment:
{validation_info}

{format_instructions}

Please provide the final answer in the specified JSON format.""",
            input_variables=["question", "options_text", "web_info", "reasoning_info", "validation_info"],
            partial_variables={"format_instructions": self.mc_parser.get_format_instructions()}
        )
        
        # Template for yes/no questions
        self.yesno_prompt = PromptTemplate(
            template="""You are a specialist physician providing accurate answers to medical questions.

Question: {question}

Information from Web Search:
{web_info}

Reasoning:
{reasoning_info}

Validator Assessment:
{validation_info}

{format_instructions}

Please provide the final answer in the specified JSON format.""",
            input_variables=["question", "web_info", "reasoning_info", "validation_info"],
            partial_variables={"format_instructions": self.yesno_parser.get_format_instructions()}
        )
    
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
        
        # Choose prompt and parser based on question type
        if question_type == "multiple_choice" and options:
            options_text = "Options:\n" + "\n".join([f"{opt}" for opt in options])
            prompt = self.mc_prompt
            parser = self.mc_parser
        else:
            options_text = ""
            prompt = self.yesno_prompt
            parser = self.yesno_parser
        
        # Generate answer
        answer_input = {
            "question": question,
            "web_info": web_info,
            "reasoning_info": reasoning_info,
            "validation_info": validation_summary
        }
        
        # Add options_text only for multiple choice
        if question_type == "multiple_choice" and options:
            answer_input["options_text"] = options_text
        
        # Format prompt and get LLM response
        formatted_prompt = prompt.format_prompt(**answer_input)
        response = self.llm.invoke(formatted_prompt.to_string())
        
        # Parse structured output
        try:
            parsed_answer = parser.parse(response.content)
            
            return {
                'answer': parsed_answer.answer,
                'explanation': parsed_answer.explanation,
                'full_response': response.content,
                'confidence': validation_result.get('overall_confidence', 0.0),
                'sources_count': web_search_result.get('total_sources', 0),
                'parsed_successfully': True
            }
        except Exception as e:
            # Fallback to manual parsing if structured parsing fails
            print(f"Warning: Structured parsing failed: {e}. Falling back to manual parsing.")
            
            result = response.content
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
                'sources_count': web_search_result.get('total_sources', 0),
                'parsed_successfully': False
            }

