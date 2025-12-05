"""Answer Generator Agent - Tổng hợp và tạo câu trả lời cuối cùng."""

import re
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
    
    def _extract_answer_with_regex(self, text: str, question_type: str = "multiple_choice") -> str:
        """
        Extract answer using multiple regex patterns as backup.
        
        Args:
            text: The response text to parse
            question_type: Type of question
            
        Returns:
            Extracted answer or empty string
        """
        if question_type == "multiple_choice":
            # Pattern list from most specific to most general
            patterns = [
                # JSON format: "answer": "B"
                r'"answer"\s*:\s*"([A-Ea-e])"',
                # Answer: B or Answer: B.
                r'(?:^|\n)\s*[Aa]nswer\s*:\s*([A-Ea-e])(?:\s|\.|\n|$)',
                # The answer is B / The correct answer is B
                r'(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-Ea-e])(?:\s|\.|\n|$)',
                # Final answer: B
                r'[Ff]inal\s+[Aa]nswer\s*:\s*([A-Ea-e])(?:\s|\.|\n|$)',
                # Conclusion: B or CONCLUSION: B
                r'[Cc]onclusion\s*:\s*([A-Ea-e])(?:\s|\.|\n|$)',
                # Option B is correct
                r'[Oo]ption\s+([A-Ea-e])\s+is\s+(?:the\s+)?correct',
                # (B) at the end or B. at the end
                r'(?:^|\n)\s*\(?([A-Ea-e])\)?\.?\s*$',
                # Standalone letter at start of line: B or B.
                r'(?:^|\n)\s*([A-Ea-e])(?:\s*[\.\)]|\s*$)',
                # **B** or *B* (markdown bold/italic)
                r'\*+([A-Ea-e])\*+',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(1).upper()
        
        elif question_type == "yes_no":
            # Yes/No patterns
            patterns = [
                r'"answer"\s*:\s*"(yes|no)"',
                r'(?:^|\n)\s*[Aa]nswer\s*:\s*(yes|no)',
                r'(?:the\s+)?answer\s+is\s+(yes|no)',
                r'(?:^|\n)\s*(yes|no)\s*[,\.]',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(1).lower()
        
        return ""
    
    def _parse_response(
        self, 
        response_content: str, 
        parser, 
        question_type: str,
        validation_result: Dict[str, Any],
        web_search_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse response with multiple fallback strategies.
        
        Returns:
            Parsed result dictionary
        """
        answer = ""
        explanation = ""
        parsed_successfully = False
        
        # Strategy 1: Pydantic parser
        try:
            parsed_answer = parser.parse(response_content)
            answer = parsed_answer.answer
            explanation = parsed_answer.explanation
            parsed_successfully = True
        except Exception as e:
            print(f"[AnswerGenerator] Pydantic parsing failed: {e}")
        
        # Strategy 2: Manual line-by-line parsing
        if not answer:
            lines = response_content.split('\n')
            for i, line in enumerate(lines):
                line_upper = line.strip().upper()
                if line_upper.startswith('ANSWER:'):
                    answer = line.split(':', 1)[1].strip() if ':' in line else ''
                    # Extract just the letter if there's more text
                    letter_match = re.match(r'^([A-Ea-e])', answer)
                    if letter_match:
                        answer = letter_match.group(1).upper()
                elif line_upper.startswith('EXPLANATION:'):
                    explanation = line.split(':', 1)[1].strip() if ':' in line else ''
                    for j in range(i+1, len(lines)):
                        if ':' in lines[j] and any(kw in lines[j].upper() for kw in ['SOURCE', 'CONFIDENCE', 'ANSWER']):
                            break
                        explanation += ' ' + lines[j].strip()
        
        # Strategy 3: Regex backup parsing
        if not answer:
            print("[AnswerGenerator] Trying regex backup parsing...")
            answer = self._extract_answer_with_regex(response_content, question_type)
            if answer:
                print(f"[AnswerGenerator] Regex extracted answer: {answer}")
        
        return {
            'answer': answer,
            'explanation': explanation.strip() if explanation else '',
            'full_response': response_content,
            'confidence': validation_result.get('overall_confidence', 0.0),
            'sources_count': web_search_result.get('total_sources', 0),
            'parsed_successfully': parsed_successfully
        }
    
    def generate(
        self,
        question: str,
        web_search_result: Dict[str, Any],
        reasoning_result: Dict[str, Any],
        validation_result: Dict[str, Any],
        options: Optional[List[str]] = None,
        question_type: str = "multiple_choice",
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Tạo câu trả lời cuối cùng với retry logic.
        
        Args:
            question: Câu hỏi gốc
            web_search_result: Kết quả từ web search
            reasoning_result: Kết quả từ reasoning
            validation_result: Kết quả từ validator
            options: Các lựa chọn (nếu là multiple choice)
            question_type: Loại câu hỏi ("multiple_choice" hoặc "yes_no")
            max_retries: Số lần retry tối đa khi không parse được answer
            
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
        
        # Generate answer input
        answer_input = {
            "question": question,
            "web_info": web_info,
            "reasoning_info": reasoning_info,
            "validation_info": validation_summary
        }
        
        # Add options_text only for multiple choice
        if question_type == "multiple_choice" and options:
            answer_input["options_text"] = options_text
        
        # Retry loop
        last_result = None
        all_responses = []
        
        for attempt in range(max_retries + 1):
            try:
                # Format prompt and get LLM response
                formatted_prompt = prompt.format_prompt(**answer_input)
                response = self.llm.invoke(formatted_prompt.to_string())
                all_responses.append(response.content)
                
                # Parse the response
                result = self._parse_response(
                    response.content,
                    parser,
                    question_type,
                    validation_result,
                    web_search_result
                )
                last_result = result
                
                # If we got an answer, return immediately
                if result['answer']:
                    if attempt > 0:
                        print(f"[AnswerGenerator] Successfully parsed answer on retry {attempt}")
                    return result
                
                # No answer parsed, will retry
                if attempt < max_retries:
                    print(f"[AnswerGenerator] No answer parsed, retrying ({attempt + 1}/{max_retries})...")
                
            except Exception as e:
                print(f"[AnswerGenerator] Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries:
                    # Last attempt failed, return error result
                    return {
                        'answer': '',
                        'explanation': f'Error generating answer: {str(e)}',
                        'full_response': '\n---\n'.join(all_responses) if all_responses else '',
                        'confidence': 0.0,
                        'sources_count': 0,
                        'parsed_successfully': False
                    }
        
        # All retries exhausted, return last result (even if answer is empty)
        if last_result:
            print(f"[AnswerGenerator] All {max_retries + 1} attempts failed to parse answer")
            return last_result
        
        # Fallback
        return {
            'answer': '',
            'explanation': 'Failed to generate answer after all retries',
            'full_response': '\n---\n'.join(all_responses) if all_responses else '',
            'confidence': 0.0,
            'sources_count': 0,
            'parsed_successfully': False
        }

