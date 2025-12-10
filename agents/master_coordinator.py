"""Master Coordinator Agent - Routes to subgraphs or answers simple questions."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, Optional
from utils.config import Config
import json


class MasterCoordinatorAgent:
    """
    Master Coordinator Agent - Điều phối cấp cao nhất.
    
    Nhiệm vụ:
    1. Phân tích câu hỏi để xác định loại (text/image/simple)
    2. Trả lời trực tiếp các câu hỏi đơn giản
    3. Route đến medical_qa subgraph cho câu hỏi y tế phức tạp
    4. Route đến image_qa subgraph cho câu hỏi liên quan ảnh
    """
    
    def __init__(self):
        """Initialize Master Coordinator Agent."""
        self.llm = ChatGoogleGenerativeAI(**Config.get_llm_config('coordinator'))
        
        # Routing prompt
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a master coordinator for a medical AI system.
Your task is to analyze the input and determine how to handle it.

Analyze these factors:
1. Input type (text question, image + question, image only)
2. Complexity level (simple, moderate, complex)
3. Whether the question can be answered directly with basic medical knowledge
4. Whether specialized medical research/reasoning is needed
5. Whether image analysis is required

Simple questions are those that:
- Ask for basic definitions or explanations
- Request general health information
- Can be answered with common medical knowledge
- Don't require complex reasoning or research

Complex questions require:
- Detailed diagnostic reasoning
- Analysis of specific clinical cases
- Integration of multiple medical concepts
- Research from medical literature
- Image interpretation

Return a JSON decision:
{{
    "input_type": "text" | "image" | "text_with_image",
    "complexity": "simple" | "moderate" | "complex",
    "can_answer_directly": true/false,
    "requires_image_analysis": true/false,
    "requires_medical_research": true/false,
    "route_to": "direct_answer" | "medical_qa" | "image_qa",
    "reasoning": "brief explanation of decision"
}}"""),
            ("human", """Input Analysis:
{conversation_context}Question: {question}
Has Image: {has_image}
Options Provided: {has_options}

Please provide routing decision.""")
        ])
        
        # Simple answer prompt
        self.simple_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful medical AI assistant.
Provide clear, accurate answers to simple medical questions.
Keep answers concise but informative.
Always include a disclaimer that your answer is for informational purposes and not medical advice.

For multiple choice questions, format your answer as:
Answer: [Letter] - [Option Text]
Explanation: [Your explanation]

For open-ended questions, provide:
Answer: [Direct answer]
Explanation: [Detailed explanation]"""),
            ("human", """{question_text}""")
        ])
        
        self.routing_chain = self.routing_prompt | self.llm | StrOutputParser()
        self.answer_chain = self.simple_answer_prompt | self.llm | StrOutputParser()
    
    def route_query(
        self,
        question: str,
        image_input: Optional[str] = None,
        options: Optional[Dict[str, str]] = None,
        conversation_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze input and determine routing.
        
        Args:
            question: The question text
            image_input: Path to image (if any)
            options: Multiple choice options (if any)
            conversation_context: Previous conversation context
            
        Returns:
            Routing decision dictionary
        """
        try:
            has_image = "yes" if image_input else "no"
            has_options = "yes" if options else "no"
            
            # Format conversation context
            context_str = ""
            if conversation_context:
                context_str = f"{conversation_context}\n\n"
            
            result = self.routing_chain.invoke({
                "conversation_context": context_str,
                "question": question or "No question provided",
                "has_image": has_image,
                "has_options": has_options
            })
            
            # Parse JSON response
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            decision = json.loads(result)
            
            # Override routing if image is present
            if image_input:
                decision['route_to'] = 'image_qa'
                decision['requires_image_analysis'] = True
            
            print(f"[Master Coordinator] Route decision: {decision['route_to']}")
            print(f"[Master Coordinator] Reasoning: {decision.get('reasoning', 'N/A')}")
            
            return decision
            
        except Exception as e:
            print(f"[Master Coordinator] Error in routing: {e}")
            # Default to medical_qa for safety
            return {
                "input_type": "text",
                "complexity": "moderate",
                "can_answer_directly": False,
                "requires_image_analysis": bool(image_input),
                "requires_medical_research": True,
                "route_to": "image_qa" if image_input else "medical_qa",
                "reasoning": f"Error in routing, defaulting to safe option: {str(e)}"
            }
    
    def answer_simple_question(
        self,
        question: str,
        options: Optional[Dict[str, str]] = None,
        conversation_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a simple question directly.
        
        Args:
            question: The question to answer
            options: Multiple choice options (if any)
            conversation_context: Previous conversation context
            
        Returns:
            Answer dictionary
        """
        try:
            print("[Master Coordinator] Answering simple question directly...")
            
            # Build question text with context
            question_text = ""
            if conversation_context:
                question_text += f"{conversation_context}\n\n"
            
            question_text += question
            
            if options:
                question_text += "\n\nOptions:\n"
                for key, value in options.items():
                    question_text += f"{key}. {value}\n"
            
            # Get answer
            result = self.answer_chain.invoke({"question_text": question_text})
            
            # Parse answer
            answer = self._parse_simple_answer(result, options)
            
            print(f"[Master Coordinator] Direct answer: {answer['answer']}")
            
            return answer
            
        except Exception as e:
            print(f"[Master Coordinator] Error answering simple question: {e}")
            return {
                "answer": "",
                "explanation": f"Error generating answer: {str(e)}",
                "confidence": 0.0,
                "method": "direct_answer",
                "error": str(e)
            }
    
    def _parse_simple_answer(
        self,
        result: str,
        options: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Parse the simple answer response.
        
        Args:
            result: Raw LLM output
            options: Multiple choice options
            
        Returns:
            Parsed answer dictionary
        """
        answer = ""
        explanation = result
        confidence = 0.8  # Default confidence for simple answers
        
        # Try to extract structured answer
        lines = result.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith('Answer:'):
                answer = line.replace('Answer:', '').strip()
                # Get explanation from remaining lines
                explanation_lines = lines[i+1:]
                explanation = '\n'.join(explanation_lines).strip()
                if explanation.startswith('Explanation:'):
                    explanation = explanation.replace('Explanation:', '').strip()
                break
        
        # If no structured format, use entire result as explanation
        if not answer:
            # Try to extract answer from options if available
            if options:
                for key, value in options.items():
                    if key in result[:100]:  # Check first 100 chars
                        answer = f"{key}"
                        break
            
            # If still no answer, use first sentence
            if not answer:
                sentences = result.split('.')
                answer = sentences[0].strip() if sentences else result[:100]
        
        return {
            "answer": answer,
            "explanation": explanation,
            "confidence": confidence,
            "method": "direct_answer",
            "raw_output": result
        }
    
    def analyze_and_route(
        self,
        question: Optional[str] = None,
        image_input: Optional[str] = None,
        options: Optional[Dict[str, str]] = None,
        conversation_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete analysis and routing decision.
        
        Args:
            question: Text question
            image_input: Image path/URL
            options: Answer options
            conversation_context: Previous conversation context
            
        Returns:
            Complete routing information
        """
        # Validate input
        if not question and not image_input:
            return {
                "route_to": "error",
                "error": "No input provided (need question or image)",
                "can_answer_directly": False
            }
        
        # Get routing decision
        routing_decision = self.route_query(
            question=question or "Analyze this image",
            image_input=image_input,
            options=options,
            conversation_context=conversation_context
        )
        
        # Add input information
        routing_decision['question'] = question
        routing_decision['image_input'] = image_input
        routing_decision['options'] = options
        
        return routing_decision

