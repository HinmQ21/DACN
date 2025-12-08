"""LangGraph workflow for medical image analysis and VQA."""

import time
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from agents import ImageAgent
from utils.config import Config


class ImageQAState(TypedDict):
    """State của Image QA workflow."""
    # Input
    image_input: str  # File path or URL
    question: Optional[str]  # Question for VQA (optional)
    options: Optional[Dict[str, str]]  # Answer options for multiple choice
    mode: str  # "analysis" or "vqa"
    
    # Image analysis outputs
    image_analysis: Dict[str, Any]
    
    # Reasoning outputs
    reasoning_result: Dict[str, Any]
    
    # Validation outputs
    validation_result: Dict[str, Any]
    
    # Final output
    final_answer: Dict[str, Any]
    
    # Metadata
    error: Optional[str]
    execution_time: float


class ImageQAWorkflow:
    """
    Subgraph workflow cho phân tích ảnh y tế và VQA.
    
    Hỗ trợ 2 mode:
    1. Analysis mode: Chỉ có ảnh đầu vào -> Phân tích tổng quan
    2. VQA mode: Ảnh + câu hỏi -> Trả lời câu hỏi dựa trên ảnh
    """
    
    def __init__(self):
        """Initialize the Image QA workflow."""
        self.image_agent = ImageAgent()
        self.graph = self._build_graph()
        
        print("[ImageQAWorkflow] Initialized with Image Agent")
        print(f"  - Model: {Config.IMAGE_MODEL}")
    
    def _image_analyzer_node(self, state: ImageQAState) -> ImageQAState:
        """
        Node: Analyze the medical image.
        """
        start_time = time.time()
        try:
            print("[DEBUG] Running Image Analyzer...")
            
            image_input = state['image_input']
            mode = state.get('mode', 'analysis')
            
            if mode == 'vqa' and state.get('question'):
                # VQA mode: Answer specific question
                result = self.image_agent.answer_question(
                    image_input=image_input,
                    question=state['question'],
                    options=state.get('options')
                )
            else:
                # Analysis mode: General image analysis
                result = self.image_agent.analyze_image(image_input)
            
            state['image_analysis'] = result
            
            elapsed = time.time() - start_time
            print(f"[DEBUG] Image Analyzer completed in {elapsed:.2f}s")
            
            if not result.get('success', False):
                state['error'] = result.get('error', 'Unknown error in image analysis')
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Image Analyzer FAILED after {elapsed:.2f}s: {str(e)}")
            state['error'] = f"Image analysis error: {str(e)}"
            state['image_analysis'] = {'success': False, 'error': str(e)}
        
        return state
    
    def _image_reasoning_node(self, state: ImageQAState) -> ImageQAState:
        """
        Node: Perform reasoning based on image analysis.
        """
        start_time = time.time()
        try:
            print("[DEBUG] Running Image Reasoning...")
            
            image_analysis = state.get('image_analysis', {})
            mode = state.get('mode', 'analysis')
            
            if not image_analysis.get('success', False):
                state['reasoning_result'] = {
                    'conclusion': 'Unable to reason due to image analysis failure',
                    'raw_output': ''
                }
                return state
            
            # Build reasoning based on mode
            if mode == 'vqa':
                reasoning = {
                    'question': state.get('question', ''),
                    'answer': image_analysis.get('answer', ''),
                    'explanation': image_analysis.get('explanation', ''),
                    'confidence': image_analysis.get('confidence', 'medium'),
                    'raw_output': image_analysis.get('raw_output', ''),
                    'reasoning_type': 'image_vqa'
                }
            else:
                # Analysis mode reasoning
                findings = image_analysis.get('findings', [])
                interpretation = image_analysis.get('interpretation', '')
                
                reasoning = {
                    'image_type': image_analysis.get('image_type', ''),
                    'region': image_analysis.get('region', ''),
                    'findings': findings,
                    'interpretation': interpretation,
                    'recommendations': image_analysis.get('recommendations', ''),
                    'conclusion': interpretation or 'Analysis completed',
                    'raw_output': image_analysis.get('raw_output', ''),
                    'reasoning_type': 'image_analysis'
                }
            
            state['reasoning_result'] = reasoning
            
            elapsed = time.time() - start_time
            print(f"[DEBUG] Image Reasoning completed in {elapsed:.2f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Image Reasoning FAILED after {elapsed:.2f}s: {str(e)}")
            state['error'] = f"Image reasoning error: {str(e)}"
            state['reasoning_result'] = {'conclusion': '', 'raw_output': ''}
        
        return state
    
    def _image_validator_node(self, state: ImageQAState) -> ImageQAState:
        """
        Node: Validate the image analysis/VQA results.
        """
        start_time = time.time()
        try:
            print("[DEBUG] Running Image Validator...")
            
            image_analysis = state.get('image_analysis', {})
            reasoning_result = state.get('reasoning_result', {})
            
            # Basic validation
            validation = {
                'is_valid': image_analysis.get('success', False),
                'has_findings': bool(image_analysis.get('findings', [])),
                'has_interpretation': bool(image_analysis.get('interpretation', '')),
                'confidence': reasoning_result.get('confidence', 'medium'),
                'recommendation': 'proceed' if image_analysis.get('success') else 'review'
            }
            
            # Map confidence to numeric value
            confidence_map = {'high': 0.9, 'medium': 0.7, 'low': 0.5}
            validation['confidence_score'] = confidence_map.get(
                validation['confidence'], 0.7
            )
            
            state['validation_result'] = validation
            
            elapsed = time.time() - start_time
            print(f"[DEBUG] Image Validator completed in {elapsed:.2f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Image Validator FAILED after {elapsed:.2f}s: {str(e)}")
            state['validation_result'] = {
                'is_valid': False,
                'confidence_score': 0.5,
                'recommendation': 'review'
            }
        
        return state
    
    def _answer_generator_node(self, state: ImageQAState) -> ImageQAState:
        """
        Node: Generate final answer/response.
        """
        start_time = time.time()
        try:
            print("[DEBUG] Running Answer Generator...")
            
            mode = state.get('mode', 'analysis')
            image_analysis = state.get('image_analysis', {})
            reasoning_result = state.get('reasoning_result', {})
            validation_result = state.get('validation_result', {})
            
            if mode == 'vqa':
                # VQA mode answer
                answer = {
                    'answer': reasoning_result.get('answer', ''),
                    'explanation': reasoning_result.get('explanation', ''),
                    'confidence': validation_result.get('confidence_score', 0.7),
                    'mode': 'vqa',
                    'question': state.get('question', ''),
                    'image_source': image_analysis.get('image_source', '')
                }
            else:
                # Analysis mode answer
                findings = reasoning_result.get('findings', [])
                findings_text = '\n'.join([f"- {f}" for f in findings]) if findings else 'No significant findings'
                
                explanation = f"""
Image Type: {reasoning_result.get('image_type', 'Unknown')}
Region: {reasoning_result.get('region', 'Unknown')}

Key Findings:
{findings_text}

Interpretation:
{reasoning_result.get('interpretation', 'N/A')}

Recommendations:
{reasoning_result.get('recommendations', 'N/A')}

Note: This is an AI-assisted analysis and should be reviewed by a qualified healthcare professional.
""".strip()
                
                answer = {
                    'answer': reasoning_result.get('interpretation', 'Analysis completed'),
                    'explanation': explanation,
                    'confidence': validation_result.get('confidence_score', 0.7),
                    'mode': 'analysis',
                    'image_type': reasoning_result.get('image_type', ''),
                    'findings': findings,
                    'image_source': image_analysis.get('image_source', '')
                }
            
            state['final_answer'] = answer
            
            elapsed = time.time() - start_time
            print(f"[DEBUG] Answer Generator completed in {elapsed:.2f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Answer Generator FAILED after {elapsed:.2f}s: {str(e)}")
            state['error'] = f"Answer generation error: {str(e)}"
            state['final_answer'] = {
                'answer': '',
                'explanation': 'Error generating answer',
                'confidence': 0.0
            }
        
        return state
    
    def _build_graph(self) -> StateGraph:
        """Build the Image QA workflow graph."""
        workflow = StateGraph(ImageQAState)
        
        # Add nodes
        workflow.add_node("image_analyzer", self._image_analyzer_node)
        workflow.add_node("image_reasoning", self._image_reasoning_node)
        workflow.add_node("image_validator", self._image_validator_node)
        workflow.add_node("answer_generator", self._answer_generator_node)
        
        # Define edges
        workflow.set_entry_point("image_analyzer")
        workflow.add_edge("image_analyzer", "image_reasoning")
        workflow.add_edge("image_reasoning", "image_validator")
        workflow.add_edge("image_validator", "answer_generator")
        workflow.add_edge("answer_generator", END)
        
        return workflow.compile()
    
    def run(
        self,
        image_input: str,
        question: str = None,
        options: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Run the Image QA workflow.
        
        Args:
            image_input: File path or URL to the medical image
            question: Question about the image (for VQA mode)
            options: Answer options for multiple choice
            
        Returns:
            Dictionary containing results
        """
        workflow_start_time = time.time()
        
        # Determine mode
        mode = 'vqa' if question else 'analysis'
        
        initial_state: ImageQAState = {
            "image_input": image_input,
            "question": question,
            "options": options,
            "mode": mode,
            "image_analysis": {},
            "reasoning_result": {},
            "validation_result": {},
            "final_answer": {},
            "error": None,
            "execution_time": 0.0
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        # Calculate total time
        total_time = time.time() - workflow_start_time
        
        print(f"\n{'='*60}")
        print(f"[IMAGE QA WORKFLOW] Total execution time: {total_time:.2f} seconds")
        print(f"{'='*60}\n")
        
        # Build result
        result = {
            "image_input": image_input,
            "mode": mode,
            "question": question,
            "answer": final_state.get('final_answer', {}).get('answer', ''),
            "explanation": final_state.get('final_answer', {}).get('explanation', ''),
            "confidence": final_state.get('final_answer', {}).get('confidence', 0.0),
            "image_analysis": final_state.get('image_analysis', {}),
            "validation": final_state.get('validation_result', {}),
            "error": final_state.get('error'),
            "execution_time": total_time
        }
        
        # Add mode-specific fields
        if mode == 'analysis':
            result['image_type'] = final_state.get('final_answer', {}).get('image_type', '')
            result['findings'] = final_state.get('final_answer', {}).get('findings', [])
        
        return result
    
    def analyze(self, image_input: str) -> Dict[str, Any]:
        """
        Shortcut for image analysis mode.
        
        Args:
            image_input: File path or URL to the medical image
            
        Returns:
            Analysis results
        """
        return self.run(image_input=image_input)
    
    def ask(
        self,
        image_input: str,
        question: str,
        options: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Shortcut for VQA mode.
        
        Args:
            image_input: File path or URL to the medical image
            question: Question about the image
            options: Answer options for multiple choice
            
        Returns:
            VQA results
        """
        return self.run(image_input=image_input, question=question, options=options)


def detect_input_type(
    question: str = None,
    image_input: str = None
) -> str:
    """
    Detect the type of input and return appropriate workflow type.
    
    Args:
        question: Text question (optional)
        image_input: Image file path or URL (optional)
        
    Returns:
        'image' if image input is present, 'text' otherwise
    """
    if image_input:
        return 'image'
    return 'text'


def create_image_workflow() -> ImageQAWorkflow:
    """
    Create an ImageQAWorkflow instance.
    
    Returns:
        Configured ImageQAWorkflow instance
    """
    return ImageQAWorkflow()

