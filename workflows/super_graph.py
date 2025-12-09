"""Super Graph - Master workflow integrating medical_qa and image_qa subgraphs."""

import time
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from agents import MasterCoordinatorAgent
from workflows.medical_qa_graph import MedicalQAWorkflow
from workflows.image_qa_graph import ImageQAWorkflow
from utils.config import Config


class SuperState(TypedDict):
    """State của Super Graph."""
    # Input
    question: Optional[str]
    image_input: Optional[str]
    options: Optional[Dict[str, str]]
    question_type: str
    
    # Master coordinator output
    routing_decision: Dict[str, Any]
    route_to: str  # "direct_answer", "medical_qa", "image_qa"
    
    # Subgraph results
    medical_qa_result: Optional[Dict[str, Any]]
    image_qa_result: Optional[Dict[str, Any]]
    direct_answer_result: Optional[Dict[str, Any]]
    
    # Final output
    final_answer: Dict[str, Any]
    
    # Metadata
    error: Optional[str]
    execution_time: float


class SuperGraph:
    """
    Super Graph - Workflow điều phối chính.
    
    Components:
    1. Master Coordinator - Phân tích và routing
    2. Medical QA Subgraph - Xử lý câu hỏi y tế phức tạp
    3. Image QA Subgraph - Xử lý câu hỏi liên quan ảnh
    4. Direct Answer - Trả lời câu hỏi đơn giản
    """
    
    def __init__(
        self,
        enable_few_shot: bool = None,
        enable_cot: bool = None,
        enable_ensemble: bool = None,
        enable_self_consistency: bool = None,
        enable_reflexion: bool = None
    ):
        """
        Initialize Super Graph.
        
        Args:
            enable_few_shot: Enable few-shot learning (for medical_qa)
            enable_cot: Enable Chain-of-Thought (for medical_qa)
            enable_ensemble: Enable ensemble (for medical_qa)
            enable_self_consistency: Enable self-consistency (for medical_qa)
            enable_reflexion: Enable reflexion (for medical_qa)
        """
        # Initialize master coordinator
        self.master_coordinator = MasterCoordinatorAgent()
        
        # Initialize subgraphs (lazy loading)
        self._medical_qa_workflow = None
        self._image_qa_workflow = None
        
        # Store configuration for subgraphs
        self.medprompt_config = {
            'enable_few_shot': enable_few_shot,
            'enable_cot': enable_cot,
            'enable_ensemble': enable_ensemble,
            'enable_self_consistency': enable_self_consistency,
            'enable_reflexion': enable_reflexion
        }
        
        # Build graph
        self.graph = self._build_graph()
        
        print("[Super Graph] Initialized with Master Coordinator")
        print("  - Medical QA Subgraph: Available (lazy loaded)")
        print("  - Image QA Subgraph: Available (lazy loaded)")
        print("  - Direct Answer: Available")
    
    @property
    def medical_qa_workflow(self):
        """Lazy load Medical QA workflow."""
        if self._medical_qa_workflow is None:
            print("[Super Graph] Loading Medical QA Subgraph...")
            self._medical_qa_workflow = MedicalQAWorkflow(
                **{k: v for k, v in self.medprompt_config.items() if v is not None}
            )
        return self._medical_qa_workflow
    
    @property
    def image_qa_workflow(self):
        """Lazy load Image QA workflow."""
        if self._image_qa_workflow is None:
            print("[Super Graph] Loading Image QA Subgraph...")
            self._image_qa_workflow = ImageQAWorkflow()
        return self._image_qa_workflow
    
    def _master_coordinator_node(self, state: SuperState) -> SuperState:
        """
        Node: Master Coordinator analyzes input and decides routing.
        """
        start_time = time.time()
        try:
            print("[DEBUG] Running Master Coordinator...")
            
            routing_decision = self.master_coordinator.analyze_and_route(
                question=state.get('question'),
                image_input=state.get('image_input'),
                options=state.get('options')
            )
            
            state['routing_decision'] = routing_decision
            state['route_to'] = routing_decision.get('route_to', 'medical_qa')
            
            elapsed = time.time() - start_time
            print(f"[DEBUG] Master Coordinator completed in {elapsed:.2f}s")
            print(f"[DEBUG] Routing to: {state['route_to']}")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Master Coordinator FAILED after {elapsed:.2f}s: {str(e)}")
            state['error'] = f"Master coordinator error: {str(e)}"
            state['routing_decision'] = {}
            state['route_to'] = 'medical_qa'  # Default fallback
        
        return state
    
    def _direct_answer_node(self, state: SuperState) -> SuperState:
        """
        Node: Answer simple question directly.
        """
        start_time = time.time()
        try:
            print("[DEBUG] Running Direct Answer...")
            
            result = self.master_coordinator.answer_simple_question(
                question=state.get('question', ''),
                options=state.get('options')
            )
            
            state['direct_answer_result'] = result
            
            elapsed = time.time() - start_time
            print(f"[DEBUG] Direct Answer completed in {elapsed:.2f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Direct Answer FAILED after {elapsed:.2f}s: {str(e)}")
            state['error'] = f"Direct answer error: {str(e)}"
            state['direct_answer_result'] = {
                'answer': '',
                'explanation': 'Error in direct answer',
                'confidence': 0.0
            }
        
        return state
    
    def _medical_qa_node(self, state: SuperState) -> SuperState:
        """
        Node: Route to Medical QA subgraph.
        """
        start_time = time.time()
        try:
            print("[DEBUG] Running Medical QA Subgraph...")
            
            result = self.medical_qa_workflow.run(
                question=state.get('question', ''),
                options=state.get('options'),
                question_type=state.get('question_type', 'multiple_choice')
            )
            
            state['medical_qa_result'] = result
            
            elapsed = time.time() - start_time
            print(f"[DEBUG] Medical QA Subgraph completed in {elapsed:.2f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Medical QA Subgraph FAILED after {elapsed:.2f}s: {str(e)}")
            state['error'] = f"Medical QA error: {str(e)}"
            state['medical_qa_result'] = {
                'answer': '',
                'explanation': 'Error in medical QA',
                'confidence': 0.0
            }
        
        return state
    
    def _image_qa_node(self, state: SuperState) -> SuperState:
        """
        Node: Route to Image QA subgraph.
        """
        start_time = time.time()
        try:
            print("[DEBUG] Running Image QA Subgraph...")
            
            result = self.image_qa_workflow.run(
                image_input=state.get('image_input', ''),
                question=state.get('question'),
                options=state.get('options')
            )
            
            state['image_qa_result'] = result
            
            elapsed = time.time() - start_time
            print(f"[DEBUG] Image QA Subgraph completed in {elapsed:.2f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Image QA Subgraph FAILED after {elapsed:.2f}s: {str(e)}")
            state['error'] = f"Image QA error: {str(e)}"
            state['image_qa_result'] = {
                'answer': '',
                'explanation': 'Error in image QA',
                'confidence': 0.0
            }
        
        return state
    
    def _aggregator_node(self, state: SuperState) -> SuperState:
        """
        Node: Aggregate results from whichever path was taken.
        """
        start_time = time.time()
        try:
            print("[DEBUG] Running Aggregator...")
            
            route_to = state.get('route_to', 'medical_qa')
            
            # Get result based on routing
            if route_to == 'direct_answer':
                result = state.get('direct_answer_result') or {}
                result['workflow_used'] = 'direct_answer'
            elif route_to == 'image_qa':
                result = state.get('image_qa_result') or {}
                result['workflow_used'] = 'image_qa'
            else:  # medical_qa
                result = state.get('medical_qa_result') or {}
                result['workflow_used'] = 'medical_qa'
            
            # Add routing information
            result['routing_decision'] = state.get('routing_decision', {})
            result['routed_to'] = route_to
            
            state['final_answer'] = result
            
            elapsed = time.time() - start_time
            print(f"[DEBUG] Aggregator completed in {elapsed:.2f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Aggregator FAILED after {elapsed:.2f}s: {str(e)}")
            state['error'] = f"Aggregator error: {str(e)}"
            state['final_answer'] = {
                'answer': '',
                'explanation': 'Error in aggregation',
                'confidence': 0.0
            }
        
        return state
    
    def _route_decision(self, state: SuperState) -> str:
        """
        Conditional edge function to determine routing.
        
        Returns:
            Node name to route to
        """
        route_to = state.get('route_to', 'medical_qa')
        print(f"[DEBUG] Routing decision: {route_to}")
        return route_to
    
    def _build_graph(self) -> StateGraph:
        """Build the super graph with conditional routing."""
        workflow = StateGraph(SuperState)
        
        # Add nodes
        workflow.add_node("master_coordinator", self._master_coordinator_node)
        workflow.add_node("direct_answer", self._direct_answer_node)
        workflow.add_node("medical_qa", self._medical_qa_node)
        workflow.add_node("image_qa", self._image_qa_node)
        workflow.add_node("aggregator", self._aggregator_node)
        
        # Set entry point
        workflow.set_entry_point("master_coordinator")
        
        # Add conditional edges from master_coordinator
        workflow.add_conditional_edges(
            "master_coordinator",
            self._route_decision,
            {
                "direct_answer": "direct_answer",
                "medical_qa": "medical_qa",
                "image_qa": "image_qa"
            }
        )
        
        # All paths lead to aggregator
        workflow.add_edge("direct_answer", "aggregator")
        workflow.add_edge("medical_qa", "aggregator")
        workflow.add_edge("image_qa", "aggregator")
        
        # Aggregator leads to end
        workflow.add_edge("aggregator", END)
        
        return workflow.compile()
    
    def run(
        self,
        question: Optional[str] = None,
        image_input: Optional[str] = None,
        options: Optional[Dict[str, str]] = None,
        question_type: str = "multiple_choice"
    ) -> Dict[str, Any]:
        """
        Run the super graph workflow.
        
        Args:
            question: Text question (optional)
            image_input: Image path/URL (optional)
            options: Answer options (optional)
            question_type: Type of question
            
        Returns:
            Complete result dictionary
        """
        # Start timing
        workflow_start_time = time.time()
        
        # Validate input
        if not question and not image_input:
            return {
                "error": "No input provided (need question or image)",
                "answer": "",
                "explanation": "Error: No input provided",
                "confidence": 0.0,
                "execution_time": 0.0
            }
        
        initial_state: SuperState = {
            "question": question,
            "image_input": image_input,
            "options": options,
            "question_type": question_type,
            "routing_decision": {},
            "route_to": "",
            "medical_qa_result": None,
            "image_qa_result": None,
            "direct_answer_result": None,
            "final_answer": {},
            "error": None,
            "execution_time": 0.0
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        # Calculate total time
        total_time = time.time() - workflow_start_time
        
        # Log timing
        print(f"\n{'='*60}")
        print(f"[SUPER GRAPH] Total execution time: {total_time:.2f} seconds")
        print(f"{'='*60}\n")
        
        # Build result
        result = final_state.get('final_answer', {})
        result['execution_time'] = total_time
        result['question'] = question
        result['image_input'] = image_input
        
        # Ensure all required fields exist
        if 'answer' not in result:
            result['answer'] = ''
        if 'explanation' not in result:
            result['explanation'] = ''
        if 'confidence' not in result:
            result['confidence'] = 0.0
        if 'error' not in result:
            result['error'] = final_state.get('error')
        
        return result


def create_super_graph(
    enable_medprompt: bool = True,
    enable_few_shot: bool = None,
    enable_cot: bool = None,
    enable_ensemble: bool = None,
    enable_self_consistency: bool = None,
    enable_reflexion: bool = None
) -> SuperGraph:
    """
    Create a SuperGraph instance.
    
    Args:
        enable_medprompt: Enable all Medprompt features
        enable_few_shot: Enable few-shot learning
        enable_cot: Enable Chain-of-Thought
        enable_ensemble: Enable ensemble
        enable_self_consistency: Enable self-consistency
        enable_reflexion: Enable reflexion
        
    Returns:
        Configured SuperGraph instance
    """
    if not enable_medprompt:
        return SuperGraph(
            enable_few_shot=False,
            enable_cot=False,
            enable_ensemble=False,
            enable_self_consistency=False,
            enable_reflexion=enable_reflexion
        )
    
    return SuperGraph(
        enable_few_shot=enable_few_shot,
        enable_cot=enable_cot,
        enable_ensemble=enable_ensemble,
        enable_self_consistency=enable_self_consistency,
        enable_reflexion=enable_reflexion
    )

