"""LangGraph workflow for medical QA system."""

from typing import TypedDict, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from agents import (
    CoordinatorAgent,
    WebSearchAgent,
    ReasoningAgent,
    ValidatorAgent,
    AnswerGeneratorAgent
)
import asyncio
from concurrent.futures import ThreadPoolExecutor


class AgentState(TypedDict):
    """State của workflow graph."""
    question: str
    options: list[str] | None
    question_type: str
    
    # Coordinator outputs
    analysis: Dict[str, Any]
    
    # Parallel agent outputs
    web_search_result: Dict[str, Any] | None
    reasoning_result: Dict[str, Any] | None
    
    # Validator output
    validation_result: Dict[str, Any] | None
    
    # Final output
    final_answer: Dict[str, Any] | None
    
    # Metadata
    error: str | None


class MedicalQAWorkflow:
    """Workflow chính sử dụng LangGraph."""
    
    def __init__(self):
        self.coordinator = CoordinatorAgent()
        self.web_search = WebSearchAgent()
        self.reasoning = ReasoningAgent()
        self.validator = ValidatorAgent()
        self.answer_generator = AnswerGeneratorAgent()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _coordinator_node(self, state: AgentState) -> AgentState:
        """Node: Coordinator analyzes the question."""
        try:
            analysis = self.coordinator.analyze(state['question'])
            state['analysis'] = analysis
        except Exception as e:
            state['error'] = f"Coordinator error: {str(e)}"
        return state
    
    def _web_search_node(self, state: AgentState) -> AgentState:
        """Node: Web search."""
        try:
            analysis = state.get('analysis', {})
            if analysis.get('needs_web_search', True):
                key_terms = analysis.get('key_terms', [])
                result = self.web_search.search(state['question'], key_terms)
                state['web_search_result'] = result
            else:
                state['web_search_result'] = {
                    'synthesis': 'Web search skipped based on coordinator analysis',
                    'total_sources': 0
                }
        except Exception as e:
            state['error'] = f"Web search error: {str(e)}"
            state['web_search_result'] = {'synthesis': 'Error in web search', 'total_sources': 0}
        return state
    
    def _reasoning_node(self, state: AgentState) -> AgentState:
        """Node: Reasoning agent performs logical analysis."""
        try:
            analysis = state.get('analysis', {})
            if analysis.get('needs_reasoning', True):
                result = self.reasoning.reason(state['question'])
                state['reasoning_result'] = result
            else:
                state['reasoning_result'] = {
                    'raw_output': 'Reasoning skipped based on coordinator analysis',
                    'conclusion': ''
                }
        except Exception as e:
            state['error'] = f"Reasoning error: {str(e)}"
            state['reasoning_result'] = {'raw_output': 'Error in reasoning', 'conclusion': ''}
        return state
    
    def _validator_node(self, state: AgentState) -> AgentState:
        """Node: Validator validates and verifies results."""
        try:
            validation = self.validator.validate(
                question=state['question'],
                web_search_result=state.get('web_search_result', {}),
                reasoning_result=state.get('reasoning_result', {})
            )
            state['validation_result'] = validation
        except Exception as e:
            state['error'] = f"Validator error: {str(e)}"
            state['validation_result'] = {
                'is_consistent': True,
                'overall_confidence': 0.5,
                'recommendation': 'proceed'
            }
        return state
    
    def _answer_generator_node(self, state: AgentState) -> AgentState:
        """Node: Answer generator produces final answer."""
        try:
            answer = self.answer_generator.generate(
                question=state['question'],
                web_search_result=state.get('web_search_result', {}),
                reasoning_result=state.get('reasoning_result', {}),
                validation_result=state.get('validation_result', {}),
                options=state.get('options'),
                question_type=state.get('question_type', 'multiple_choice')
            )
            state['final_answer'] = answer
        except Exception as e:
            state['error'] = f"Answer generator error: {str(e)}"
            state['final_answer'] = {'answer': '', 'explanation': 'Error generating answer'}
        return state
    
    def _parallel_search_and_reason(self, state: AgentState) -> AgentState:
        """Node: Run parallel web search and reasoning."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            web_future = executor.submit(self._web_search_node, state.copy())
            reason_future = executor.submit(self._reasoning_node, state.copy())
            
            # Get results
            web_state = web_future.result()
            reason_state = reason_future.result()
            
            # Merge results
            state['web_search_result'] = web_state.get('web_search_result')
            state['reasoning_result'] = reason_state.get('reasoning_result')
            
            # Merge errors if any
            errors = []
            if web_state.get('error'):
                errors.append(web_state['error'])
            if reason_state.get('error'):
                errors.append(reason_state['error'])
            if errors:
                state['error'] = '; '.join(errors)
        
        return state
    
    def _build_graph(self) -> StateGraph:
        """Build workflow graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("coordinator", self._coordinator_node)
        workflow.add_node("parallel_search", self._parallel_search_and_reason)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("answer_generator", self._answer_generator_node)
        
        # Define edges
        workflow.set_entry_point("coordinator")
        workflow.add_edge("coordinator", "parallel_search")
        workflow.add_edge("parallel_search", "validator")
        workflow.add_edge("validator", "answer_generator")
        workflow.add_edge("answer_generator", END)
        
        return workflow.compile()
    
    def run(
        self, 
        question: str, 
        options: list[str] = None,
        question_type: str = "multiple_choice"
    ) -> Dict[str, Any]:
        """
        Run workflow for a question.
        
        Args:
            question: Question to answer
            options: Answer options (if any)
            question_type: Question type
            
        Returns:
            Dictionary containing final results
        """
        initial_state = {
            "question": question,
            "options": options,
            "question_type": question_type,
            "analysis": {},
            "web_search_result": None,
            "reasoning_result": None,
            "validation_result": None,
            "final_answer": None,
            "error": None
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "question": question,
            "answer": final_state.get('final_answer', {}).get('answer', ''),
            "explanation": final_state.get('final_answer', {}).get('explanation', ''),
            "confidence": final_state.get('final_answer', {}).get('confidence', 0.0),
            "sources_count": final_state.get('final_answer', {}).get('sources_count', 0),
            "coordinator_analysis": final_state.get('analysis', {}),
            "validation": final_state.get('validation_result', {}),
            "error": final_state.get('error')
        }
    
    def visualize(self) -> str:
        """Visualize workflow graph."""
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_mermaid_png()))
            return "Graph visualized"
        except Exception as e:
            return f"Visualization not available: {e}"

