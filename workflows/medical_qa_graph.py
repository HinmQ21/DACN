"""LangGraph workflow for medical QA system with Medprompt integration."""

import time
from typing import TypedDict, Annotated, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from agents import (
    CoordinatorAgent,
    WebSearchAgent,
    ReasoningAgent,
    ValidatorAgent,
    AnswerGeneratorAgent
)
from utils.config import Config
import asyncio
from concurrent.futures import ThreadPoolExecutor


class AgentState(TypedDict):
    """State của workflow graph."""
    question: str
    options: dict[str, str] | None
    question_type: str
    
    # Coordinator outputs
    analysis: Dict[str, Any]
    few_shot_examples: str  # Formatted few-shot examples
    
    # Parallel agent outputs
    web_search_result: Dict[str, Any] | None
    reasoning_result: Dict[str, Any] | None
    
    # Validator output (including ensemble results)
    validation_result: Dict[str, Any] | None
    ensemble_result: Dict[str, Any] | None
    
    # Final output
    final_answer: Dict[str, Any] | None
    
    # Metadata
    error: str | None
    medprompt_enabled: bool


class MedicalQAWorkflow:
    """
    Workflow chính sử dụng LangGraph với tích hợp Medprompt.
    
    Medprompt components:
    1. Dynamic Few-shot Selection (Coordinator)
    2. Self-Generated CoT (Reasoning)
    3. Choice Shuffling Ensemble (Validator)
    """
    
    def __init__(
        self,
        enable_few_shot: bool = None,
        enable_cot: bool = None,
        enable_ensemble: bool = None,
        enable_self_consistency: bool = None
    ):
        """
        Initialize the workflow.
        
        Args:
            enable_few_shot: Enable dynamic few-shot selection
            enable_cot: Enable Chain-of-Thought reasoning
            enable_ensemble: Enable choice shuffling ensemble
            enable_self_consistency: Enable self-consistency (multiple sampling)
        """
        # Load Medprompt config
        medprompt_config = Config.get_medprompt_config()
        
        self.enable_few_shot = enable_few_shot if enable_few_shot is not None else medprompt_config['enable_few_shot']
        self.enable_cot = enable_cot if enable_cot is not None else medprompt_config['enable_cot']
        self.enable_ensemble = enable_ensemble if enable_ensemble is not None else medprompt_config['enable_ensemble']
        self.enable_self_consistency = enable_self_consistency if enable_self_consistency is not None else medprompt_config['enable_self_consistency']
        self.self_consistency_samples = medprompt_config.get('self_consistency_samples', 3)
        
        # Initialize agents with Medprompt features
        self.coordinator = CoordinatorAgent(enable_few_shot=self.enable_few_shot)
        self.web_search = WebSearchAgent()
        self.reasoning = ReasoningAgent(enable_cot=self.enable_cot)
        self.validator = ValidatorAgent(enable_ensemble=self.enable_ensemble)
        self.answer_generator = AnswerGeneratorAgent()
        
        # Build graph
        self.graph = self._build_graph()
        
        print(f"[Workflow] Initialized with Medprompt features:")
        print(f"  - Few-shot Selection: {self.enable_few_shot}")
        print(f"  - Chain-of-Thought: {self.enable_cot}")
        print(f"  - Choice Shuffling Ensemble: {self.enable_ensemble}")
        print(f"  - Self-Consistency: {self.enable_self_consistency} (samples={self.self_consistency_samples})")
    
    def _coordinator_node(self, state: AgentState) -> AgentState:
        """
        Node: Coordinator analyzes the question and retrieves few-shot examples.
        """
        start_time = time.time()
        try:
            print("[DEBUG] Running Coordinator with Few-shot Selection...")
            analysis = self.coordinator.analyze(
                state['question'],
                options=state.get('options')
            )
            state['analysis'] = analysis
            state['few_shot_examples'] = analysis.get('few_shot_examples_formatted', '')
            
            if analysis.get('num_similar_examples', 0) > 0:
                print(f"[DEBUG] Retrieved {analysis['num_similar_examples']} similar examples")
            
            elapsed = time.time() - start_time
            print(f"[DEBUG] Coordinator completed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Coordinator FAILED after {elapsed:.2f}s: {str(e)}")
            state['error'] = f"Coordinator error: {str(e)}"
            state['analysis'] = {}
            state['few_shot_examples'] = ''
        return state
    
    def _web_search_node(self, state: AgentState) -> AgentState:
        """Node: Web search."""
        start_time = time.time()
        try:
            print("[DEBUG] Running Web Search...")
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
            elapsed = time.time() - start_time
            print(f"[DEBUG] Web Search completed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Web Search FAILED after {elapsed:.2f}s: {str(e)}")
            state['error'] = f"Web search error: {str(e)}"
            state['web_search_result'] = {'synthesis': 'Error in web search', 'total_sources': 0}
        return state
    
    def _reasoning_node(self, state: AgentState) -> AgentState:
        """
        Node: Reasoning agent performs logical analysis with CoT.
        Supports self-consistency (multiple sampling) when enabled.
        """
        start_time = time.time()
        try:
            reasoning_mode = "Self-Consistency" if self.enable_self_consistency else "CoT"
            print(f"[DEBUG] Running Reasoning with {reasoning_mode}...")
            analysis = state.get('analysis', {}) or {}
            
            if analysis.get('needs_reasoning', True):
                # Get few-shot examples and web context
                few_shot_examples = state.get('few_shot_examples', '') or ''
                web_search_result = state.get('web_search_result') or {}
                web_context = web_search_result.get('synthesis', '') if isinstance(web_search_result, dict) else ''
                options = state.get('options')
                
                # Use self-consistency if enabled (multiple sampling with voting)
                if self.enable_self_consistency:
                    result = self.reasoning.reason_with_self_consistency(
                        question=state['question'],
                        options=options,
                        few_shot_examples=few_shot_examples,
                        num_samples=self.self_consistency_samples,
                        context=web_context
                    )
                    print(f"[DEBUG] Self-consistency samples: {result.get('num_samples', 0)}")
                else:
                    # Standard reasoning (with CoT if enabled)
                    result = self.reasoning.reason(
                        question=state['question'],
                        context=web_context,
                        few_shot_examples=few_shot_examples,
                        options=options
                    )
                
                state['reasoning_result'] = result
                print(f"[DEBUG] Reasoning type: {result.get('reasoning_type', 'unknown')}")
            else:
                state['reasoning_result'] = {
                    'raw_output': 'Reasoning skipped based on coordinator analysis',
                    'conclusion': ''
                }
            elapsed = time.time() - start_time
            print(f"[DEBUG] Reasoning completed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Reasoning FAILED after {elapsed:.2f}s: {str(e)}")
            state['error'] = f"Reasoning error: {str(e)}"
            state['reasoning_result'] = {'raw_output': 'Error in reasoning', 'conclusion': ''}
        return state
    
    def _validator_node(self, state: AgentState) -> AgentState:
        """
        Node: Validator validates results.
        For multiple choice with ensemble enabled, uses choice shuffling.
        """
        start_time = time.time()
        try:
            print("[DEBUG] Running Validator...")
            
            # Safely get results with None handling
            web_search_result = state.get('web_search_result') or {}
            reasoning_result = state.get('reasoning_result') or {}
            
            # Standard validation
            validation = self.validator.validate(
                question=state['question'],
                web_search_result=web_search_result,
                reasoning_result=reasoning_result
            )
            state['validation_result'] = validation
            
            elapsed = time.time() - start_time
            print(f"[DEBUG] Validator completed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Validator FAILED after {elapsed:.2f}s: {str(e)}")
            state['error'] = f"Validator error: {str(e)}"
            state['validation_result'] = {
                'is_consistent': True,
                'overall_confidence': 0.5,
                'recommendation': 'proceed'
            }
        return state
    
    def _ensemble_node(self, state: AgentState) -> AgentState:
        """
        Node: Choice Shuffling Ensemble (for multiple choice questions).
        """
        start_time = time.time()
        try:
            options = state.get('options')
            
            if self.enable_ensemble and self.validator.is_multiple_choice(options):
                print("[DEBUG] Running Choice Shuffling Ensemble...")
                
                # Define reasoning function for ensemble
                def reasoning_func(question, opts, web_result, few_shot):
                    web_context = ''
                    if web_result and isinstance(web_result, dict):
                        web_context = web_result.get('synthesis', '')
                    return self.reasoning.reason(
                        question=question,
                        context=web_context,
                        few_shot_examples=few_shot or '',
                        options=opts
                    )
                
                ensemble_result = self.validator.validate_with_ensemble(
                    question=state['question'],
                    options=options,
                    reasoning_func=reasoning_func,
                    web_search_result=state.get('web_search_result') or {},
                    few_shot_examples=state.get('few_shot_examples', '') or ''
                )
                
                state['ensemble_result'] = ensemble_result
                
                elapsed = time.time() - start_time
                print(f"[DEBUG] Ensemble completed in {elapsed:.2f}s: answer={ensemble_result.get('answer')}, "
                      f"confidence={ensemble_result.get('confidence', 0):.2f}, "
                      f"consistency={ensemble_result.get('consistency', 0):.2f}")
            else:
                state['ensemble_result'] = None
                elapsed = time.time() - start_time
                print(f"[DEBUG] Ensemble skipped in {elapsed:.2f}s (not MC or disabled)")
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Ensemble FAILED after {elapsed:.2f}s: {str(e)}")
            state['ensemble_result'] = None
            # Don't set error - ensemble is optional
        
        return state
    
    def _answer_generator_node(self, state: AgentState) -> AgentState:
        """
        Node: Answer generator produces final answer.
        Uses ensemble result if available.
        """
        start_time = time.time()
        try:
            print("[DEBUG] Running Answer Generator...")
            
            # Safely get results with None handling
            web_search_result = state.get('web_search_result') or {}
            reasoning_result = state.get('reasoning_result') or {}
            validation_result = state.get('validation_result') or {}
            options = state.get('options') or {}
            
            # Check if we have ensemble result
            ensemble_result = state.get('ensemble_result')
            
            if ensemble_result and ensemble_result.get('ensemble_used'):
                # Use ensemble answer
                answer = self.answer_generator.generate(
                    question=state['question'],
                    web_search_result=web_search_result,
                    reasoning_result=reasoning_result,
                    validation_result=ensemble_result.get('validation', validation_result),
                    options=list(options.values()) if options else None,
                    question_type=state.get('question_type', 'multiple_choice')
                )
                
                # Override answer with ensemble result ONLY if ensemble has valid answer
                ensemble_answer = ensemble_result.get('answer', '')
                if ensemble_answer:  # Only use ensemble answer if not empty
                    answer['answer'] = ensemble_answer
                # else: keep answer from answer_generator
                
                answer['confidence'] = ensemble_result.get('confidence', answer.get('confidence', 0.0))
                answer['ensemble_consistency'] = ensemble_result.get('consistency', 0.0)
                answer['ensemble_used'] = True
                answer['all_predictions'] = ensemble_result.get('all_predictions', [])
                answer['vote_distribution'] = ensemble_result.get('vote_distribution', {})
                
            else:
                # Standard answer generation
                answer = self.answer_generator.generate(
                    question=state['question'],
                    web_search_result=web_search_result,
                    reasoning_result=reasoning_result,
                    validation_result=validation_result,
                    options=list(options.values()) if options else None,
                    question_type=state.get('question_type', 'multiple_choice')
                )
                answer['ensemble_used'] = False
            
            state['final_answer'] = answer
            elapsed = time.time() - start_time
            print(f"[DEBUG] Answer Generator completed in {elapsed:.2f}s: {answer.get('answer', 'N/A')}")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DEBUG] Answer Generator FAILED after {elapsed:.2f}s: {str(e)}")
            state['error'] = f"Answer generator error: {str(e)}"
            state['final_answer'] = {'answer': '', 'explanation': 'Error generating answer'}
        
        return state
    
    def _parallel_search_and_reason(self, state: AgentState) -> AgentState:
        """Node: Run parallel web search and reasoning."""
        start_time = time.time()
        print("[DEBUG] Running Parallel Search + Reasoning...")
        
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
        
        elapsed = time.time() - start_time
        print(f"[DEBUG] Parallel Search + Reasoning completed in {elapsed:.2f}s")
        
        return state
    
    def _build_graph(self) -> StateGraph:
        """Build workflow graph with Medprompt integration."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("coordinator", self._coordinator_node)
        workflow.add_node("parallel_search", self._parallel_search_and_reason)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("ensemble", self._ensemble_node)
        workflow.add_node("answer_generator", self._answer_generator_node)
        
        # Define edges
        # Coordinator -> Parallel (Web Search + Reasoning)
        workflow.set_entry_point("coordinator")
        workflow.add_edge("coordinator", "parallel_search")
        
        # Parallel -> Validator -> Ensemble -> Answer Generator
        workflow.add_edge("parallel_search", "validator")
        workflow.add_edge("validator", "ensemble")
        workflow.add_edge("ensemble", "answer_generator")
        workflow.add_edge("answer_generator", END)
        
        return workflow.compile()
    
    def run(
        self, 
        question: str, 
        options: dict[str, str] = None,
        question_type: str = "multiple_choice"
    ) -> Dict[str, Any]:
        """
        Run workflow for a question.
        
        Args:
            question: Question to answer
            options: Answer options (dict format: {"A": "option1", "B": "option2"})
            question_type: Question type
            
        Returns:
            Dictionary containing final results
        """
        # Start timing
        workflow_start_time = time.time()
        
        initial_state = {
            "question": question,
            "options": options,
            "question_type": question_type,
            "analysis": {},
            "few_shot_examples": "",
            "web_search_result": None,
            "reasoning_result": None,
            "validation_result": None,
            "ensemble_result": None,
            "final_answer": None,
            "error": None,
            "medprompt_enabled": self.enable_few_shot or self.enable_cot or self.enable_ensemble
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        # Calculate total time
        workflow_end_time = time.time()
        total_time = workflow_end_time - workflow_start_time
        
        # Log timing
        print(f"\n{'='*60}")
        print(f"[WORKFLOW] Total execution time: {total_time:.2f} seconds")
        print(f"{'='*60}\n")
        
        # Build result
        result = {
            "question": question,
            "answer": final_state.get('final_answer', {}).get('answer', ''),
            "explanation": final_state.get('final_answer', {}).get('explanation', ''),
            "confidence": final_state.get('final_answer', {}).get('confidence', 0.0),
            "sources_count": final_state.get('final_answer', {}).get('sources_count', 0),
            "parsed_successfully": final_state.get('final_answer', {}).get('parsed_successfully', False),
            "coordinator_analysis": final_state.get('analysis', {}),
            "validation": final_state.get('validation_result', {}),
            "error": final_state.get('error'),
            "execution_time": total_time,  # Add execution time to result
            
            # Medprompt-specific outputs
            "medprompt": {
                "enabled": final_state.get('medprompt_enabled', False),
                "few_shot_count": final_state.get('analysis', {}).get('num_similar_examples', 0),
                "cot_used": final_state.get('reasoning_result', {}).get('reasoning_type') == 'cot',
                "ensemble_used": final_state.get('final_answer', {}).get('ensemble_used', False),
                "ensemble_consistency": final_state.get('final_answer', {}).get('ensemble_consistency', None),
                "all_predictions": final_state.get('final_answer', {}).get('all_predictions', []),
                "vote_distribution": final_state.get('final_answer', {}).get('vote_distribution', {})
            }
        }
        
        return result
    
    def run_simple(
        self,
        question: str,
        options: dict[str, str] = None
    ) -> str:
        """
        Run workflow and return just the answer.
        
        Args:
            question: Question to answer
            options: Answer options
            
        Returns:
            The answer string
        """
        result = self.run(question, options)
        return result.get('answer', '')


# Factory function for easy instantiation
def create_workflow(
    enable_medprompt: bool = True,
    enable_few_shot: bool = None,
    enable_cot: bool = None,
    enable_ensemble: bool = None,
    enable_self_consistency: bool = None
) -> MedicalQAWorkflow:
    """
    Create a MedicalQAWorkflow instance.
    
    Args:
        enable_medprompt: Enable all Medprompt features (can be overridden)
        enable_few_shot: Enable few-shot selection
        enable_cot: Enable Chain-of-Thought
        enable_ensemble: Enable choice shuffling ensemble
        enable_self_consistency: Enable self-consistency (multiple sampling)
        
    Returns:
        Configured MedicalQAWorkflow instance
    """
    if not enable_medprompt:
        return MedicalQAWorkflow(
            enable_few_shot=False,
            enable_cot=False,
            enable_ensemble=False,
            enable_self_consistency=False
        )
    
    return MedicalQAWorkflow(
        enable_few_shot=enable_few_shot,
        enable_cot=enable_cot,
        enable_ensemble=enable_ensemble,
        enable_self_consistency=enable_self_consistency
    )
