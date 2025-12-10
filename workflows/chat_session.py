"""Chat Session - Multi-turn conversation management."""

from typing import Dict, Any, Optional, List
from workflows.super_graph import SuperGraph
from utils.memory_manager import MemoryManager
import time


class ChatSession:
    """
    Chat Session for multi-turn conversations.
    
    Features:
    - Manages conversation history with MemoryManager
    - Provides context-aware responses
    - Supports all SuperGraph capabilities
    - Tracks session metadata
    """
    
    def __init__(
        self,
        session_id: str = None,
        max_recent_turns: int = 3,
        summarize_threshold: int = 5,
        use_summarization: bool = True,
        **super_graph_kwargs
    ):
        """
        Initialize Chat Session.
        
        Args:
            session_id: Unique session identifier
            max_recent_turns: Number of recent turns to keep in memory
            summarize_threshold: Summarize when history exceeds this threshold
            use_summarization: Enable/disable summarization
            **super_graph_kwargs: Arguments to pass to SuperGraph
        """
        self.session_id = session_id or f"session_{int(time.time())}"
        
        # Initialize components
        self.memory = MemoryManager(
            max_recent_turns=max_recent_turns,
            summarize_threshold=summarize_threshold,
            use_summarization=use_summarization
        )
        
        self.super_graph = SuperGraph(**super_graph_kwargs)
        
        # Session metadata
        self.created_at = time.time()
        self.last_activity = time.time()
        
        print(f"[Chat Session] Created session: {self.session_id}")
        print(f"  - Memory: {max_recent_turns} recent turns, summarize at {summarize_threshold}")
        print(f"  - Summarization: {'Enabled' if use_summarization else 'Disabled'}")
    
    def chat(
        self,
        message: str,
        image_input: Optional[str] = None,
        options: Optional[Dict[str, str]] = None,
        question_type: str = "multiple_choice"
    ) -> Dict[str, Any]:
        """
        Send a message in the conversation.
        
        Args:
            message: User's message/question
            image_input: Optional image input
            options: Optional multiple choice options
            question_type: Type of question
            
        Returns:
            Response dictionary with answer and metadata
        """
        self.last_activity = time.time()
        
        print(f"\n{'='*60}")
        print(f"[Chat Session {self.session_id}] Turn {self.memory.turn_counter + 1}")
        print(f"{'='*60}")
        
        # Get conversation context from memory
        conversation_context = self.memory.get_conversation_context_for_llm(message)
        
        if conversation_context:
            print(f"[Chat Session] Using conversation context ({len(conversation_context)} chars)")
        else:
            print(f"[Chat Session] No previous context (first turn)")
        
        # Run super graph with context
        result = self.super_graph.run(
            question=message,
            image_input=image_input,
            options=options,
            question_type=question_type,
            conversation_context=conversation_context
        )
        
        # Add to memory
        self.memory.add_turn(
            user_message=message,
            assistant_response=result.get('answer', ''),
            metadata={
                'workflow_used': result.get('workflow_used'),
                'confidence': result.get('confidence'),
                'execution_time': result.get('execution_time'),
                'routing_decision': result.get('routing_decision', {})
            }
        )
        
        # Add session info to result
        result['session_id'] = self.session_id
        result['turn_number'] = self.memory.turn_counter
        result['memory_stats'] = self.memory.get_stats()
        
        return result
    
    def get_history(self, n: int = None) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            n: Number of recent turns (default: all stored turns)
            
        Returns:
            List of conversation turns
        """
        turns = self.memory.get_recent_turns(n) if n else self.memory.get_full_history()
        return [turn.to_dict() for turn in turns]
    
    def get_summary(self) -> str:
        """Get current conversation summary."""
        return self.memory.get_summary()
    
    def clear_history(self):
        """Clear conversation history."""
        print(f"[Chat Session {self.session_id}] Clearing history...")
        self.memory.clear()
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information."""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'last_activity': self.last_activity,
            'duration': time.time() - self.created_at,
            'memory_stats': self.memory.get_stats()
        }
    
    def export_session(self) -> Dict[str, Any]:
        """Export complete session data."""
        return {
            'session_info': self.get_session_info(),
            'memory': self.memory.to_dict(),
            'history': self.get_history()
        }
    
    def print_history(self, n: int = None):
        """
        Print conversation history.
        
        Args:
            n: Number of recent turns to print (default: all)
        """
        print(f"\n{'='*60}")
        print(f"Conversation History - Session: {self.session_id}")
        print(f"{'='*60}")
        
        turns = self.memory.get_recent_turns(n) if n else self.memory.get_full_history()
        
        if not turns:
            print("No conversation history yet.")
            return
        
        for turn in turns:
            print(f"\n--- Turn {turn.turn_id} ---")
            print(f"User: {turn.user_message}")
            print(f"Assistant: {turn.assistant_response[:200]}..." 
                  if len(turn.assistant_response) > 200 
                  else f"Assistant: {turn.assistant_response}")
            
            # Show metadata if available
            if turn.metadata:
                workflow = turn.metadata.get('workflow_used', 'N/A')
                confidence = turn.metadata.get('confidence', 0)
                print(f"(Workflow: {workflow}, Confidence: {confidence:.2f})")
        
        # Show summary if exists
        summary = self.memory.get_summary()
        if summary:
            print(f"\n--- Conversation Summary ---")
            print(summary)
        
        print(f"\n{'='*60}")
    
    def __str__(self) -> str:
        """String representation."""
        return f"ChatSession(id={self.session_id}, turns={self.memory.turn_counter})"


def create_chat_session(
    session_id: str = None,
    max_recent_turns: int = 3,
    summarize_threshold: int = 5,
    use_summarization: bool = True,
    enable_medprompt: bool = True,
    enable_few_shot: bool = None,
    enable_cot: bool = None,
    enable_ensemble: bool = None,
    enable_self_consistency: bool = None,
    enable_reflexion: bool = None
) -> ChatSession:
    """
    Create a chat session with specified configuration.
    
    Args:
        session_id: Unique session identifier
        max_recent_turns: Number of recent turns to keep in memory
        summarize_threshold: Summarize when history exceeds this threshold
        use_summarization: Enable/disable summarization
        enable_medprompt: Enable all Medprompt features
        enable_few_shot: Enable few-shot learning
        enable_cot: Enable Chain-of-Thought
        enable_ensemble: Enable ensemble
        enable_self_consistency: Enable self-consistency
        enable_reflexion: Enable reflexion
        
    Returns:
        Configured ChatSession instance
    """
    return ChatSession(
        session_id=session_id,
        max_recent_turns=max_recent_turns,
        summarize_threshold=summarize_threshold,
        use_summarization=use_summarization,
        enable_few_shot=enable_few_shot,
        enable_cot=enable_cot,
        enable_ensemble=enable_ensemble,
        enable_self_consistency=enable_self_consistency,
        enable_reflexion=enable_reflexion
    )


