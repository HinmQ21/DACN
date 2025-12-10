"""Memory Manager - Manages conversation history with summarization."""

from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.config import Config
import json


class ConversationTurn:
    """Represents a single turn in the conversation."""
    
    def __init__(
        self,
        turn_id: int,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a conversation turn.
        
        Args:
            turn_id: Unique identifier for the turn
            user_message: User's input message
            assistant_response: Assistant's response
            metadata: Additional metadata (routing, confidence, etc.)
        """
        self.turn_id = turn_id
        self.user_message = user_message
        self.assistant_response = assistant_response
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_id": self.turn_id,
            "user_message": self.user_message,
            "assistant_response": self.assistant_response,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"Turn {self.turn_id}:\nUser: {self.user_message}\nAssistant: {self.assistant_response}"


class MemoryManager:
    """
    Memory Manager for multi-turn conversations.
    
    Features:
    - Stores conversation history
    - Summarizes old conversations when threshold is reached
    - Provides context for new queries
    - Maintains short-term memory for recent turns
    """
    
    def __init__(
        self,
        max_recent_turns: int = 3,
        summarize_threshold: int = 5,
        use_summarization: bool = True
    ):
        """
        Initialize Memory Manager.
        
        Args:
            max_recent_turns: Number of recent turns to keep in full detail
            summarize_threshold: Summarize when history exceeds this many turns
            use_summarization: Enable/disable summarization
        """
        self.max_recent_turns = max_recent_turns
        self.summarize_threshold = summarize_threshold
        self.use_summarization = use_summarization
        
        # Storage
        self.conversation_history: List[ConversationTurn] = []
        self.summary: str = ""
        self.turn_counter = 0
        
        # LLM for summarization
        self.llm = ChatGoogleGenerativeAI(**Config.get_llm_config('coordinator'))
        
        # Summarization prompt
        self.summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that summarizes medical conversations.
Create a concise summary of the conversation history that captures:
1. Main topics discussed
2. Key medical information shared
3. Important questions asked and answered
4. Any diagnostic or treatment information provided

Keep the summary brief but informative, focusing on medical context.
Format: Write in third person, past tense."""),
            ("human", """Previous Summary (if any):
{previous_summary}

New Conversation Turns to Summarize:
{conversation_turns}

Please provide a concise summary of the entire conversation.""")
        ])
        
        # Context building prompt
        self.context_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that builds conversation context.
Given a conversation summary and recent turns, create a brief context paragraph
that will help answer the new question while maintaining conversation continuity.

Focus on relevant medical information and previous topics that may relate to the new question."""),
            ("human", """Conversation Summary:
{summary}

Recent Turns:
{recent_turns}

New Question:
{new_question}

Provide a brief context paragraph (2-3 sentences) that's relevant to the new question.""")
        ])
        
        self.summarization_chain = self.summarization_prompt | self.llm | StrOutputParser()
        self.context_chain = self.context_prompt | self.llm | StrOutputParser()
        
        print("[Memory Manager] Initialized")
        print(f"  - Max recent turns: {max_recent_turns}")
        print(f"  - Summarize threshold: {summarize_threshold}")
        print(f"  - Summarization: {'Enabled' if use_summarization else 'Disabled'}")
    
    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """
        Add a new conversation turn.
        
        Args:
            user_message: User's input
            assistant_response: Assistant's response
            metadata: Additional metadata
            
        Returns:
            The created conversation turn
        """
        self.turn_counter += 1
        turn = ConversationTurn(
            turn_id=self.turn_counter,
            user_message=user_message,
            assistant_response=assistant_response,
            metadata=metadata
        )
        
        self.conversation_history.append(turn)
        
        print(f"[Memory Manager] Added turn {self.turn_counter}")
        
        # Check if summarization is needed
        if self.use_summarization and len(self.conversation_history) >= self.summarize_threshold:
            self._summarize_old_turns()
        
        return turn
    
    def _summarize_old_turns(self):
        """Summarize older conversation turns to save context."""
        print("[Memory Manager] Summarizing old conversation turns...")
        
        # Keep recent turns, summarize the rest
        num_to_summarize = len(self.conversation_history) - self.max_recent_turns
        
        if num_to_summarize <= 0:
            return
        
        turns_to_summarize = self.conversation_history[:num_to_summarize]
        
        # Format conversation for summarization
        conversation_text = "\n\n".join([str(turn) for turn in turns_to_summarize])
        
        try:
            # Generate summary
            new_summary = self.summarization_chain.invoke({
                "previous_summary": self.summary if self.summary else "None",
                "conversation_turns": conversation_text
            })
            
            self.summary = new_summary.strip()
            
            # Remove summarized turns
            self.conversation_history = self.conversation_history[num_to_summarize:]
            
            print(f"[Memory Manager] Summarized {num_to_summarize} turns")
            print(f"[Memory Manager] Summary length: {len(self.summary)} chars")
            
        except Exception as e:
            print(f"[Memory Manager] Error in summarization: {e}")
            # Keep all turns if summarization fails
    
    def get_context(self, new_question: str = "") -> str:
        """
        Get conversation context for a new question.
        
        Args:
            new_question: The new question being asked
            
        Returns:
            Context string to prepend to the question
        """
        if not self.conversation_history and not self.summary:
            return ""
        
        # Format recent turns
        recent_turns_text = "\n".join([
            f"User: {turn.user_message}\nAssistant: {turn.assistant_response[:200]}..."
            if len(turn.assistant_response) > 200
            else f"User: {turn.user_message}\nAssistant: {turn.assistant_response}"
            for turn in self.conversation_history[-self.max_recent_turns:]
        ])
        
        # Simple context building (just return summary + recent turns)
        context_parts = []
        
        if self.summary:
            context_parts.append(f"Previous conversation summary:\n{self.summary}")
        
        if recent_turns_text:
            context_parts.append(f"Recent conversation:\n{recent_turns_text}")
        
        return "\n\n".join(context_parts)
    
    def get_conversation_context_for_llm(self, new_question: str = "") -> str:
        """
        Get formatted conversation context optimized for LLM consumption.
        
        Args:
            new_question: The new question being asked
            
        Returns:
            Formatted context string
        """
        if not self.conversation_history and not self.summary:
            return ""
        
        context_parts = []
        
        # Add summary if exists
        if self.summary:
            context_parts.append(f"[Conversation Context]\n{self.summary}")
        
        # Add recent turns
        if self.conversation_history:
            recent_turns = self.conversation_history[-self.max_recent_turns:]
            turns_text = "\n".join([
                f"Previous Q: {turn.user_message}\nPrevious A: {turn.assistant_response[:300]}..."
                if len(turn.assistant_response) > 300
                else f"Previous Q: {turn.user_message}\nPrevious A: {turn.assistant_response}"
                for turn in recent_turns
            ])
            context_parts.append(f"[Recent Exchanges]\n{turns_text}")
        
        if context_parts:
            return "\n\n".join(context_parts) + "\n\n[Current Question]"
        
        return ""
    
    def get_full_history(self) -> List[ConversationTurn]:
        """Get the full conversation history."""
        return self.conversation_history.copy()
    
    def get_recent_turns(self, n: int = None) -> List[ConversationTurn]:
        """
        Get recent conversation turns.
        
        Args:
            n: Number of recent turns (default: max_recent_turns)
            
        Returns:
            List of recent conversation turns
        """
        n = n or self.max_recent_turns
        return self.conversation_history[-n:] if self.conversation_history else []
    
    def get_summary(self) -> str:
        """Get current conversation summary."""
        return self.summary
    
    def clear(self):
        """Clear all conversation history and summary."""
        self.conversation_history = []
        self.summary = ""
        self.turn_counter = 0
        print("[Memory Manager] Cleared all history")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_turns": self.turn_counter,
            "stored_turns": len(self.conversation_history),
            "has_summary": bool(self.summary),
            "summary_length": len(self.summary) if self.summary else 0,
            "max_recent_turns": self.max_recent_turns,
            "summarize_threshold": self.summarize_threshold
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export memory state to dictionary."""
        return {
            "turn_counter": self.turn_counter,
            "summary": self.summary,
            "conversation_history": [turn.to_dict() for turn in self.conversation_history],
            "config": {
                "max_recent_turns": self.max_recent_turns,
                "summarize_threshold": self.summarize_threshold,
                "use_summarization": self.use_summarization
            }
        }
    
    def __str__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"MemoryManager(turns={stats['total_turns']}, "
            f"stored={stats['stored_turns']}, "
            f"has_summary={stats['has_summary']})"
        )


