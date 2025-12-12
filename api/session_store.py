"""Session store for managing ChatSession instances."""

import os
import time
import glob
from typing import Dict, Optional, List
from threading import Lock
from workflows.chat_session import ChatSession, create_chat_session

# Upload directory path
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")


class SessionStore:
    """
    In-memory store for ChatSession instances.
    
    Features:
    - Thread-safe session management
    - Auto-cleanup of inactive sessions
    - Session timeout support
    """
    
    def __init__(self, session_timeout: int = 3600):
        """
        Initialize SessionStore.
        
        Args:
            session_timeout: Session timeout in seconds (default: 1 hour)
        """
        self._sessions: Dict[str, ChatSession] = {}
        self._lock = Lock()
        self._session_timeout = session_timeout
    
    def create_session(
        self,
        session_id: Optional[str] = None,
        **kwargs
    ) -> ChatSession:
        """
        Create a new chat session.
        
        Args:
            session_id: Optional session ID (auto-generated if not provided)
            **kwargs: Additional arguments for ChatSession
            
        Returns:
            New ChatSession instance
        """
        with self._lock:
            # Generate session ID if not provided
            if not session_id:
                session_id = f"session_{int(time.time() * 1000)}"
            
            # Check if session already exists
            if session_id in self._sessions:
                return self._sessions[session_id]
            
            # Create new session with default settings
            session = create_chat_session(
                session_id=session_id,
                max_recent_turns=kwargs.get('max_recent_turns', 5),
                summarize_threshold=kwargs.get('summarize_threshold', 10),
                use_summarization=kwargs.get('use_summarization', True),
                enable_medprompt=kwargs.get('enable_medprompt', True),
                enable_few_shot=kwargs.get('enable_few_shot', None),
                enable_cot=kwargs.get('enable_cot', None),
                enable_ensemble=kwargs.get('enable_ensemble', None),
                enable_self_consistency=kwargs.get('enable_self_consistency', None),
                enable_reflexion=kwargs.get('enable_reflexion', None)
            )
            
            self._sessions[session_id] = session
            return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get an existing session.
        
        Args:
            session_id: Session ID
            
        Returns:
            ChatSession if exists, None otherwise
        """
        with self._lock:
            return self._sessions.get(session_id)
    
    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        **kwargs
    ) -> ChatSession:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Session ID
            **kwargs: Arguments for session creation
            
        Returns:
            ChatSession instance
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        return self.create_session(session_id=session_id, **kwargs)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its associated uploaded images.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                
                # Delete associated images
                self._delete_session_images(session)
                
                del self._sessions[session_id]
                return True
            return False
    
    def _delete_session_images(self, session: ChatSession):
        """Delete all uploaded images associated with a session."""
        try:
            history = session.get_history()
            for turn in history:
                metadata = turn.get('metadata', {})
                image_url = metadata.get('image_input')
                if image_url:
                    # Convert URL (/uploads/xxx.jpg) to file path
                    filename = os.path.basename(image_url)
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"[SessionStore] Deleted image: {filename}")
        except Exception as e:
            print(f"[SessionStore] Error deleting session images: {e}")
    
    def list_sessions(self) -> List[Dict]:
        """
        List all sessions with their info.
        
        Returns:
            List of session info dictionaries
        """
        with self._lock:
            sessions = []
            for session_id, session in self._sessions.items():
                info = session.get_session_info()
                sessions.append({
                    'session_id': session_id,
                    'created_at': info['created_at'],
                    'last_activity': info['last_activity'],
                    'duration': info['duration'],
                    'turn_count': info['memory_stats']['total_turns'],
                    'has_summary': info['memory_stats']['has_summary']
                })
            return sessions
    
    def cleanup_inactive_sessions(self) -> int:
        """
        Remove sessions that have been inactive longer than timeout.
        Also deletes associated uploaded images.
        
        Returns:
            Number of sessions removed
        """
        current_time = time.time()
        removed = 0
        
        with self._lock:
            sessions_to_remove = []
            
            for session_id, session in self._sessions.items():
                info = session.get_session_info()
                if current_time - info['last_activity'] > self._session_timeout:
                    sessions_to_remove.append((session_id, session))
            
            for session_id, session in sessions_to_remove:
                # Delete associated images
                self._delete_session_images(session)
                del self._sessions[session_id]
                removed += 1
        
        return removed
    
    def get_session_count(self) -> int:
        """Get total number of active sessions."""
        with self._lock:
            return len(self._sessions)
    
    def cleanup_all_uploads(self):
        """
        Delete all uploaded images.
        Called on server shutdown.
        """
        try:
            if os.path.exists(UPLOAD_DIR):
                files = glob.glob(os.path.join(UPLOAD_DIR, "*"))
                for f in files:
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"[SessionStore] Error deleting {f}: {e}")
                print(f"[SessionStore] Cleaned up {len(files)} uploaded files")
        except Exception as e:
            print(f"[SessionStore] Error cleaning up uploads: {e}")
    
    def cleanup_all_sessions(self):
        """
        Delete all sessions and their associated images.
        Called on server shutdown.
        """
        with self._lock:
            for session_id, session in list(self._sessions.items()):
                self._delete_session_images(session)
            self._sessions.clear()
            print("[SessionStore] All sessions cleaned up")


# Global session store instance
session_store = SessionStore()

