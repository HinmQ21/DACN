"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum


class QuestionType(str, Enum):
    """Question type enum."""
    MULTIPLE_CHOICE = "multiple_choice"
    YES_NO = "yes_no"
    OPEN_ENDED = "open_ended"


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's message/question")
    session_id: Optional[str] = Field(None, description="Session ID (auto-generated if not provided)")
    image_path: Optional[str] = Field(None, description="Path to uploaded image")
    options: Optional[Dict[str, str]] = Field(None, description="Multiple choice options (e.g., {'A': 'Option 1', 'B': 'Option 2'})")
    question_type: QuestionType = Field(QuestionType.MULTIPLE_CHOICE, description="Type of question")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str = Field(..., description="The answer to the question")
    explanation: str = Field("", description="Detailed explanation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    session_id: str = Field(..., description="Session ID")
    turn_number: int = Field(..., description="Turn number in conversation")
    workflow_used: Optional[str] = Field(None, description="Which workflow was used")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    sources_count: Optional[int] = Field(None, description="Number of sources used")
    routing_decision: Optional[Dict[str, Any]] = Field(None, description="Routing decision details")
    medprompt_info: Optional[Dict[str, Any]] = Field(None, description="Medprompt information")
    reflexion_info: Optional[Dict[str, Any]] = Field(None, description="Reflexion information")


class StreamChunk(BaseModel):
    """Model for streaming response chunks."""
    type: str = Field(..., description="Chunk type: 'token', 'status', 'metadata', 'done', 'error'")
    content: Any = Field(..., description="Chunk content")


class SessionInfo(BaseModel):
    """Session information model."""
    session_id: str
    created_at: float
    last_activity: float
    duration: float
    turn_count: int
    has_summary: bool


class SessionListResponse(BaseModel):
    """Response for listing sessions."""
    sessions: List[SessionInfo]
    total: int


class ConversationTurn(BaseModel):
    """A single turn in conversation."""
    turn_id: int
    user_message: str
    assistant_response: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class HistoryResponse(BaseModel):
    """Response for conversation history."""
    session_id: str
    turns: List[ConversationTurn]
    summary: Optional[str] = None


class ExportData(BaseModel):
    """Complete session export data."""
    session_info: SessionInfo
    history: List[ConversationTurn]
    summary: Optional[str] = None
    memory_stats: Optional[Dict[str, Any]] = None


class ImageUploadResponse(BaseModel):
    """Response for image upload."""
    success: bool
    file_path: Optional[str] = None
    filename: Optional[str] = None
    url: Optional[str] = None  # Web-accessible URL for frontend display
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None


class ParseQuestionRequest(BaseModel):
    """Request model for parsing a question."""
    text: str = Field(..., description="The raw question text to parse")


class ParseQuestionResponse(BaseModel):
    """Response model for parsed question."""
    success: bool = True
    question: str = Field(..., description="The extracted question without options")
    question_type: str = Field("open_ended", description="Type: multiple_choice, yes_no, or open_ended")
    options: Optional[Dict[str, str]] = Field(None, description="Extracted options {A: '...', B: '...'}")
    has_options: bool = Field(False, description="Whether options were detected")
    original_text: str = Field(..., description="Original input text")

