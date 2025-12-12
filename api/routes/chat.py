"""Chat API routes."""

import os
import time
import json
import uuid
import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse

from api.schemas import (
    ChatRequest, ChatResponse, SessionInfo, SessionListResponse,
    HistoryResponse, ConversationTurn, ExportData, ImageUploadResponse,
    ErrorResponse, ParseQuestionRequest, ParseQuestionResponse
)
from api.session_store import session_store
from utils.config import Config

router = APIRouter(prefix="/api", tags=["chat"])

# Upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/chat/send", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Send a message and get a response.
    
    This endpoint processes the message synchronously and returns the complete response.
    """
    try:
        # Get or create session
        session = session_store.get_or_create_session(session_id=request.session_id)
        
        # Process the message
        result = session.chat(
            message=request.message,
            image_input=request.image_path,
            options=request.options,
            question_type=request.question_type.value if request.question_type else "multiple_choice"
        )
        
        # Build response
        response = ChatResponse(
            answer=result.get('answer', ''),
            explanation=result.get('explanation', ''),
            confidence=result.get('confidence', 0.0),
            session_id=session.session_id,
            turn_number=result.get('turn_number', 0),
            workflow_used=result.get('workflow_used'),
            execution_time=result.get('execution_time'),
            sources_count=result.get('sources_count'),
            routing_decision=result.get('routing_decision'),
            medprompt_info=result.get('medprompt'),
            reflexion_info=result.get('reflexion')
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def stream_message(request: ChatRequest):
    """
    Send a message and get a streaming response using Server-Sent Events (SSE).
    
    The response streams chunks as they become available.
    """
    async def generate():
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'node', 'content': 'Initializing Session'})}\n\n"
            
            # Get or create session
            session = session_store.get_or_create_session(session_id=request.session_id)
            
            yield f"data: {json.dumps({'type': 'node', 'content': 'Session Ready'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Determine if this is an image or text question
            has_image = bool(request.image_path)
            has_options = bool(request.options)
            
            # Send workflow start status
            yield f"data: {json.dumps({'type': 'node', 'content': 'Running Master Coordinator'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Create a queue for status updates
            import queue
            import threading
            import sys
            from io import StringIO
            
            status_queue = queue.Queue()
            result_holder = {'result': None, 'error': None}
            
            # Custom stdout capture to get workflow logs
            class LogCapture:
                def __init__(self, original_stdout, status_queue):
                    self.original_stdout = original_stdout
                    self.status_queue = status_queue
                    self.node_keywords = [
                        ('Running Master Coordinator', 'Master Coordinator'),
                        ('Master Coordinator] Route decision', 'Route Decision'),
                        ('Routing to:', 'Routing'),
                        ('Running Medical QA Subgraph', 'Medical QA Subgraph'),
                        ('Running Image QA Subgraph', 'Image QA Subgraph'),
                        ('Running Coordinator', 'Coordinator'),
                        ('Running Parallel Search', 'Parallel Search + Reasoning'),
                        ('Running Web Search', 'Web Search'),
                        ('Running Reasoning', 'Reasoning'),
                        ('Running Validator', 'Validator'),
                        ('Running Choice Shuffling', 'Choice Shuffling Ensemble'),
                        ('Running Answer Generator', 'Answer Generator'),
                        ('Running Reflexion', 'Reflexion (Self-Correction)'),
                        ('Reflexion] Starting', 'Reflexion Iteration'),
                        ('Direct Answer', 'Direct Answer'),
                    ]
                
                def write(self, text):
                    self.original_stdout.write(text)
                    # Check for node keywords
                    for keyword, node_name in self.node_keywords:
                        if keyword in text:
                            self.status_queue.put(node_name)
                            break
                
                def flush(self):
                    self.original_stdout.flush()
            
            def run_chat():
                # Capture stdout
                original_stdout = sys.stdout
                log_capture = LogCapture(original_stdout, status_queue)
                sys.stdout = log_capture
                
                try:
                    result_holder['result'] = session.chat(
                        message=request.message,
                        image_input=request.image_path,
                        options=request.options,
                        question_type=request.question_type.value if request.question_type else "multiple_choice"
                    )
                except Exception as e:
                    result_holder['error'] = e
                finally:
                    sys.stdout = original_stdout
            
            # Start chat in background thread
            chat_thread = threading.Thread(target=run_chat)
            chat_thread.start()
            
            # Stream status updates while chat is running
            last_node = None
            while chat_thread.is_alive():
                try:
                    node = status_queue.get_nowait()
                    if node != last_node:
                        yield f"data: {json.dumps({'type': 'node', 'content': node})}\n\n"
                        last_node = node
                except queue.Empty:
                    pass
                await asyncio.sleep(0.1)
            
            # Get any remaining status updates
            while not status_queue.empty():
                try:
                    node = status_queue.get_nowait()
                    if node != last_node:
                        yield f"data: {json.dumps({'type': 'node', 'content': node})}\n\n"
                        last_node = node
                except queue.Empty:
                    break
            
            # Check for errors
            if result_holder['error']:
                raise result_holder['error']
            
            result = result_holder['result']
            
            yield f"data: {json.dumps({'type': 'node', 'content': 'Generating Response'})}\n\n"
            
            # Get answer and explanation
            answer = result.get('answer', '')
            explanation = result.get('explanation', '')
            
            # Build full response with answer and explanation
            full_response = f"**Answer:** {answer}"
            if explanation:
                full_response += f"\n\n**Explanation:**\n{explanation}"
            
            # Stream the full response in chunks for better UX
            chunk_size = 50  # Characters per chunk
            
            for i in range(0, len(full_response), chunk_size):
                chunk = full_response[i:i + chunk_size]
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                await asyncio.sleep(0.02)  # Small delay for visual effect
            
            # Send metadata
            metadata = {
                'type': 'metadata',
                'content': {
                    'answer': answer,
                    'explanation': explanation,
                    'confidence': result.get('confidence', 0.0),
                    'session_id': session.session_id,
                    'turn_number': result.get('turn_number', 0),
                    'workflow_used': result.get('workflow_used'),
                    'execution_time': result.get('execution_time'),
                    'sources_count': result.get('sources_count'),
                    'routing_decision': result.get('routing_decision'),
                    'medprompt_info': result.get('medprompt'),
                    'reflexion_info': result.get('reflexion')
                }
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            # Send done signal
            yield f"data: {json.dumps({'type': 'done', 'content': 'Complete'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/parse-question", response_model=ParseQuestionResponse)
async def parse_question(request: ParseQuestionRequest):
    """
    Parse a question using LLM to extract:
    - The main question text
    - Multiple choice options (A, B, C, D, E) if present
    - Question type (multiple_choice, yes_no, open_ended)
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            temperature=0,
            google_api_key=Config.GOOGLE_API_KEY
        )
        
        # Prompt for parsing
        parse_prompt = f"""Analyze the following text and extract structured information.

TEXT:
{request.text}

Your task:
1. Identify if this is a multiple choice question (has options A, B, C, D, E), a yes/no question, or an open-ended question
2. Extract the main question text (without the options)
3. Extract any options if present

Respond in this exact JSON format (no markdown, just pure JSON):
{{
    "question_type": "multiple_choice" | "yes_no" | "open_ended",
    "question": "the main question text without options",
    "has_options": true | false,
    "options": {{"A": "option A text", "B": "option B text", ...}} or null if no options
}}

Rules:
- For yes/no questions, if options are "Yes" and "No", set options as {{"A": "Yes", "B": "No"}}
- For multiple choice, extract all options (A through E if present)
- Keep the question text clean without option labels
- If the text is just a simple question without options, set has_options to false and options to null

Respond ONLY with the JSON, no explanation."""

        # Call LLM
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: llm.invoke(parse_prompt)
        )
        
        # Parse response
        response_text = response.content.strip()
        
        # Clean up response (remove markdown code blocks if present)
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        # Parse JSON
        parsed = json.loads(response_text)
        
        return ParseQuestionResponse(
            success=True,
            question=parsed.get('question', request.text),
            question_type=parsed.get('question_type', 'open_ended'),
            options=parsed.get('options'),
            has_options=parsed.get('has_options', False),
            original_text=request.text
        )
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, return original text as open-ended
        return ParseQuestionResponse(
            success=False,
            question=request.text,
            question_type='open_ended',
            options=None,
            has_options=False,
            original_text=request.text
        )
    except Exception as e:
        return ParseQuestionResponse(
            success=False,
            question=request.text,
            question_type='open_ended',
            options=None,
            has_options=False,
            original_text=request.text
        )


@router.post("/upload/image", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload a medical image for analysis.
    
    Supported formats: JPEG, PNG, GIF, WebP
    """
    try:
        # Validate file type
        allowed_types = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
        if file.content_type not in allowed_types:
            return ImageUploadResponse(
                success=False,
                error=f"Invalid file type: {file.content_type}. Allowed: JPEG, PNG, GIF, WebP"
            )
        
        # Generate unique filename
        ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Return both file path (for backend) and URL (for frontend)
        web_url = f"/uploads/{unique_filename}"
        
        return ImageUploadResponse(
            success=True,
            file_path=file_path,
            filename=unique_filename,
            url=web_url  # Web-accessible URL for frontend display
        )
        
    except Exception as e:
        return ImageUploadResponse(
            success=False,
            error=str(e)
        )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    """List all active sessions."""
    sessions_data = session_store.list_sessions()
    
    sessions = [
        SessionInfo(
            session_id=s['session_id'],
            created_at=s['created_at'],
            last_activity=s['last_activity'],
            duration=s['duration'],
            turn_count=s['turn_count'],
            has_summary=s['has_summary']
        )
        for s in sessions_data
    ]
    
    return SessionListResponse(
        sessions=sessions,
        total=len(sessions)
    )


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get information about a specific session."""
    session = session_store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    info = session.get_session_info()
    
    return SessionInfo(
        session_id=session_id,
        created_at=info['created_at'],
        last_activity=info['last_activity'],
        duration=info['duration'],
        turn_count=info['memory_stats']['total_turns'],
        has_summary=info['memory_stats']['has_summary']
    )


@router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
async def get_session_history(session_id: str, n: Optional[int] = None):
    """
    Get conversation history for a session.
    
    Args:
        session_id: Session ID
        n: Optional number of recent turns to return
    """
    session = session_store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = session.get_history(n)
    
    turns = [
        ConversationTurn(
            turn_id=turn.get('turn_id', i),
            user_message=turn.get('user_message', ''),
            assistant_response=turn.get('assistant_response', ''),
            timestamp=turn.get('timestamp', 0),
            metadata=turn.get('metadata')
        )
        for i, turn in enumerate(history)
    ]
    
    return HistoryResponse(
        session_id=session_id,
        turns=turns,
        summary=session.get_summary()
    )


@router.get("/sessions/{session_id}/export", response_model=ExportData)
async def export_session(session_id: str):
    """Export complete session data as JSON."""
    session = session_store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    export_data = session.export_session()
    info = export_data['session_info']
    
    turns = [
        ConversationTurn(
            turn_id=turn.get('turn_id', i),
            user_message=turn.get('user_message', ''),
            assistant_response=turn.get('assistant_response', ''),
            timestamp=turn.get('timestamp', 0),
            metadata=turn.get('metadata')
        )
        for i, turn in enumerate(export_data.get('history', []))
    ]
    
    return ExportData(
        session_info=SessionInfo(
            session_id=session_id,
            created_at=info['created_at'],
            last_activity=info['last_activity'],
            duration=info['duration'],
            turn_count=info['memory_stats']['total_turns'],
            has_summary=info['memory_stats']['has_summary']
        ),
        history=turns,
        summary=session.get_summary(),
        memory_stats=info.get('memory_stats')
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    deleted = session_store.delete_session(session_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": f"Session {session_id} deleted successfully"}


@router.post("/sessions/{session_id}/clear")
async def clear_session_history(session_id: str):
    """Clear conversation history for a session."""
    session = session_store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.clear_history()
    
    return {"message": f"History cleared for session {session_id}"}

