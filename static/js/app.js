/**
 * Medical QA Chat Application
 * Main JavaScript module for chat functionality
 */

// ===========================================
// State Management
// ===========================================

const state = {
    currentSessionId: null,
    sessions: [],
    isStreaming: false,
    uploadedImagePath: null,
    uploadedImagePreview: null,
    uploadedImageUrl: null,  // Web URL for display
    pendingRequestId: null   // Track pending request to prevent duplicates
};

// ===========================================
// API Functions
// ===========================================

const API = {
    baseUrl: '',
    
    async sendMessage(message, options = {}) {
        const response = await fetch(`${this.baseUrl}/api/chat/send`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                session_id: state.currentSessionId,
                image_path: options.imagePath || null,
                options: options.mcOptions || null,
                question_type: options.questionType || 'multiple_choice'
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to send message');
        }
        
        return response.json();
    },
    
    streamMessage(message, options = {}) {
        return new Promise((resolve, reject) => {
            const body = JSON.stringify({
                message,
                session_id: state.currentSessionId,
                image_path: options.imagePath || null,
                options: options.mcOptions || null,
                question_type: options.questionType || (options.mcOptions ? 'multiple_choice' : 'open_ended')
            });
            
            fetch(`${this.baseUrl}/api/chat/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body
            }).then(response => {
                if (!response.ok) {
                    throw new Error('Stream request failed');
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                const processStream = async () => {
                    let buffer = '';
                    let result = null;
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        
                        if (done) break;
                        
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop(); // Keep incomplete line in buffer
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    
                                    switch (data.type) {
                                        case 'token':
                                            options.onToken?.(data.content);
                                            break;
                                        case 'status':
                                            options.onStatus?.(data.content);
                                            break;
                                        case 'node':
                                            options.onNode?.(data.content);
                                            break;
                                        case 'metadata':
                                            result = data.content;
                                            options.onMetadata?.(data.content);
                                            break;
                                        case 'done':
                                            options.onDone?.();
                                            break;
                                        case 'error':
                                            throw new Error(data.content);
                                    }
                                } catch (e) {
                                    if (e.message !== 'Unexpected end of JSON input') {
                                        console.error('Parse error:', e);
                                    }
                                }
                            }
                        }
                    }
                    
                    resolve(result);
                };
                
                processStream().catch(reject);
            }).catch(reject);
        });
    },
    
    async uploadImage(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${this.baseUrl}/api/upload/image`, {
            method: 'POST',
            body: formData
        });
        
        return response.json();
    },
    
    async getSessions() {
        const response = await fetch(`${this.baseUrl}/api/sessions`);
        return response.json();
    },
    
    async getSession(sessionId) {
        const response = await fetch(`${this.baseUrl}/api/sessions/${sessionId}`);
        return response.json();
    },
    
    async getHistory(sessionId, n = null) {
        const url = n 
            ? `${this.baseUrl}/api/sessions/${sessionId}/history?n=${n}`
            : `${this.baseUrl}/api/sessions/${sessionId}/history`;
        const response = await fetch(url);
        return response.json();
    },
    
    async exportSession(sessionId) {
        const response = await fetch(`${this.baseUrl}/api/sessions/${sessionId}/export`);
        return response.json();
    },
    
    async deleteSession(sessionId) {
        const response = await fetch(`${this.baseUrl}/api/sessions/${sessionId}`, {
            method: 'DELETE'
        });
        return response.json();
    },
    
    async clearSession(sessionId) {
        const response = await fetch(`${this.baseUrl}/api/sessions/${sessionId}/clear`, {
            method: 'POST'
        });
        return response.json();
    },
    
    async parseQuestion(text) {
        const response = await fetch(`${this.baseUrl}/api/parse-question`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        return response.json();
    }
};

// ===========================================
// UI Components
// ===========================================

const UI = {
    elements: {
        messagesContainer: () => document.getElementById('messages-container'),
        messages: () => document.getElementById('messages'),
        welcomeMessage: () => document.getElementById('welcome-message'),
        messageInput: () => document.getElementById('message-input'),
        sendBtn: () => document.getElementById('send-btn'),
        sessionsList: () => document.getElementById('sessions-list'),
        currentSessionInfo: () => document.getElementById('current-session-info'),
        imagePreviewContainer: () => document.getElementById('image-preview-container'),
        imagePreview: () => document.getElementById('image-preview'),
        historyPanel: () => document.getElementById('history-panel'),
        historyContent: () => document.getElementById('history-content'),
        loadingOverlay: () => document.getElementById('loading-overlay'),
        toastContainer: () => document.getElementById('toast-container')
    },
    
    showWelcome() {
        const welcome = this.elements.welcomeMessage();
        const messages = this.elements.messages();
        if (welcome) welcome.style.display = 'block';
        if (messages) messages.innerHTML = '';
    },
    
    hideWelcome() {
        const welcome = this.elements.welcomeMessage();
        if (welcome) welcome.style.display = 'none';
    },
    
    addMessage(role, content, metadata = {}) {
        this.hideWelcome();
        const messages = this.elements.messages();
        
        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}`;
        
        const avatarIcon = role === 'user' 
            ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
            : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>';
        
        const authorName = role === 'user' ? 'You' : 'MedQA Assistant';
        const timeStr = new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
        
        let imageHtml = '';
        if (metadata.imageSrc) {
            imageHtml = `<img class="message-image" src="${metadata.imageSrc}" alt="Uploaded image">`;
        }
        
        let metadataHtml = '';
        if (role === 'assistant' && metadata.confidence !== undefined) {
            metadataHtml = `
                <div class="message-metadata">
                    <span class="metadata-item confidence">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                            <polyline points="22 4 12 14.01 9 11.01"/>
                        </svg>
                        Confidence: ${(metadata.confidence * 100).toFixed(1)}%
                    </span>
                    ${metadata.workflow_used ? `
                        <span class="metadata-item workflow">
                            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="3"/>
                                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                            </svg>
                            ${metadata.workflow_used}
                        </span>
                    ` : ''}
                    ${metadata.execution_time ? `
                        <span class="metadata-item time">
                            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"/>
                                <polyline points="12 6 12 12 16 14"/>
                            </svg>
                            ${metadata.execution_time.toFixed(2)}s
                        </span>
                    ` : ''}
                </div>
            `;
        }
        
        const isAssistant = role === 'assistant';
        const hasOptions = metadata.hasOptions || false;
        // Use markdown for assistant messages OR user messages with options
        const useMarkdown = isAssistant || hasOptions;
        const textClass = useMarkdown ? 'message-text markdown-body' : 'message-text';
        const formattedContent = this.formatText(content, useMarkdown);
        
        messageEl.innerHTML = `
            <div class="message-avatar">${avatarIcon}</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="message-author">${authorName}</span>
                    <span class="message-time">${timeStr}</span>
                </div>
                <div class="message-body">
                    <div class="${textClass}">${formattedContent}</div>
                    ${imageHtml}
                    ${metadataHtml}
                </div>
            </div>
        `;
        
        messages.appendChild(messageEl);
        this.scrollToBottom();
        
        return messageEl;
    },
    
    addStreamingMessage() {
        this.hideWelcome();
        this.resetStreamingBuffer(); // Reset buffer for new message
        const messages = this.elements.messages();
        
        const messageEl = document.createElement('div');
        messageEl.className = 'message assistant';
        messageEl.id = 'streaming-message';
        
        messageEl.innerHTML = `
            <div class="message-avatar">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-header">
                    <span class="message-author">MedQA Assistant</span>
                    <span class="message-time">${new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })}</span>
                </div>
                <div class="message-body">
                    <div class="workflow-status" id="workflow-status">
                        <div class="status-indicator">
                            <div class="status-spinner"></div>
                            <span class="status-text">Initializing...</span>
                        </div>
                        <div class="status-nodes" id="status-nodes"></div>
                    </div>
                    <div class="message-text markdown-body"></div>
                    <div class="streaming-indicator" style="display: none;">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>
        `;
        
        messages.appendChild(messageEl);
        this.scrollToBottom();
        
        return messageEl;
    },
    
    updateWorkflowStatus(nodeName) {
        const statusText = document.querySelector('#streaming-message .status-text');
        const statusNodes = document.getElementById('status-nodes');
        
        if (statusText) {
            statusText.textContent = nodeName;
        }
        
        if (statusNodes) {
            // Add node to history
            const nodeEl = document.createElement('div');
            nodeEl.className = 'status-node completed';
            nodeEl.innerHTML = `
                <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"/>
                </svg>
                <span>${nodeName}</span>
            `;
            statusNodes.appendChild(nodeEl);
            this.scrollToBottom();
        }
    },
    
    hideWorkflowStatus() {
        const workflowStatus = document.querySelector('#streaming-message .workflow-status');
        const streamingIndicator = document.querySelector('#streaming-message .streaming-indicator');
        
        if (workflowStatus) {
            workflowStatus.style.display = 'none';
        }
        if (streamingIndicator) {
            streamingIndicator.style.display = 'flex';
        }
    },
    
    // Store accumulated streaming text
    _streamingBuffer: '',
    
    updateStreamingMessage(text) {
        // Use the streaming message container to find the text element
        // This avoids issues with duplicate IDs from previous messages
        const streamingMessage = document.getElementById('streaming-message');
        const streamingText = streamingMessage?.querySelector('.message-text');
        if (streamingText) {
            // Accumulate text and render as markdown
            this._streamingBuffer += text;
            streamingText.innerHTML = this.formatText(this._streamingBuffer);
            this.scrollToBottom();
        }
    },
    
    resetStreamingBuffer() {
        this._streamingBuffer = '';
    },
    
    finalizeStreamingMessage(metadata = {}) {
        const streamingMessage = document.getElementById('streaming-message');
        if (!streamingMessage) return;
        
        // Remove workflow status
        const workflowStatus = streamingMessage.querySelector('.workflow-status');
        if (workflowStatus) workflowStatus.remove();
        
        // Remove streaming indicator
        const indicator = streamingMessage.querySelector('.streaming-indicator');
        if (indicator) indicator.remove();
        
        // Add metadata
        if (metadata.confidence !== undefined) {
            const body = streamingMessage.querySelector('.message-body');
            const metadataHtml = `
                <div class="message-metadata">
                    <span class="metadata-item confidence">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                            <polyline points="22 4 12 14.01 9 11.01"/>
                        </svg>
                        Confidence: ${(metadata.confidence * 100).toFixed(1)}%
                    </span>
                    ${metadata.workflow_used ? `
                        <span class="metadata-item workflow">
                            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="3"/>
                                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                            </svg>
                            ${metadata.workflow_used}
                        </span>
                    ` : ''}
                    ${metadata.execution_time ? `
                        <span class="metadata-item time">
                            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"/>
                                <polyline points="12 6 12 12 16 14"/>
                            </svg>
                            ${metadata.execution_time.toFixed(2)}s
                        </span>
                    ` : ''}
                </div>
            `;
            body.insertAdjacentHTML('beforeend', metadataHtml);
        }
        
        // Remove streaming ID to allow next message to use it
        streamingMessage.removeAttribute('id');
    },
    
    formatText(text, useMarkdown = true) {
        // Use marked.js for markdown rendering (only for assistant messages)
        if (useMarkdown && typeof marked !== 'undefined') {
            // Configure marked options
            marked.setOptions({
                breaks: false,       // Don't convert single \n to <br> (use double \n for paragraphs)
                gfm: true,           // GitHub Flavored Markdown
                headerIds: false,    // Don't add IDs to headers
                mangle: false        // Don't escape email addresses
            });
            
            // Clean up excessive newlines before parsing
            const cleanedText = text
                .replace(/\n{3,}/g, '\n\n')  // Replace 3+ newlines with 2
                .trim();
            
            return marked.parse(cleanedText);
        }
        // Basic formatting for user messages - escape HTML and preserve line breaks
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\n/g, '<br>');
    },
    
    scrollToBottom() {
        const container = this.elements.messagesContainer();
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    },
    
    updateSessionInfo(info) {
        const el = this.elements.currentSessionInfo();
        if (el && info) {
            el.textContent = `Session: ${info.session_id || state.currentSessionId} | Turns: ${info.turn_count || 0}`;
        }
    },
    
    renderSessions(sessions) {
        const list = this.elements.sessionsList();
        if (!list) return;
        
        if (sessions.length === 0) {
            list.innerHTML = '<div class="no-sessions">No active sessions</div>';
            return;
        }
        
        list.innerHTML = sessions.map(session => `
            <div class="session-item ${session.session_id === state.currentSessionId ? 'active' : ''}">
                <div class="session-item-content" onclick="selectSession('${session.session_id}')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                    </svg>
                    <span class="session-item-text">${session.session_id}</span>
                </div>
                <button class="session-delete-btn" onclick="event.stopPropagation(); deleteSession('${session.session_id}')" title="Delete session">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"/>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                </button>
            </div>
        `).join('');
    },
    
    renderHistory(history) {
        const content = this.elements.historyContent();
        if (!content) return;
        
        if (!history.turns || history.turns.length === 0) {
            content.innerHTML = '<div class="no-history">No conversation history</div>';
            return;
        }
        
        content.innerHTML = history.turns.map(turn => `
            <div class="history-turn">
                <div class="history-turn-header">
                    <span class="turn-number">Turn ${turn.turn_id + 1}</span>
                    <span class="turn-time">${new Date(turn.timestamp * 1000).toLocaleTimeString('vi-VN')}</span>
                </div>
                <div class="history-user">
                    <div class="history-label">User</div>
                    <div class="history-text">${this.formatText(turn.user_message)}</div>
                </div>
                <div class="history-assistant">
                    <div class="history-label">Assistant</div>
                    <div class="history-text">${this.formatText(turn.assistant_response.substring(0, 200))}${turn.assistant_response.length > 200 ? '...' : ''}</div>
                </div>
            </div>
        `).join('');
    },
    
    showLoading(show = true) {
        const overlay = this.elements.loadingOverlay();
        if (overlay) {
            overlay.style.display = show ? 'flex' : 'none';
        }
    },
    
    showToast(message, type = 'info') {
        const container = this.elements.toastContainer();
        if (!container) return;
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `<span class="toast-message">${message}</span>`;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    },
    
    setInputEnabled(enabled) {
        const input = this.elements.messageInput();
        const btn = this.elements.sendBtn();
        
        if (input) input.disabled = !enabled;
        if (btn) btn.disabled = !enabled;
    }
};

// ===========================================
// Core Functions
// ===========================================

async function sendMessage() {
    // CRITICAL: Check streaming state FIRST before any other processing
    // This prevents race condition when user presses Enter multiple times quickly
    if (state.isStreaming) {
        console.log('Request blocked: already streaming');
        // Re-enable input if it was disabled by event handler but request was blocked
        UI.setInputEnabled(true);
        return;
    }
    
    // Double-check: if there's already a pending request, block this one
    if (state.pendingRequestId) {
        console.log('Request blocked: pending request exists', state.pendingRequestId);
        UI.setInputEnabled(true);
        return;
    }
    
    const input = UI.elements.messageInput();
    const message = input?.value?.trim() || '';
    
    // Must have either message or image
    const hasImage = !!state.uploadedImagePath;
    if (!message && !hasImage) {
        UI.showToast('Please enter a message or upload an image', 'info');
        // Re-enable input since validation failed
        UI.setInputEnabled(true);
        return;
    }
    
    // Generate unique request ID to prevent duplicate submissions
    const requestId = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // CRITICAL: Lock state IMMEDIATELY after validation passes
    // This must happen BEFORE any async operation to prevent duplicate requests
    state.isStreaming = true;
    state.pendingRequestId = requestId;
    UI.setInputEnabled(false);  // Ensure disabled (may already be from event handler)
    
    console.log('Request started:', requestId);
    
    // Save image data BEFORE clearing (important!)
    const imagePath = state.uploadedImagePath;
    const imagePreview = state.uploadedImagePreview;
    
    // Initialize options (will be auto-detected by LLM if present)
    let mcOptions = null;
    let questionType = 'multiple_choice';
    let questionToSend = message;
    let displayMessage = message;
    
    // For image-only messages, set a default question
    if (!message && hasImage) {
        questionToSend = "Analyze this medical image and describe what you see.";
        displayMessage = questionToSend;
        UI.showToast('Image analysis mode', 'info');
    }
    // If has text (no image), try to auto-parse options using LLM
    else if (message && !hasImage) {
        try {
            UI.showToast('Analyzing question...', 'info');
            const parsed = await API.parseQuestion(message);
            
            if (parsed.success && parsed.has_options && parsed.options) {
                mcOptions = parsed.options;
                questionType = parsed.question_type;
                questionToSend = parsed.question;
                
                // Build display message with parsed options
                displayMessage = parsed.question + '\n\n**Options:**';
                for (const [key, value] of Object.entries(parsed.options)) {
                    displayMessage += `\n- **${key}.** ${value}`;
                }
                
                // Show notification about detected options
                const optionCount = Object.keys(mcOptions).length;
                const typeLabel = questionType === 'yes_no' ? 'Yes/No' : 'Multiple Choice';
                UI.showToast(`Detected ${typeLabel} question (${optionCount} options)`, 'success');
            }
        } catch (parseError) {
            console.log('Parse failed, using original message:', parseError);
            // Continue with original message if parsing fails
        }
    }
    
    // Show user message with options and image
    UI.addMessage('user', displayMessage, {
        imageSrc: imagePreview,
        hasOptions: !!mcOptions
    });
    
    // Clear input and image AFTER saving the paths
    input.value = '';
    input.style.height = 'auto';
    removeUploadedImage();
    
    // Note: state.isStreaming is already true (set at the beginning)
    // UI input is already disabled
    
    try {
        // Add streaming message placeholder
        UI.addStreamingMessage();
        
        // Track if we've started receiving tokens
        let tokensStarted = false;
        
        // Stream the response (use questionToSend which may be cleaned by LLM parsing)
        const metadata = await API.streamMessage(questionToSend, {
            mcOptions,
            questionType,
            imagePath,  // Pass the saved image path
            onNode: (nodeName) => {
                console.log('Node:', nodeName);
                UI.updateWorkflowStatus(nodeName);
            },
            onToken: (token) => {
                // Hide workflow status when tokens start
                if (!tokensStarted) {
                    tokensStarted = true;
                    UI.hideWorkflowStatus();
                }
                UI.updateStreamingMessage(token);
            },
            onStatus: (status) => {
                console.log('Status:', status);
            },
            onMetadata: (meta) => {
                // Update session ID if new
                if (meta.session_id && !state.currentSessionId) {
                    state.currentSessionId = meta.session_id;
                }
                UI.updateSessionInfo({ 
                    session_id: meta.session_id,
                    turn_count: meta.turn_number 
                });
            },
            onDone: () => {
                console.log('Stream complete');
            }
        });
        
        // Finalize the streaming message with metadata
        UI.finalizeStreamingMessage(metadata || {});
        
        // Refresh sessions list
        loadSessions();
        
    } catch (error) {
        console.error('Send error:', error);
        UI.showToast(error.message || 'Failed to send message', 'error');
        
        // Remove streaming message on error
        const streamingMsg = document.getElementById('streaming-message');
        if (streamingMsg) streamingMsg.remove();
    } finally {
        console.log('Request completed:', state.pendingRequestId);
        state.isStreaming = false;
        state.pendingRequestId = null;  // Clear pending request
        UI.setInputEnabled(true);
        UI.elements.messageInput()?.focus();
    }
}


async function handleImageUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        UI.showToast('Please select an image file', 'error');
        return;
    }
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        state.uploadedImagePreview = e.target.result;
        const preview = UI.elements.imagePreview();
        const container = UI.elements.imagePreviewContainer();
        
        if (preview && container) {
            preview.src = e.target.result;
            container.style.display = 'inline-block';
        }
    };
    reader.readAsDataURL(file);
    
    // Upload to server
    try {
        UI.showLoading(true);
        const result = await API.uploadImage(file);
        
        if (result.success) {
            state.uploadedImagePath = result.file_path;
            state.uploadedImageUrl = result.url;  // Store URL for display
            UI.showToast('Image uploaded successfully', 'success');
        } else {
            throw new Error(result.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        UI.showToast(error.message || 'Failed to upload image', 'error');
        removeUploadedImage();
    } finally {
        UI.showLoading(false);
        if (event.target && event.target.value !== undefined) {
            event.target.value = ''; // Reset file input
        }
    }
}

function removeUploadedImage() {
    state.uploadedImagePath = null;
    state.uploadedImagePreview = null;
    state.uploadedImageUrl = null;
    
    const container = UI.elements.imagePreviewContainer();
    if (container) {
        container.style.display = 'none';
    }
    
    const preview = UI.elements.imagePreview();
    if (preview) {
        preview.src = '';
    }
}


function toggleHistoryPanel() {
    const panel = UI.elements.historyPanel();
    if (panel) {
        panel.classList.toggle('open');
        
        if (panel.classList.contains('open') && state.currentSessionId) {
            loadHistory();
        }
    }
}

async function loadHistory() {
    if (!state.currentSessionId) return;
    
    try {
        const history = await API.getHistory(state.currentSessionId);
        UI.renderHistory(history);
    } catch (error) {
        console.error('Load history error:', error);
    }
}

async function loadSessions() {
    try {
        const data = await API.getSessions();
        state.sessions = data.sessions || [];
        UI.renderSessions(state.sessions);
    } catch (error) {
        console.error('Load sessions error:', error);
    }
}

async function selectSession(sessionId) {
    state.currentSessionId = sessionId;
    
    // Update UI
    UI.renderSessions(state.sessions);
    
    try {
        // Load session info
        const info = await API.getSession(sessionId);
        UI.updateSessionInfo(info);
        
        // Load and display history
        const history = await API.getHistory(sessionId);
        
        // Clear current messages
        UI.showWelcome();
        
        // Display history as messages
        if (history.turns && history.turns.length > 0) {
            for (const turn of history.turns) {
                // Extract data from metadata (stored with assistant response)
                const metadata = turn.metadata || {};
                const userMetadata = {};
                
                // If there was an image, include it in user message display
                if (metadata.image_input) {
                    userMetadata.imageSrc = metadata.image_input;
                }
                
                // Format user message with options if present
                let userMessage = turn.user_message;
                if (metadata.options && Object.keys(metadata.options).length > 0) {
                    userMessage = turn.user_message + '\n\n**Options:**';
                    for (const [key, value] of Object.entries(metadata.options)) {
                        userMessage += `\n- **${key}.** ${value}`;
                    }
                    userMetadata.hasOptions = true;
                }
                
                UI.addMessage('user', userMessage, userMetadata);
                UI.addMessage('assistant', turn.assistant_response, metadata);
            }
        }
        
    } catch (error) {
        console.error('Select session error:', error);
        UI.showToast('Failed to load session', 'error');
    }
}

function createNewSession() {
    // Don't allow creating new session while streaming
    if (state.isStreaming) {
        UI.showToast('Please wait for current request to complete', 'info');
        return;
    }
    
    state.currentSessionId = null;
    state.uploadedImagePath = null;
    state.uploadedImagePreview = null;
    state.uploadedImageUrl = null;
    state.pendingRequestId = null;  // Clear any pending request
    
    UI.showWelcome();
    UI.updateSessionInfo({ session_id: 'New Session', turn_count: 0 });
    UI.renderSessions(state.sessions);
    
    removeUploadedImage();
    
    const input = UI.elements.messageInput();
    if (input) {
        input.value = '';
        input.focus();
    }
}

async function clearCurrentSession() {
    if (!state.currentSessionId) {
        UI.showToast('No active session to clear', 'info');
        return;
    }
    
    if (!confirm('Are you sure you want to clear the conversation history?')) {
        return;
    }
    
    try {
        await API.clearSession(state.currentSessionId);
        UI.showWelcome();
        UI.showToast('History cleared', 'success');
    } catch (error) {
        console.error('Clear session error:', error);
        UI.showToast('Failed to clear history', 'error');
    }
}

async function deleteSession(sessionId) {
    if (!sessionId) return;
    
    if (!confirm('Are you sure you want to delete this session? This cannot be undone.')) {
        return;
    }
    
    try {
        await API.deleteSession(sessionId);
        
        // If deleting current session, reset state
        if (sessionId === state.currentSessionId) {
            state.currentSessionId = null;
            UI.showWelcome();
            UI.updateSessionInfo({ session_id: 'New Session', turn_count: 0 });
        }
        
        // Refresh session list
        await loadSessions();
        
        UI.showToast('Session deleted', 'success');
    } catch (error) {
        console.error('Delete session error:', error);
        UI.showToast('Failed to delete session', 'error');
    }
}

async function exportCurrentSession() {
    if (!state.currentSessionId) {
        UI.showToast('No active session to export', 'info');
        return;
    }
    
    try {
        const data = await API.exportSession(state.currentSessionId);
        
        // Create download
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `medqa-session-${state.currentSessionId}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        UI.showToast('Session exported successfully', 'success');
    } catch (error) {
        console.error('Export error:', error);
        UI.showToast('Failed to export session', 'error');
    }
}

// ===========================================
// Event Handlers
// ===========================================

// Debounce timer for preventing rapid submissions
let sendDebounceTimer = null;

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        event.stopPropagation();
        
        // Immediately check and disable to prevent any further events
        const input = UI.elements.messageInput();
        const btn = UI.elements.sendBtn();
        
        // If already disabled or streaming, ignore completely
        if (input?.disabled || state.isStreaming || state.pendingRequestId) {
            console.log('KeyPress blocked: input disabled or streaming');
            return;
        }
        
        // Disable immediately BEFORE any async operation
        if (input) input.disabled = true;
        if (btn) btn.disabled = true;
        
        // Clear any existing debounce timer
        if (sendDebounceTimer) {
            clearTimeout(sendDebounceTimer);
        }
        
        // Use small delay to batch rapid keypresses into one call
        sendDebounceTimer = setTimeout(() => {
            sendDebounceTimer = null;
            sendMessage();
        }, 50);
    }
}

function handleSendClick() {
    const input = UI.elements.messageInput();
    const btn = UI.elements.sendBtn();
    
    // If already disabled or streaming, ignore
    if (input?.disabled || state.isStreaming || state.pendingRequestId) {
        console.log('Click blocked: input disabled or streaming');
        return;
    }
    
    // Disable immediately
    if (input) input.disabled = true;
    if (btn) btn.disabled = true;
    
    // Clear any existing debounce timer
    if (sendDebounceTimer) {
        clearTimeout(sendDebounceTimer);
    }
    
    sendDebounceTimer = setTimeout(() => {
        sendDebounceTimer = null;
        sendMessage();
    }, 50);
}

function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

// ===========================================
// Initialization
// ===========================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('Medical QA Chat initialized');
    
    // Load sessions
    loadSessions();
    
    // Focus input
    UI.elements.messageInput()?.focus();
    
    // Set up paste image functionality
    document.addEventListener('paste', async (e) => {
        const items = e.clipboardData?.items;
        if (!items) return;
        
        for (const item of items) {
            if (item.type.startsWith('image/')) {
                e.preventDefault();
                
                const file = item.getAsFile();
                if (file) {
                    // Show preview immediately
                    const reader = new FileReader();
                    reader.onload = (event) => {
                        state.uploadedImagePreview = event.target.result;
                        const preview = UI.elements.imagePreview();
                        const container = UI.elements.imagePreviewContainer();
                        
                        if (preview && container) {
                            preview.src = event.target.result;
                            container.style.display = 'inline-block';
                        }
                    };
                    reader.readAsDataURL(file);
                    
                    // Upload to server
                    try {
                        UI.showToast('Uploading pasted image...', 'info');
                        const result = await API.uploadImage(file);
                        
                        if (result.success) {
                            state.uploadedImagePath = result.file_path;
                            state.uploadedImageUrl = result.url;  // Store URL for display
                            UI.showToast('Image pasted successfully', 'success');
                        } else {
                            throw new Error(result.error || 'Upload failed');
                        }
                    } catch (error) {
                        console.error('Paste upload error:', error);
                        UI.showToast('Failed to upload pasted image', 'error');
                        removeUploadedImage();
                    }
                }
                break; // Only handle first image
            }
        }
    });
    
    // Set up drag and drop for images
    const container = UI.elements.messagesContainer();
    if (container) {
        container.addEventListener('dragover', (e) => {
            e.preventDefault();
            container.style.background = 'rgba(0, 217, 255, 0.05)';
        });
        
        container.addEventListener('dragleave', (e) => {
            e.preventDefault();
            container.style.background = '';
        });
        
        container.addEventListener('drop', (e) => {
            e.preventDefault();
            container.style.background = '';
            
            const file = e.dataTransfer.files?.[0];
            if (file && file.type.startsWith('image/')) {
                // Trigger the file input handler
                const input = document.getElementById('image-input');
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                input.files = dataTransfer.files;
                handleImageUpload({ target: input });
            }
        });
    }
});

