/**
 * ChatTemplate - Reusable chat interface template
 * Provides all core chat functionality that can be extended for specific use cases
 */
class ChatTemplate {
    constructor(config = {}) {
        // Default configuration
        this.config = {
            // UI selectors
            chatHistory: '#chat-history',
            messageInput: '#message-input',
            sendBtn: '#send-btn',
            attachmentBtn: '#attachment-btn',
            micBtn: '#mic-btn',
            fileInput: '#file-input',
            statusDot: '#status-dot',
            backBtn: '#back-btn',
            
            // API endpoints
            generateEndpoint: '/api/generate/stream',
            ttsEndpoint: '/api/tts',
            healthEndpoint: '/health',
            
            // Behavior
            autoFocus: true,
            enableTTS: true,
            enableAttachments: true,
            enableAudio: true,
            maxTokens: 32768,
            
            // Customization hooks
            welcomeMessage: this.getDefaultWelcomeMessage(),
            backUrl: '/',
            
            ...config
        };

        // Core state
        this.attachedFiles = new Map();
        this.isGenerating = false;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.ttsAvailable = false;
        this.currentAudio = null;
        this.isStreaming = false;
        this.currentStreamingMessage = null;
        
        // Initialize markdown-it
        this.md = markdownit({
            html: true,
            linkify: true,
            typographer: true,
            breaks: true
        });
    }

    /**
     * Initialize the chat interface
     */
    async initialize() {
        this.initializeElements();
        this.initializeEventListeners();
        this.setupStatusBar();
        this.setupWelcomeMessage();
        this.autoResizeTextarea();
        this.updateSendButton();
        await this.checkServerStatus();
        this.startStatusPolling();
        
        if (this.config.autoFocus) {
            this.messageInput.focus();
        }
    }

    /**
     * Get DOM elements
     */
    initializeElements() {
        this.chatHistory = document.querySelector(this.config.chatHistory);
        this.messageInput = document.querySelector(this.config.messageInput);
        this.sendBtn = document.querySelector(this.config.sendBtn);
        this.attachmentBtn = document.querySelector(this.config.attachmentBtn);
        this.micBtn = document.querySelector(this.config.micBtn);
        this.fileInput = document.querySelector(this.config.fileInput);
        this.statusDot = document.querySelector(this.config.statusDot);
        this.backBtn = document.querySelector(this.config.backBtn);
        
        if (!this.chatHistory || !this.messageInput || !this.sendBtn) {
            throw new Error('Required elements not found. Check your HTML structure.');
        }
    }

    /**
     * Set up event listeners
     */
    initializeEventListeners() {
        // Back button
        if (this.backBtn) {
            this.backBtn.addEventListener('click', () => {
                window.location.href = this.config.backUrl;
            });
        }

        // Message input and sending
        this.sendBtn.addEventListener('click', this.handleSubmit.bind(this));
        this.messageInput.addEventListener('keydown', this.handleKeyDown.bind(this));
        this.messageInput.addEventListener('input', () => {
            this.autoResizeTextarea();
            this.updateSendButton();
        });

        // File attachments
        if (this.attachmentBtn && this.fileInput && this.config.enableAttachments) {
            this.attachmentBtn.addEventListener('click', () => {
                this.fileInput.click();
            });
            this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }

        // Audio recording
        if (this.micBtn && this.config.enableAudio) {
            this.micBtn.addEventListener('click', this.handleMicClick.bind(this));
        }

        // Page visibility and beforeunload handlers
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.currentAudio) {
                this.currentAudio.pause();
            }
        });

        window.addEventListener('beforeunload', (e) => {
            if (this.messageInput.value.trim() || this.attachedFiles.size > 0) {
                e.preventDefault();
                e.returnValue = '';
            }
        });
    }

    /**
     * Set up status bar - can be overridden by subclasses
     */
    setupStatusBar() {
        // Default implementation - can be overridden
    }

    /**
     * Set up welcome message
     */
    setupWelcomeMessage() {
        if (this.config.welcomeMessage) {
            this.chatHistory.innerHTML = this.config.welcomeMessage;
        }
    }

    /**
     * Default welcome message - can be overridden
     */
    getDefaultWelcomeMessage() {
        return `
            <div class="welcome-message">
                <p>Chat with frende</p>
                <p>Your private on-device AI companion</p>
            </div>
        `;
    }

    /**
     * Auto-resize textarea
     */
    autoResizeTextarea() {
        const textarea = this.messageInput;
        textarea.style.height = 'auto';
        
        const maxHeight = 120;
        if (textarea.scrollHeight > maxHeight) {
            textarea.style.height = maxHeight + 'px';
            textarea.style.overflowY = 'auto';
        } else {
            textarea.style.height = textarea.scrollHeight + 'px';
            textarea.style.overflowY = 'hidden';
        }
    }

    /**
     * Update send button state
     */
    updateSendButton() {
        const hasText = this.messageInput.value.trim().length > 0;
        const hasFiles = this.attachedFiles.size > 0;
        this.sendBtn.disabled = !hasText && !hasFiles;
        
        if (this.isGenerating) {
            this.sendBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M6 6h12v12H6z"/></svg>'; // Stop icon
        } else {
            this.sendBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>'; // Send icon
        }
    }

    /**
     * Handle keyboard events
     */
    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!this.sendBtn.disabled || this.isGenerating) {
                this.handleSubmit();
            }
        }
    }

    /**
     * Handle file selection
     */
    handleFileSelect(e) {
        const files = e.target.files;
        for (let file of files) {
            if (!this.attachedFiles.has(file.name)) {
                this.attachedFiles.set(file.name, file);
            }
        }
        this.renderAttachments();
        this.updateSendButton();
        e.target.value = '';
    }

    /**
     * Handle microphone recording
     */
    async handleMicClick() {
        if (!this.isRecording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                this.mediaRecorder = new MediaRecorder(stream);
                this.audioChunks = [];
                
                this.mediaRecorder.ondataavailable = (event) => {
                    this.audioChunks.push(event.data);
                };
                
                this.mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(this.audioChunks, { type: "audio/webm" });
                    const audioName = `recording-${Date.now()}.webm`;
                    this.attachedFiles.set(audioName, audioBlob);
                    this.renderAttachments();
                    this.updateSendButton();
                    stream.getTracks().forEach(track => track.stop());
                };
                
                this.mediaRecorder.start();
                this.isRecording = true;
                this.micBtn.classList.add('recording');
            } catch (err) {
                alert("Could not access microphone. Please check permissions.");
            }
        } else {
            if (this.mediaRecorder) this.mediaRecorder.stop();
            this.isRecording = false;
            this.micBtn.classList.remove('recording');
        }
    }

    /**
     * Render file attachments
     */
    renderAttachments() {
        const display = document.getElementById('attachment-display');
        if (!display) return;
        
        display.innerHTML = '';
        
        if (this.attachedFiles.size === 0) {
            display.classList.remove('active');
            return;
        }
        
        display.classList.add('active');
        
        this.attachedFiles.forEach((file, name) => {
            const pill = document.createElement('div');
            pill.className = 'attachment-pill';
            
            const span = document.createElement('span');
            span.textContent = name;
            
            const btn = document.createElement('button');
            btn.className = 'remove-attachment';
            btn.textContent = '×';
            btn.onclick = () => {
                this.attachedFiles.delete(name);
                this.renderAttachments();
                this.updateSendButton();
            };
            
            pill.appendChild(span);
            pill.appendChild(btn);
            display.appendChild(pill);
        });
    }

    /**
     * Handle message submission - can be overridden for custom processing
     */
    async handleSubmit() {
        if (this.isGenerating) {
            this.stopGeneration();
            return;
        }
        
        const message = this.messageInput.value.trim();
        if (!message && this.attachedFiles.size === 0) return;

        this.isGenerating = true;
        this.updateSendButton();

        // Process message - can be overridden by subclasses
        const processedMessage = this.processMessage(message);

        // Store attachments for the message before clearing
        const messageAttachments = new Map(this.attachedFiles);

        // Add user message to chat
        this.addMessage('user', message, messageAttachments);
        
        // Clear input and attachments immediately
        this.messageInput.value = '';
        this.autoResizeTextarea();
        this.clearAttachments();
        
        try {
            const formData = new FormData();
            formData.append('text', processedMessage);
            formData.append('max_tokens', this.config.maxTokens.toString());
            
            // Add attached files from the stored copy
            messageAttachments.forEach((file, name) => {
                if (file.type.startsWith('image/')) {
                    formData.append('image', file);
                } else if (file.type.startsWith('audio/')) {
                    formData.append('audio', file);
                }
            });

            await this.handleStreamingGeneration(formData);

        } catch (error) {
            this.addMessage('assistant', `Error: ${error.message}`, null, true);
        } finally {
            this.isGenerating = false;
            this.isStreaming = false;
            this.updateSendButton();
        }
    }

    /**
     * Process message before sending - override in subclasses for custom behavior
     */
    processMessage(message) {
        return message; // Default: no processing
    }

    /**
     * Handle streaming generation
     */
    async handleStreamingGeneration(formData) {
        this.isStreaming = true;
        
        // Create assistant message bubble for streaming
        const assistantMessageId = this.addMessage('assistant', '');
        this.currentStreamingMessage = document.getElementById(assistantMessageId);
        const messageContent = this.currentStreamingMessage.querySelector('.message-content');
        
        // Add cursor for streaming effect
        const cursor = document.createElement('span');
        cursor.className = 'streaming-cursor';
        messageContent.appendChild(cursor);

        try {
            const response = await fetch(this.config.generateEndpoint, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let fullResponse = '';

            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.substring(6));
                            
                            if (data.error) {
                                throw new Error(data.error);
                            }
                            
                            if (!data.finished && data.token) {
                                fullResponse += data.token;
                                
                                const textDiv = document.createElement('div');
                                textDiv.innerHTML = this.md.render(fullResponse);
                                
                                messageContent.innerHTML = '';
                                messageContent.appendChild(textDiv);
                                messageContent.appendChild(cursor);
                                
                                this.scrollToBottom();
                            }
                            
                            if (data.finished) {
                                cursor.remove();
                                this.addMessageActions(this.currentStreamingMessage, fullResponse);
                                return;
                            }
                        } catch (parseError) {
                            console.error('Failed to parse SSE data:', parseError);
                        }
                    }
                }
            }
        } catch (error) {
            cursor.remove();
            messageContent.innerHTML = `<div style="color: #dc2626;">Error: ${error.message}</div>`;
            throw error;
        }
    }

    /**
     * Stop generation
     */
    stopGeneration() {
        this.isGenerating = false;
        this.isStreaming = false;
        this.updateSendButton();
        
        if (this.currentStreamingMessage) {
            const cursor = this.currentStreamingMessage.querySelector('.streaming-cursor');
            if (cursor) cursor.remove();
            this.currentStreamingMessage = null;
        }
    }

    /**
     * Add message to chat
     */
    addMessage(role, content, attachments = null, isError = false) {
        const messageId = 'msg-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        messageDiv.id = messageId;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (isError) {
            contentDiv.style.background = '#ff4444';
        }

        // Add attachments if any
        if (attachments && attachments.size > 0) {
            const attachmentsDiv = document.createElement('div');
            attachmentsDiv.className = 'message-attachments';
            
            for (let [name, file] of attachments) {
                const attachment = document.createElement('div');
                attachment.className = 'attachment';
                
                if (file.type && file.type.startsWith('image/')) {
                    attachment.innerHTML = '🖼️ ' + name;
                } else if (file.type && file.type.startsWith('audio/')) {
                    attachment.innerHTML = '🎵 ' + name;
                } else {
                    attachment.innerHTML = '📎 ' + name;
                }
                
                attachmentsDiv.appendChild(attachment);
            }
            
            contentDiv.appendChild(attachmentsDiv);
        }

        // Add text content
        const textDiv = document.createElement('div');
        
        // Render markdown for assistant messages, plain text for user messages
        if (role === 'assistant' && !isError && content) {
            textDiv.innerHTML = this.md.render(content);
        } else if (content) {
            textDiv.textContent = content;
        }
        
        contentDiv.appendChild(textDiv);

        // Create message container
        const messageContainer = document.createElement('div');
        messageContainer.appendChild(contentDiv);

        messageDiv.appendChild(messageContainer);

        // Remove welcome message if it exists
        const welcomeMsg = this.chatHistory.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }

        this.chatHistory.appendChild(messageDiv);
        this.scrollToBottom();

        // Add action buttons for both user and assistant messages (if content exists)
        if (!isError && content) {
            this.addMessageActions(messageDiv, content);
        }

        return messageId;
    }

    /**
     * Add message action buttons
     */
    addMessageActions(messageDiv, content) {
        const messageContainer = messageDiv.querySelector('div');
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';

        // Copy button for all messages
        const copyBtn = document.createElement('button');
        copyBtn.className = 'action-button';
        copyBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>';
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(content);
            copyBtn.innerHTML = '✓';
            setTimeout(() => {
                copyBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>';
            }, 2000);
        };

        // TTS button (only for assistant messages and only if available)
        if (messageDiv.classList.contains('assistant') && this.ttsAvailable && this.config.enableTTS) {
            const ttsBtn = document.createElement('button');
            ttsBtn.className = 'action-button';
            ttsBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
            ttsBtn.onclick = () => this.handleTTS(content, ttsBtn);
            actionsDiv.appendChild(ttsBtn);
        }

        actionsDiv.appendChild(copyBtn);
        messageContainer.appendChild(actionsDiv);
    }

    /**
     * Handle text-to-speech
     */
    async handleTTS(text, button) {
        if (!this.ttsAvailable) {
            alert('Text-to-speech is not available');
            return;
        }

        // If this button is currently playing, stop the audio
        if (button.classList.contains('playing')) {
            if (this.currentAudio) {
                this.currentAudio.pause();
                this.currentAudio = null;
                button.classList.remove('playing');
                button.innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
            }
            return;
        }

        // Stop any other currently playing audio
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio = null;
            // Reset all play buttons
            document.querySelectorAll('.action-button.playing').forEach(btn => {
                btn.classList.remove('playing');
                btn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
            });
        }

        // Show loading state
        button.classList.add('loading');
        button.disabled = true;

        try {
            const formData = new FormData();
            formData.append('text', text);

            const response = await fetch(this.config.ttsEndpoint, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`TTS failed: ${response.status}`);
            }

            // Create audio from response
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);

            // Update button to playing state
            button.classList.remove('loading');
            button.classList.add('playing');
            button.innerHTML = '<svg viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>';
            
            this.currentAudio = audio;

            // Handle audio events
            audio.onended = () => {
                button.classList.remove('playing');
                button.innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
                URL.revokeObjectURL(audioUrl);
                this.currentAudio = null;
            };

            audio.onerror = () => {
                button.classList.remove('playing');
                button.innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
                URL.revokeObjectURL(audioUrl);
                this.currentAudio = null;
                alert('Failed to play audio');
            };

            // Play the audio
            await audio.play();

        } catch (error) {
            console.error('TTS error:', error);
            alert('Failed to generate speech');
            button.classList.remove('loading');
        } finally {
            button.disabled = false;
        }
    }

    /**
     * Check server status
     */
    async checkServerStatus() {
        try {
            const response = await fetch(this.config.healthEndpoint);
            if (response.ok) {
                const data = await response.json();
                this.ttsAvailable = data.tts_available;
                this.updateStatus(true);
            } else {
                this.updateStatus(false);
            }
        } catch (error) {
            this.updateStatus(false);
        }
    }

    /**
     * Update connection status
     */
    updateStatus(connected) {
        if (this.statusDot) {
            this.statusDot.classList.toggle('connected', connected);
        }
    }

    /**
     * Start periodic status polling
     */
    startStatusPolling() {
        setInterval(() => {
            if (!this.isGenerating) {
                this.checkServerStatus();
            }
        }, 30000); // Check every 30 seconds
    }

    /**
     * Clear attachments
     */
    clearAttachments() {
        this.attachedFiles.clear();
        this.renderAttachments();
    }

    /**
     * Scroll to bottom of chat
     */
    scrollToBottom() {
        window.scrollTo({
            top: document.body.scrollHeight,
            behavior: 'smooth'
        });
    }

    /**
     * Show error message
     */
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        this.chatHistory.appendChild(errorDiv);
        this.scrollToBottom();
        
        // Remove error after 5 seconds
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatTemplate;
} else if (typeof window !== 'undefined') {
    window.ChatTemplate = ChatTemplate;
}
