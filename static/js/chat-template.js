/**
 * ChatTemplate - Complete reusable chat interface template
 * Provides HTML, CSS, and all functionality that can be extended for specific use cases
 */
class ChatTemplate {
    constructor(config = {}) {
        // Default configuration
        this.config = {
            // API endpoints
            generateEndpoint: '/api/generate/stream',
            ttsEndpoint: '/api/tts',
            healthEndpoint: '/health',
            
            // Behavior
            autoFocus: true,
            enableTTS: true,
            enableAttachments: true,
            enableAudio: true,
            enableSettings: true,
            maxTokens: 32768,
            
            // Customization
            modeText: 'CHAT',
            backUrl: '/',
            welcomeTitle: 'Chat with frende',
            welcomeSubtitle: 'Your private on-device AI companion',
            placeholder: 'Type your message...',
            
            // Callbacks
            onSettingsClick: null, // Override this for settings functionality
            
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
     * Get the complete CSS for the chat interface
     */
    getCSS() {
        return `
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", Inter, system-ui, sans-serif;
                background: #f8f9fa;
                color: #1a1a1a;
                height: 100vh;
                overflow: hidden;
                font-size: 16px;
                line-height: 1.5;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }

            .chat-container {
                display: flex;
                flex-direction: column;
                height: 100vh;
                width: 100%;
                max-width: none;
                margin: 0;
                background: #f8f9fa;
            }

            .content-wrapper {
                max-width: 1000px;
                margin: 0 auto;
                width: 100%;
                display: flex;
                flex-direction: column;
                height: 100vh;
            }

            .chat-history {
                flex: 1;
                overflow-y: auto;
                padding: 24px;
                display: flex;
                flex-direction: column;
                gap: 20px;
                background: #f8f9fa;
                max-width: 1000px;
                margin: 0 auto;
                width: 100%;
            }

            .input-section {
                flex-shrink: 0;
                width: 100%;
                background: #f8f9fa;
                padding: 20px 24px;
                max-width: 1000px;
                margin: 0 auto;
            }

            .status-bar {
                flex-shrink: 0;
                background: #f8f9fa;
                padding: 4px 150px 4px 120px;
                font-size: 14px;
                color: #666666;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-weight: 500;
            }

            .status-indicator {
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #ff4444;
                animation: pulse 2s infinite;
            }

            .status-dot.connected {
                background: #22c55e;
                animation: none;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }

            .back-button {
                background: none;
                border: none;
                cursor: pointer;
                padding: 4px;
                border-radius: 4px;
                display: flex;
                align-items: center;
                gap: 4px;
                color: #666666;
                font-size: 13px;
                transition: background 200ms ease;
            }

            .back-button:hover {
                background: #e5e5e5;
            }

            .back-button svg {
                width: 16px;
                height: 16px;
                fill: currentColor;
            }

            .mode-section {
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .settings-button {
                background: none;
                border: none;
                cursor: pointer;
                padding: 4px;
                border-radius: 4px;
                display: flex;
                align-items: center;
                color: #666666;
                transition: background 200ms ease;
            }

            .settings-button:hover {
                background: #e5e5e5;
            }

            .settings-button svg {
                width: 18px;
                height: 18px;
                fill: currentColor;
            }

            .mode-indicator {
                background: #e8f4ff;
                color: #0066cc;
                padding: 4px 8px;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 500;
                text-transform: uppercase;
            }

            .message {
                display: flex;
                gap: 12px;
                align-items: flex-start;
                max-width: 90%;
                position: relative;
            }

            .message.user {
                align-self: flex-end;
                flex-direction: row-reverse;
            }

            .message.assistant {
                align-self: flex-start;
            }

            .message-content {
                background: #f5f5f5;
                padding: 16px 20px;
                border-radius: 18px;
                line-height: 1.5;
                word-wrap: break-word;
                color: #1a1a1a;
                font-size: 16px;
                white-space: pre-wrap;
                position: relative;
                min-width: 200px;
            }

            .message.user .message-content {
                background: #0066cc;
                color: #ffffff;
            }

            .message.assistant .message-content {
                white-space: normal;
            }

            .message-content h1, .message-content h2, .message-content h3,
            .message-content h4, .message-content h5, .message-content h6 {
                margin: 16px 0 8px 0;
                font-weight: 600;
            }

            .message-content h1 { font-size: 1.5em; }
            .message-content h2 { font-size: 1.3em; }
            .message-content h3 { font-size: 1.1em; }

            .message-content p {
                margin: 8px 0;
            }

            .message-content ul, .message-content ol {
                margin: 8px 0;
                padding-left: 20px;
            }

            .message-content li {
                margin: 4px 0;
            }

            .message-content code {
                background: rgba(0, 0, 0, 0.08);
                padding: 2px 6px;
                border-radius: 4px;
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                font-size: 0.9em;
            }

            .message-content pre {
                background: rgba(0, 0, 0, 0.08);
                padding: 12px;
                border-radius: 8px;
                overflow-x: auto;
                margin: 12px 0;
            }

            .message-content pre code {
                background: none;
                padding: 0;
            }

            .message-content blockquote {
                border-left: 4px solid #0066cc;
                padding-left: 16px;
                margin: 12px 0;
                color: #666666;
                font-style: italic;
            }

            .message-content table {
                border-collapse: collapse;
                width: 100%;
                margin: 12px 0;
            }

            .message-content th, .message-content td {
                border: 1px solid #e5e5e5;
                padding: 8px 12px;
                text-align: left;
            }

            .message-content th {
                background: #f8f9fa;
                font-weight: 600;
            }

            .message-content strong {
                font-weight: 600;
            }

            .message-content em {
                font-style: italic;
            }

            .streaming-cursor {
                display: inline-block;
                width: 2px;
                height: 1.2em;
                background: #0066cc;
                margin-left: 2px;
                animation: blink 1s infinite;
            }

            @keyframes blink {
                0%, 50% { opacity: 1; }
                51%, 100% { opacity: 0; }
            }

            .message-actions {
                display: flex;
                align-items: center;
                gap: 4px;
                margin-top: 8px;
                opacity: 0;
                transition: opacity 200ms ease;
            }

            .message:hover .message-actions {
                opacity: 1;
            }

            .message.user .message-actions {
                justify-content: flex-end;
            }

            .message.assistant .message-actions {
                justify-content: flex-start;
            }

            .action-button {
                background: none;
                border: none;
                cursor: pointer;
                padding: 6px;
                border-radius: 6px;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 200ms ease;
                font-size: 14px;
            }

            .action-button:hover {
                background: rgba(0, 0, 0, 0.1);
            }

            .action-button:disabled {
                opacity: 0.3;
                cursor: not-allowed;
            }

            .action-button:disabled:hover {
                background: none;
            }

            .action-button svg {
                width: 16px;
                height: 16px;
                fill: #666666;
            }

            .action-button.playing svg {
                fill: #0066cc;
            }

            .action-button.loading {
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }

            .message-attachments {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 8px;
            }

            .attachment {
                background: #e5e5e5;
                padding: 6px 12px;
                border-radius: 12px;
                font-size: 12px;
                color: #666666;
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .attachments {
                display: none;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 16px;
            }

            .attachments.active {
                display: flex;
            }

            .attachment-pill {
                display: flex;
                align-items: center;
                background: #f5f5f5;
                padding: 8px 12px;
                border-radius: 16px;
                font-size: 14px;
                color: #666666;
            }

            .attachment-pill span {
                max-width: 150px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .remove-attachment {
                background: none;
                border: none;
                cursor: pointer;
                padding: 0 0 0 8px;
                color: #999999;
                font-size: 16px;
                line-height: 1;
            }

            .remove-attachment:hover {
                color: #666666;
            }

            .input-container {
                position: relative;
                width: 80%;
                margin: 0 auto;
            }

            .input-field {
                width: 100%;
                min-height: 56px;
                max-height: 120px;
                padding: 16px 120px 16px 56px;
                font-size: 16px;
                font-family: inherit;
                border: 2px solid #e5e5e5;
                border-radius: 28px;
                resize: none;
                background: #ffffff;
                transition: all 200ms ease;
                line-height: 1.4;
            }

            .input-field:focus {
                outline: none;
                border-color: #0066cc;
                border-width: 2px;
            }

            .input-field::placeholder {
                color: #999999;
            }

            .input-controls {
                position: absolute;
                right: 8px;
                top: 50%;
                transform: translateY(-50%);
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .control-button {
                width: 40px;
                height: 40px;
                background: none;
                border: none;
                border-radius: 20px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 200ms ease;
                font-size: 16px;
            }

            .control-button:hover {
                background: #f5f5f5;
            }

            .control-button:disabled {
                opacity: 0.3;
                cursor: not-allowed;
            }

            .control-button:disabled:hover {
                background: none;
            }

            .control-button svg {
                width: 20px;
                height: 20px;
                fill: #666666;
            }

            .send-button {
                background: #0066cc;
            }

            .send-button:hover:not(:disabled) {
                background: #0056b3;
            }

            .send-button svg {
                fill: #ffffff;
            }

            .send-button:disabled {
                background: #e5e5e5;
            }

            .send-button:disabled svg {
                fill: #999999;
            }

            .attachment-button {
                position: absolute;
                left: 8px;
                top: 50%;
                transform: translateY(-50%);
                width: 40px;
                height: 40px;
                background: none;
                border: none;
                border-radius: 20px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background 200ms ease;
                font-size: 16px;
            }

            .attachment-button:hover {
                background: #f5f5f5;
            }

            .attachment-button svg {
                width: 20px;
                height: 20px;
                fill: #666666;
            }

            .recording {
                background: #ff4444 !important;
            }

            .recording svg {
                fill: #ffffff !important;
            }

            #file-input {
                display: none;
            }

            .loading {
                display: flex;
                align-items: center;
                gap: 8px;
                color: #666666;
                font-size: 14px;
            }

            .loading-dots {
                display: flex;
                gap: 4px;
            }

            .loading-dot {
                width: 4px;
                height: 4px;
                border-radius: 50%;
                background: #666666;
                animation: loading 1.4s infinite ease-in-out;
            }

            .loading-dot:nth-child(1) { animation-delay: -0.32s; }
            .loading-dot:nth-child(2) { animation-delay: -0.16s; }

            @keyframes loading {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
            }

            .welcome-message {
                text-align: center;
                color: #666666;
                padding: 60px 20px;
                font-size: 16px;
                line-height: 1.6;
                font-weight: 400;
            }

            .welcome-message p:first-child {
                font-size: 18px;
                font-weight: 500;
                color: #1a1a1a;
                margin-bottom: 8px;
            }

            .error-message {
                background: #fee2e2;
                color: #dc2626;
                border: 1px solid #fecaca;
                padding: 12px 16px;
                border-radius: 12px;
                margin: 16px 24px;
                font-size: 14px;
            }

            .chat-history::-webkit-scrollbar {
                width: 8px;
            }

            .chat-history::-webkit-scrollbar-track {
                background: #f1f1f1;
            }

            .chat-history::-webkit-scrollbar-thumb {
                background: #c1c1c1;
                border-radius: 4px;
            }

            .chat-history::-webkit-scrollbar-thumb:hover {
                background: #a8a8a8;
            }

            @media (max-width: 768px) {
                .chat-container {
                    max-width: 100%;
                }
                
                .message {
                    max-width: 95%;
                }
                
                .input-field {
                    padding: 14px 100px 14px 48px;
                    min-height: 48px;
                }
                
                .attachment-button,
                .control-button {
                    width: 36px;
                    height: 36px;
                }
                
                .control-button svg,
                .attachment-button svg {
                    width: 18px;
                    height: 18px;
                }

                .status-bar {
                    padding: 8px 24px;
                }
            }
        `;
    }

    /**
     * Get the complete HTML structure
     */
    getHTML() {
        return `
            <div class="chat-container">
                <div class="content-wrapper">
                    <div class="status-bar">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <button class="back-button" id="back-btn">
                                <svg viewBox="0 0 24 24">
                                    <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/>
                                </svg>
                                back
                            </button>
                        </div>
                        <div class="mode-section">
                            ${this.config.enableSettings ? `
                                <button class="settings-button" id="settings-btn">
                                    <svg viewBox="0 0 24 24">
                                        <path d="M19.14,12.94c0.04-0.3,0.06-0.61,0.06-0.94c0-0.32-0.02-0.64-0.07-0.94l2.03-1.58c0.18-0.14,0.23-0.41,0.12-0.61 l-1.92-3.32c-0.12-0.22-0.37-0.29-0.59-0.22l-2.39,0.96c-0.5-0.38-1.03-0.7-1.62-0.94L14.4,2.81c-0.04-0.24-0.24-0.41-0.48-0.41 h-3.84c-0.24,0-0.43,0.17-0.47,0.41L9.25,5.35C8.66,5.59,8.12,5.92,7.63,6.29L5.24,5.33c-0.22-0.08-0.47,0-0.59,0.22L2.74,8.87 C2.62,9.08,2.66,9.34,2.86,9.48l2.03,1.58C4.84,11.36,4.82,11.69,4.82,12s0.02,0.64,0.07,0.94l-2.03,1.58 c-0.18,0.14-0.23,0.41-0.12,0.61l1.92,3.32c0.12,0.22,0.37,0.29,0.59,0.22l2.39-0.96c0.5,0.38,1.03,0.7,1.62,0.94l0.36,2.54 c0.05,0.24,0.24,0.41,0.48,0.41h3.84c0.24,0,0.44-0.17,0.47-0.41l0.36-2.54c0.59-0.24,1.13-0.56,1.62-0.94l2.39,0.96 c0.22,0.08,0.47,0,0.59-0.22l1.92-3.32c0.12-0.22,0.07-0.47-0.12-0.61L19.14,12.94z M12,15.6c-1.98,0-3.6-1.62-3.6-3.6 s1.62-3.6,3.6-3.6s3.6,1.62,3.6,3.6S13.98,15.6,12,15.6z"/>
                                    </svg>
                                </button>
                            ` : ''}
                            <span id="mode-indicator" class="mode-indicator">${this.config.modeText}</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <div class="status-indicator">
                                <div class="status-dot" id="status-dot"></div>
                            </div>
                        </div>
                    </div>

                    <div class="chat-history" id="chat-history">
                        <div class="welcome-message">
                            <p>${this.config.welcomeTitle}</p>
                            <p>${this.config.welcomeSubtitle}</p>
                        </div>
                    </div>

                    <div class="input-section">
                        <div class="attachments" id="attachment-display"></div>
                        
                        <div class="input-container">
                            ${this.config.enableAttachments ? `
                                <button class="attachment-button" id="attachment-btn">
                                    <svg viewBox="0 0 24 24">
                                        <path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v11.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z"/>
                                    </svg>
                                </button>
                            ` : ''}
                            
                            <textarea
                                id="message-input"
                                class="input-field"
                                placeholder="${this.config.placeholder}"
                                rows="1"
                            ></textarea>
                            
                            <div class="input-controls">
                                ${this.config.enableAudio ? `
                                    <button class="control-button" id="mic-btn">
                                        <svg viewBox="0 0 24 24">
                                            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.49 6-3.31 6-6.72h-1.7z"/>
                                        </svg>
                                    </button>
                                ` : ''}
                                
                                <button class="control-button send-button" id="send-btn" disabled>
                                    <svg viewBox="0 0 24 24">
                                        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                                    </svg>
                                </button>
                            </div>
                            
                            ${this.config.enableAttachments ? `
                                <input type="file" id="file-input" multiple accept="image/*,audio/*" hidden>
                            ` : ''}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Initialize the complete chat interface
     */
    async initialize() {
        // Inject CSS
        const style = document.createElement('style');
        style.textContent = this.getCSS();
        document.head.appendChild(style);

        // Inject HTML
        document.body.innerHTML = this.getHTML();

        // Initialize elements and functionality
        this.initializeElements();
        this.initializeEventListeners();
        this.setupStatusBar();
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
        this.chatHistory = document.querySelector('#chat-history');
        this.messageInput = document.querySelector('#message-input');
        this.sendBtn = document.querySelector('#send-btn');
        this.attachmentBtn = document.querySelector('#attachment-btn');
        this.micBtn = document.querySelector('#mic-btn');
        this.fileInput = document.querySelector('#file-input');
        this.statusDot = document.querySelector('#status-dot');
        this.backBtn = document.querySelector('#back-btn');
        this.modeIndicator = document.querySelector('#mode-indicator');
        this.settingsBtn = document.querySelector('#settings-btn');
        
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

        // Settings button
        if (this.settingsBtn) {
            this.settingsBtn.addEventListener('click', () => {
                if (this.config.onSettingsClick) {
                    this.config.onSettingsClick();
                } else {
                    // Default settings behavior
                    console.log('Settings clicked - override onSettingsClick in config');
                }
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
     * Set up status bar
     */
    setupStatusBar() {
        // Status bar is already set up in HTML template
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
        this.chatHistory.scrollTo({
            top: this.chatHistory.scrollHeight,
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

    /**
     * Handle transition data from other pages
     */
    handleTransitionData() {
        const transitionData = sessionStorage.getItem('chatTransition');
        if (transitionData) {
            try {
                const data = JSON.parse(transitionData);
                
                // Handle message if present
                if (data.message) {
                    this.messageInput.value = data.message;
                    this.autoResizeTextarea();
                }
                
                // Handle files if present
                if (data.files && data.files.length > 0) {
                    data.files.forEach(([name, fileData]) => {
                        fetch(fileData.data)
                            .then(res => res.blob())
                            .then(blob => {
                                const file = new File([blob], fileData.name, { type: fileData.type });
                                this.attachedFiles.set(name, file);
                                this.renderAttachments();
                                this.updateSendButton();
                            });
                    });
                }
                
                // Auto-send if from direct mode
                if (data.mode === 'direct' && (data.message || (data.files && data.files.length > 0))) {
                    setTimeout(() => {
                        if (this.messageInput.value.trim() || this.attachedFiles.size > 0) {
                            this.handleSubmit();
                        }
                    }, 500);
                }
                
                // Clear transition data
                sessionStorage.removeItem('chatTransition');
                
            } catch (e) {
                console.error('Error parsing transition data:', e);
                sessionStorage.removeItem('chatTransition');
            }
        }
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatTemplate;
} else if (typeof window !== 'undefined') {
    window.ChatTemplate = ChatTemplate;
}
