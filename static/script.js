class GemmaApp {
    constructor() {
        this.apiBase = window.location.origin;
        this.statusElement = document.getElementById('status');
        this.responseElement = document.getElementById('response');
        this.loadingElement = document.getElementById('loading');
        
        this.init();
    }
    
    init() {
        this.setupTabs();
        this.setupForms();
        this.checkHealth();
        
        // Check health every 30 seconds
        setInterval(() => this.checkHealth(), 30000);
    }
    
    setupTabs() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabId = button.getAttribute('data-tab');
                
                // Update active tab button
                tabButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                // Update active tab content
                tabContents.forEach(content => {
                    content.classList.remove('active');
                    if (content.id === `${tabId}-tab`) {
                        content.classList.add('active');
                    }
                });
            });
        });
    }
    
    setupForms() {
        // Text-only form
        const textForm = document.getElementById('text-form');
        textForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleTextSubmit(new FormData(textForm));
        });
        
        // Multimodal form
        const multimodalForm = document.getElementById('multimodal-form');
        multimodalForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleMultimodalSubmit(new FormData(multimodalForm));
        });
    }
    
    async checkHealth() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy' && data.model_loaded) {
                this.updateStatus('Model loaded and ready', 'connected');
            } else {
                this.updateStatus('Model loading...', 'loading');
            }
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateStatus('Connection failed', 'disconnected');
        }
    }
    
    updateStatus(message, type) {
        this.statusElement.textContent = message;
        this.statusElement.className = `status ${type}`;
    }
    
    showLoading() {
        this.loadingElement.classList.remove('hidden');
        this.responseElement.style.display = 'none';
    }
    
    hideLoading() {
        this.loadingElement.classList.add('hidden');
        this.responseElement.style.display = 'block';
    }
    
    displayResponse(response, isError = false) {
        this.hideLoading();
        
        if (isError) {
            this.responseElement.innerHTML = `<p style="color: #f44336; font-family: inherit;"><strong>Error:</strong> ${response}</p>`;
        } else {
            this.responseElement.innerHTML = `<pre style="font-family: inherit; white-space: pre-wrap;">${response}</pre>`;
        }
    }
    
    async handleTextSubmit(formData) {
        const submitButton = document.querySelector('#text-form button[type="submit"]');
        
        try {
            submitButton.disabled = true;
            this.showLoading();
            
            const response = await fetch(`${this.apiBase}/api/generate/text`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.displayResponse(data.response);
            
        } catch (error) {
            console.error('Text generation failed:', error);
            this.displayResponse(`Failed to generate response: ${error.message}`, true);
        } finally {
            submitButton.disabled = false;
        }
    }
    
    async handleMultimodalSubmit(formData) {
        const submitButton = document.querySelector('#multimodal-form button[type="submit"]');
        
        try {
            submitButton.disabled = true;
            this.showLoading();
            
            const response = await fetch(`${this.apiBase}/api/generate/multimodal`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.displayResponse(data.response);
            
        } catch (error) {
            console.error('Multimodal generation failed:', error);
            this.displayResponse(`Failed to generate response: ${error.message}`, true);
        } finally {
            submitButton.disabled = false;
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new GemmaApp();
});
