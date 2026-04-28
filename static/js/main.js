document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.querySelector('.send-btn');
    const chatForm = document.getElementById('gpt-input-form');
    const chatMessages = document.getElementById('chat-messages');
    const historyList = document.getElementById('history-list');
    const modelSelector = document.querySelector('.model-selector');
    
    // State
    let currentMode = 'chat';
    let isProcessing = false;

    // Initialize
    loadChatHistory();
    setupEventListeners();

    // Event Listeners
    function setupEventListeners() {
        // Input handling
        chatInput.addEventListener('input', () => {
            sendBtn.disabled = chatInput.value.trim().length === 0;
            if (chatInput.value.trim().length > 0) {
                sendBtn.style.backgroundColor = 'white';
                sendBtn.style.color = 'black';
            } else {
                sendBtn.style.backgroundColor = 'transparent';
                sendBtn.style.color = '#b4b4b4';
            }
        });

        // Form submission
        chatForm.addEventListener('submit', handleChatSubmit);

        // Sidebar Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const mode = item.dataset.mode;
                switchMode(mode);
            });
        });

        // Other forms
        const searchForm = document.getElementById('search-form');
        if(searchForm) searchForm.addEventListener('submit', handleSearchSubmit);
        
        const pdfForm = document.getElementById('pdf-summary-form');
        if(pdfForm) pdfForm.addEventListener('submit', handlePdfSubmit);
        
        const servicesForm = document.getElementById('services-search-form');
        if(servicesForm) servicesForm.addEventListener('submit', handleServicesSubmit);

        // PDF file picker
        const pdfDrop = document.getElementById('pdf-summary-drop');
        const pdfInput = document.getElementById('pdf-summary-file');
        if (pdfDrop && pdfInput) {
            pdfDrop.addEventListener('click', () => pdfInput.click());
            pdfInput.addEventListener('change', () => {
                if (pdfInput.files.length > 0) {
                    pdfDrop.querySelector('p').textContent = pdfInput.files[0].name;
                } else {
                    pdfDrop.querySelector('p').textContent = 'Upload PDF';
                }
            });
        }
    }

    // Switch Mode
    function switchMode(mode) {
        // Update active nav item
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
            if(item.dataset.mode === mode) item.classList.add('active');
        });

        // Update active view
        document.querySelectorAll('.mode-view').forEach(view => {
            view.classList.remove('active');
        });
        document.getElementById(`${mode}-mode`).classList.add('active');

        currentMode = mode;
        
        // Focus input if in chat mode
        if (mode === 'chat') {
            chatInput.focus();
        }
    }

    // Handle Chat Submit
    async function handleChatSubmit(e) {
        e.preventDefault();
        const message = chatInput.value.trim();
        if (!message || isProcessing) return;

        // Reset input
        chatInput.value = '';
        sendBtn.disabled = true;
        sendBtn.style.backgroundColor = 'transparent';
        sendBtn.style.color = '#b4b4b4';
        
        // Hide empty state if first message
        const emptyState = document.querySelector('.empty-state');
        if (emptyState) emptyState.style.display = 'none';

        // Add user message
        addChatMessage(message, 'user');
        
        isProcessing = true;

        // Get or create session ID for memory
        let sessionId = localStorage.getItem('smartbot_session_id');
        if (!sessionId) {
            sessionId = 'session_' + Math.random().toString(36).substring(2, 11);
            localStorage.setItem('smartbot_session_id', sessionId);
        }

        try {
            // Call API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: message,
                    session_id: sessionId
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `API Error: ${response.status}`);
            }
            const data = await response.json();
            
            // Add bot response
            addChatMessage(data.response, 'bot');
            
            // Save to history
            saveToHistory(message);

        } catch (error) {
            console.error(error);
            // DEBUG: Show actual error to user
            addChatMessage(`Error Details: ${error.message || error}`, 'bot');
        } finally {
            isProcessing = false;
        }
    }

    // Add Chat Message to UI
    function addChatMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${sender}`;
        
        // Avatar
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        
        if (sender === 'user') {
            avatarDiv.style.backgroundColor = '#7b1fa2'; // Purple context
            avatarDiv.textContent = 'S'; // Initials
            avatarDiv.style.display = 'flex';
            avatarDiv.style.alignItems = 'center';
            avatarDiv.style.justifyContent = 'center';
            avatarDiv.style.color = 'white';
            avatarDiv.style.fontSize = '12px';
            avatarDiv.style.fontWeight = 'bold';
        } else {
            avatarDiv.innerHTML = `
                <div style="background: #10a37f; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; border-radius: 2px;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                        <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/>
                    </svg>
                </div>
            `;
        }
        
        // Content
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text; // Ideally use markdown parser here
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    }

    // History Logic
    function loadChatHistory() {
        const history = JSON.parse(localStorage.getItem('smartbot_chat_history') || '[]');
        renderSidebarHistory(history);
    }

    function saveToHistory(firstMessage) {
        // Simple history implementation: just save the first user query as title
        const history = JSON.parse(localStorage.getItem('smartbot_chat_history') || '[]');
        const newItem = {
            id: Date.now(),
            title: firstMessage.substring(0, 30) + (firstMessage.length > 30 ? '...' : ''),
            date: new Date()
        };
        history.unshift(newItem);
        localStorage.setItem('smartbot_chat_history', JSON.stringify(history.slice(0, 50))); // Keep last 50
        renderSidebarHistory(history);
    }

    function renderSidebarHistory(history) {
        if (!historyList) return;
        historyList.innerHTML = '';
        history.forEach(item => {
            const div = document.createElement('div');
            div.className = 'history-item';
            div.textContent = item.title;
            div.onclick = () => loadHistoryItem(item);
            historyList.appendChild(div);
        });
    }

    function loadHistoryItem(item) {
        chatMessages.innerHTML = '';
        addChatMessage(item.title, 'user');
        addChatMessage("I recall this conversation. How can I help further?", 'bot');
    }

    // New Chat
    window.startNewChat = function() {
        chatMessages.innerHTML = `
            <div class="empty-state">
                <div class="gpt-logo-large">
                     <svg width="41" height="41" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="50" cy="50" r="50" fill="white"/>
                        <path d="M70 35C70 26.7157 63.2843 20 55 20H45C36.7157 20 30 26.7157 30 35V45C30 53.2843 36.7157 60 45 60H55C63.2843 60 70 66.7157 70 75V75" stroke="black" stroke-width="12" stroke-linecap="round"/>
                        <circle cx="45" cy="35" r="5" fill="black"/>
                    </svg>
                </div>
            </div>
        `;
    };

    // Mode specific handlers
    async function handleSearchSubmit(e) {
        e.preventDefault();
        const input = document.getElementById('search-input');
        const query = input.value;
        const resultsArea = document.getElementById('search-results');
        
        resultsArea.innerHTML = '<div class="loader">Searching the web...</div>';
        
        try {
            const res = await fetch('/api/search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query})
            });
            const data = await res.json();
            
            if (data.results && data.results.length > 0) {
                resultsArea.innerHTML = `
                    <div class="search-results-container">
                        ${data.results.map(item => `
                            <div class="search-card">
                                <a href="${item.link}" target="_blank" class="search-card-title">${item.title}</a>
                                <p class="search-card-snippet">${item.snippet}</p>
                            </div>
                        `).join('')}
                    </div>
                `;
            } else {
                resultsArea.innerHTML = '<div style="padding: 20px;">No results found.</div>';
            }
        } catch(e) {
            resultsArea.innerHTML = '<div style="color: #ff4a4a; padding: 20px;">Search failed. Please try again.</div>';
        }
    }

    async function handlePdfSubmit(e) {
        e.preventDefault();
        const fileInput = document.getElementById('pdf-summary-file');
        const resultsArea = document.getElementById('pdf-summary-results');
        
        if (!fileInput.files || fileInput.files.length === 0) {
            resultsArea.innerHTML = '<div style="color: #ff4a4a;">Please select a PDF file first.</div>';
            return;
        }

        resultsArea.innerHTML = '<div class="loader">Analyzing PDF...</div>';
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const res = await fetch('/api/pdf/summary', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (!res.ok) {
                throw new Error(data.error || 'Failed to analyze PDF');
            }
            
            resultsArea.innerHTML = `<div class="pdf-analysis-result" style="white-space: pre-wrap; line-height: 1.5; padding: 15px; background: #2a2a35; border-radius: 8px;">${data.response}</div>`;
        } catch (error) {
            resultsArea.innerHTML = `<div style="color: #ff4a4a;">Error: ${error.message}</div>`;
        }
    }

    async function handleServicesSubmit(e) {
        e.preventDefault();
    }

    // Login Modal Logic
    const loginModal = document.getElementById('login-modal');
    const signupModal = document.getElementById('signup-modal');
    const loginBtnTop = document.getElementById('login-btn-top');
    const signupBtnTop = document.getElementById('signup-btn-top');
    const closeModal = document.querySelector('.close-modal');
    const closeModalSignup = document.querySelector('.close-modal-signup');
    const loginForm = document.getElementById('login-form');
    const signupForm = document.getElementById('signup-form');
    const switchToSignup = document.getElementById('switch-to-signup');
    const switchToLogin = document.getElementById('switch-to-login');

    if (loginBtnTop && loginModal) {
        loginBtnTop.addEventListener('click', () => {
            loginModal.classList.add('active');
        });
    }

    if (signupBtnTop && signupModal) {
        signupBtnTop.addEventListener('click', () => {
            signupModal.classList.add('active');
        });
    }

    if (closeModal && loginModal) {
        closeModal.addEventListener('click', () => {
            loginModal.classList.remove('active');
        });
    }

    if (closeModalSignup && signupModal) {
        closeModalSignup.addEventListener('click', () => {
            signupModal.classList.remove('active');
        });
    }

    if (switchToSignup) {
        switchToSignup.addEventListener('click', (e) => {
            e.preventDefault();
            loginModal.classList.remove('active');
            signupModal.classList.add('active');
        });
    }

    if (switchToLogin) {
        switchToLogin.addEventListener('click', (e) => {
            e.preventDefault();
            signupModal.classList.remove('active');
            loginModal.classList.add('active');
        });
    }

    // Close on outside click
    window.addEventListener('click', (e) => {
        if (e.target === loginModal) loginModal.classList.remove('active');
        if (e.target === signupModal) signupModal.classList.remove('active');
    });

    if (loginForm) {
        loginForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const email = document.getElementById('login-email').value;
            alert('Welcome back, ' + email);
            loginModal.classList.remove('active');
        });
    }

    if (signupForm) {
        signupForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const name = document.getElementById('signup-name').value;
            alert('Account created for ' + name + '! You can now log in.');
            signupModal.classList.remove('active');
            loginModal.classList.add('active');
        });
    }

    // Suggestion Chips
    document.querySelectorAll('.suggestion-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            chatInput.value = chip.textContent;
            chatForm.dispatchEvent(new Event('submit'));
        });
    });
});
