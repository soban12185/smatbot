/* ===================================================
   SmartBot – main.js
   =================================================== */

document.addEventListener('DOMContentLoaded', () => {

    // ── DOM refs ──────────────────────────────────────
    const chatForm        = document.getElementById('chat-form');
    const chatInput       = document.getElementById('chat-input');
    const sendBtn         = document.getElementById('send-btn');
    const chatArea        = document.getElementById('chat-area');
    const messagesContainer = document.getElementById('messages-container');
    const welcomeState    = document.getElementById('welcome-state');
    const historyList     = document.getElementById('history-list');

    // Search
    const searchForm    = document.getElementById('search-form');
    const searchInput   = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');

    // PDF
    const pdfForm       = document.getElementById('pdf-form');
    const pdfDropZone   = document.getElementById('drop-zone');
    const pdfFileInput  = document.getElementById('pdf-file-input');
    const pdfSubmitBtn  = document.getElementById('pdf-submit-btn');
    const pdfResults    = document.getElementById('pdf-results');

    // Services
    const servicesForm    = document.getElementById('services-form');
    const servicesInput   = document.getElementById('services-input');
    const servicesResults = document.getElementById('services-results');

    // Auth
    const btnLogin  = document.getElementById('btn-login');
    const btnSignup = document.getElementById('btn-signup');
    const modalLogin  = document.getElementById('modal-login');
    const modalSignup = document.getElementById('modal-signup');
    const loginForm   = document.getElementById('login-form');
    const signupForm  = document.getElementById('signup-form');
    const goSignup    = document.getElementById('go-signup');
    const goLogin     = document.getElementById('go-login');

    // Mobile sidebar
    const sidebar        = document.getElementById('sidebar');
    const menuToggle     = document.getElementById('menu-toggle');
    const sidebarOverlay = document.getElementById('sidebar-overlay');

    // ── DATA ──────────────────────────────────────────
    const MOCK_SERVICES = [
        { title: 'Premium Catering', desc: 'Expert food service for any event size.', price: '$50/plate', category: 'catering' },
        { title: 'Royal Decoration', desc: 'Elegant decoration for weddings & parties.', price: '$200/event', category: 'decoration' },
        { title: 'StarBand Live Music', desc: 'Professional live music entertainment.', price: '$500/show', category: 'entertainment' },
        { title: 'FastFix Plumbing', desc: '24/7 emergency plumbing and pipe repair.', price: '$80/hr', category: 'plumbing' },
        { title: 'CloudIT Consulting', desc: 'Cloud migration & cybersecurity experts.', price: 'Custom quote', category: 'it' },
        { title: 'ProPhoto Studio', desc: 'Professional photography for all occasions.', price: '$300/session', category: 'photography' },
    ];

    // State
    let isProcessing = false;
    let sessionId = localStorage.getItem('sb_session') || generateId();
    localStorage.setItem('sb_session', sessionId);

    // ─────────────────────────────────────────────────
    //  INIT
    // ─────────────────────────────────────────────────
    loadHistory();
    setupNavigation();
    setupChat();
    setupSearch();
    setupPDF();
    setupServices();
    setupAuth();
    setupMobile();

    // ─────────────────────────────────────────────────
    //  NAVIGATION  (sidebar tabs)
    // ─────────────────────────────────────────────────
    function setupNavigation() {
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const mode = btn.dataset.mode;
                switchView(mode);
                // Close sidebar on mobile after tap
                if (window.innerWidth <= 768) closeSidebar();
            });
        });
    }

    function switchView(mode) {
        // Update nav
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
        // Update view
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        const target = document.getElementById(`view-${mode}`);
        if (target) target.classList.add('active');
    }

    // ─────────────────────────────────────────────────
    //  CHAT
    // ─────────────────────────────────────────────────
    function setupChat() {
        // Auto-grow textarea
        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + 'px';
            sendBtn.disabled = chatInput.value.trim() === '';
        });

        // Enter to send (Shift+Enter = new line)
        chatInput.addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!sendBtn.disabled) chatForm.requestSubmit();
            }
        });

        // Form submit
        chatForm.addEventListener('submit', handleChatSubmit);

        // Suggestion chips
        document.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const prompt = chip.dataset.prompt || chip.textContent.trim();
                chatInput.value = prompt;
                chatInput.dispatchEvent(new Event('input'));
                chatForm.requestSubmit();
            });
        });
    }

    async function handleChatSubmit(e) {
        e.preventDefault();
        const message = chatInput.value.trim();
        if (!message || isProcessing) return;

        // Hide welcome state
        if (welcomeState) welcomeState.style.display = 'none';

        // Reset input
        chatInput.value = '';
        chatInput.style.height = 'auto';
        sendBtn.disabled = true;

        // Render user bubble
        appendUserMessage(message);

        // Show typing
        const typingEl = appendTypingIndicator();
        isProcessing = true;

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: message, session_id: sessionId })
            });

            const data = await res.json();
            typingEl.remove();

            if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
            appendBotMessage(data.response);
            saveHistory(message);

        } catch (err) {
            typingEl.remove();
            appendBotMessage(`⚠️ ${err.message || 'Something went wrong. Please try again.'}`);
        } finally {
            isProcessing = false;
        }
    }

    // ── Message renderers ──────────────────────────
    function appendUserMessage(text) {
        const row = document.createElement('div');
        row.className = 'msg-row user';
        row.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
        messagesContainer.appendChild(row);
        scrollToBottom();
    }

    function appendBotMessage(text) {
        const row = document.createElement('div');
        row.className = 'msg-row bot';
        row.innerHTML = `
            <div class="bot-logo">
                <svg width="18" height="18" viewBox="0 0 100 100" fill="none">
                    <circle cx="50" cy="50" r="50" fill="white"/>
                    <path d="M70 35C70 26.7 63.3 20 55 20H45C36.7 20 30 26.7 30 35V45C30 53.3 36.7 60 45 60H55C63.3 60 70 66.7 70 75" stroke="black" stroke-width="12" stroke-linecap="round"/>
                    <circle cx="45" cy="35" r="5" fill="black"/>
                </svg>
            </div>
            <div class="msg-body">
                <div class="msg-text">${formatMarkdown(text)}</div>
                <div class="msg-actions">
                    <button class="msg-action-btn" title="Copy" onclick="copyText(this, ${JSON.stringify(text)})">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                    </button>
                    <button class="msg-action-btn" title="Good response">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.3a2 2 0 0 0 2-1.7l1.4-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
                    </button>
                    <button class="msg-action-btn" title="Bad response">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.7a2 2 0 0 0-2 1.7l-1.4 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h2.3A2 2 0 0 1 21.4 4v7a2 2 0 0 1-2 2H17"/></svg>
                    </button>
                </div>
            </div>
        `;
        messagesContainer.appendChild(row);
        scrollToBottom();
    }

    function appendTypingIndicator() {
        const row = document.createElement('div');
        row.className = 'typing-row';
        row.innerHTML = `
            <div class="bot-logo">
                <svg width="18" height="18" viewBox="0 0 100 100" fill="none">
                    <circle cx="50" cy="50" r="50" fill="white"/>
                    <path d="M70 35C70 26.7 63.3 20 55 20H45C36.7 20 30 26.7 30 35V45C30 53.3 36.7 60 45 60H55C63.3 60 70 66.7 70 75" stroke="black" stroke-width="12" stroke-linecap="round"/>
                    <circle cx="45" cy="35" r="5" fill="black"/>
                </svg>
            </div>
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        messagesContainer.appendChild(row);
        scrollToBottom();
        return row;
    }

    function scrollToBottom() {
        chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
    }

    // ─────────────────────────────────────────────────
    //  WEB SEARCH
    // ─────────────────────────────────────────────────
    function setupSearch() {
        if (!searchForm) return;
        searchForm.addEventListener('submit', async e => {
            e.preventDefault();
            const q = searchInput.value.trim();
            if (!q) return;

            searchResults.innerHTML = `<div class="loader-text">Searching the web…</div>`;

            try {
                const res  = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: q })
                });
                const data = await res.json();

                if (data.results && data.results.length) {
                    searchResults.innerHTML = data.results.map(r => `
                        <div class="result-card">
                            <a href="${r.link || '#'}" target="_blank" class="result-card-title">${r.title || 'Untitled'}</a>
                            <div class="result-card-url">${r.link || ''}</div>
                            <p class="result-card-snippet">${r.snippet || ''}</p>
                        </div>
                    `).join('');
                } else {
                    searchResults.innerHTML = `<div class="loader-text">No results found for "${escapeHtml(q)}".</div>`;
                }
            } catch {
                searchResults.innerHTML = `<div class="loader-text">Search failed. Please try again.</div>`;
            }
        });
    }

    // ─────────────────────────────────────────────────
    //  PDF UPLOAD
    // ─────────────────────────────────────────────────
    function setupPDF() {
        if (!pdfDropZone) return;

        // Click to open file picker
        pdfDropZone.addEventListener('click', () => pdfFileInput.click());

        // File selected
        pdfFileInput.addEventListener('change', () => {
            if (pdfFileInput.files[0]) {
                pdfDropZone.querySelector('.drop-title').textContent = pdfFileInput.files[0].name;
                pdfSubmitBtn.disabled = false;
            }
        });

        // Drag & drop
        pdfDropZone.addEventListener('dragover', e => { e.preventDefault(); pdfDropZone.classList.add('dragover'); });
        pdfDropZone.addEventListener('dragleave', () => pdfDropZone.classList.remove('dragover'));
        pdfDropZone.addEventListener('drop', e => {
            e.preventDefault();
            pdfDropZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type === 'application/pdf') {
                const dt = new DataTransfer();
                dt.items.add(file);
                pdfFileInput.files = dt.files;
                pdfDropZone.querySelector('.drop-title').textContent = file.name;
                pdfSubmitBtn.disabled = false;
            }
        });

        // Submit
        pdfForm.addEventListener('submit', async e => {
            e.preventDefault();
            if (!pdfFileInput.files[0]) return;

            pdfResults.innerHTML = `<div class="loader-text">Analyzing document…</div>`;

            const formData = new FormData();
            formData.append('pdf', pdfFileInput.files[0]);

            try {
                const res  = await fetch('/api/pdf/summarize', { method: 'POST', body: formData });
                const data = await res.json();
                pdfResults.innerHTML = `
                    <div class="result-card">
                        <span class="result-card-title">📄 Document Summary</span>
                        <p class="result-card-snippet" style="white-space:pre-wrap;">${data.summary || data.response || 'No summary returned.'}</p>
                    </div>
                `;
            } catch {
                pdfResults.innerHTML = `<div class="loader-text">Failed to analyze PDF. Please try again.</div>`;
            }
        });
    }

    // ─────────────────────────────────────────────────
    //  SERVICES
    // ─────────────────────────────────────────────────
    function setupServices() {
        if (!servicesForm) return;
        
        servicesForm.addEventListener('submit', e => {
            e.preventDefault();
            const q = servicesInput.value.trim().toLowerCase();
            if (q === '') {
                servicesResults.innerHTML = '';
                return;
            }
            const filtered = MOCK_SERVICES.filter(s =>
                s.title.toLowerCase().includes(q) ||
                s.desc.toLowerCase().includes(q) ||
                s.category.includes(q)
            );
            renderServiceCards(filtered, q);
        });
        
        // Also search as you type for better UX
        servicesInput.addEventListener('input', () => {
            const q = servicesInput.value.trim().toLowerCase();
            if (q === '') {
                servicesResults.innerHTML = '';
                return;
            }
            const filtered = MOCK_SERVICES.filter(s =>
                s.title.toLowerCase().includes(q) ||
                s.category.includes(q)
            );
            renderServiceCards(filtered, q);
        });
    }

    function renderServiceCards(services, query = '') {
        if (!services.length) {
            servicesResults.innerHTML = `<div class="loader-text">No services found matching "${escapeHtml(query)}".</div>`;
            return;
        }
        servicesResults.innerHTML = services.map(s => `
            <div class="result-card">
                <span class="result-card-title">${s.title}</span>
                <p class="result-card-snippet">${s.desc} — <strong>${s.price}</strong></p>
                <div class="result-card-actions">
                    <button class="btn-card">Book Now</button>
                </div>
            </div>
        `).join('');
    }

    // ─────────────────────────────────────────────────
    //  AUTH MODALS
    // ─────────────────────────────────────────────────
    function setupAuth() {
        btnLogin?.addEventListener('click',  () => openModal('modal-login'));
        btnSignup?.addEventListener('click', () => openModal('modal-signup'));

        // Close buttons
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', () => closeModal(btn.dataset.close));
        });

        // Close on overlay click
        document.querySelectorAll('.modal-overlay').forEach(overlay => {
            overlay.addEventListener('click', e => {
                if (e.target === overlay) closeModal(overlay.id);
            });
        });

        // Switch links
        goSignup?.addEventListener('click', e => { e.preventDefault(); closeModal('modal-login'); openModal('modal-signup'); });
        goLogin?.addEventListener('click',  e => { e.preventDefault(); closeModal('modal-signup'); openModal('modal-login'); });

        // Form submissions (stub)
        loginForm?.addEventListener('submit', e => {
            e.preventDefault();
            alert(`Welcome back, ${document.getElementById('login-email').value}!`);
            closeModal('modal-login');
        });
        signupForm?.addEventListener('submit', e => {
            e.preventDefault();
            alert(`Account created for ${document.getElementById('signup-name').value}! Please log in.`);
            closeModal('modal-signup');
            openModal('modal-login');
        });
    }

    function openModal(id)  { document.getElementById(id)?.classList.add('open'); }
    function closeModal(id) { document.getElementById(id)?.classList.remove('open'); }

    // ─────────────────────────────────────────────────
    //  MOBILE SIDEBAR
    // ─────────────────────────────────────────────────
    function setupMobile() {
        menuToggle?.addEventListener('click', () => {
            sidebar.classList.add('open');
            sidebarOverlay.classList.add('open');
        });
        sidebarOverlay?.addEventListener('click', closeSidebar);
    }

    function closeSidebar() {
        sidebar?.classList.remove('open');
        sidebarOverlay?.classList.remove('open');
    }

    // ─────────────────────────────────────────────────
    //  CHAT HISTORY  (localStorage)
    // ─────────────────────────────────────────────────
    function loadHistory() {
        const history = getHistory();
        renderHistory(history);
    }

    function saveHistory(message) {
        const history = getHistory();
        const title = message.length > 32 ? message.slice(0, 32) + '…' : message;
        history.unshift({ id: Date.now(), title, ts: new Date().toISOString() });
        localStorage.setItem('sb_history', JSON.stringify(history.slice(0, 50)));
        renderHistory(history);
    }

    function getHistory() {
        return JSON.parse(localStorage.getItem('sb_history') || '[]');
    }

    function renderHistory(history) {
        if (!historyList) return;
        historyList.innerHTML = history.map(item => `
            <div class="history-item" title="${escapeHtml(item.title)}">${escapeHtml(item.title)}</div>
        `).join('') || '<div style="padding:8px 12px;font-size:13px;color:var(--text-dim)">No chats yet</div>';
    }

    // ─────────────────────────────────────────────────
    //  NEW CHAT
    // ─────────────────────────────────────────────────
    window.startNewChat = function () {
        messagesContainer.innerHTML = '';
        if (welcomeState) welcomeState.style.display = '';
        chatInput.value = '';
        chatInput.style.height = 'auto';
        sendBtn.disabled = true;
        sessionId = generateId();
        localStorage.setItem('sb_session', sessionId);
        switchView('chat');
        chatInput.focus();
        if (window.innerWidth <= 768) closeSidebar();
    };

    // ─────────────────────────────────────────────────
    //  COPY TEXT HELPER (global so onclick can call it)
    // ─────────────────────────────────────────────────
    window.copyText = async function (btn, text) {
        try {
            await navigator.clipboard.writeText(text);
            btn.title = 'Copied!';
            setTimeout(() => { btn.title = 'Copy'; }, 2000);
        } catch {}
    };

    // ─────────────────────────────────────────────────
    //  UTILITIES
    // ─────────────────────────────────────────────────
    function escapeHtml(str) {
        return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    // Very basic markdown → HTML
    function formatMarkdown(text) {
        return text
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/`(.+?)`/g, '<code style="background:#2a2a2a;padding:2px 6px;border-radius:4px;font-family:monospace">$1</code>')
            .replace(/\n/g, '<br>');
    }

    function generateId() {
        return 'sess_' + Math.random().toString(36).slice(2, 11);
    }
});
