/**
 * Senter Web UI Application
 * Communicates with Python backend via pywebview API
 */

// State
let currentView = 'chat';
let isLoading = false;

// DOM Elements
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const messagesContainer = document.getElementById('messagesContainer');
const loadingIndicator = document.getElementById('loadingIndicator');
const sidebarBtns = document.querySelectorAll('.sidebar-btn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadResearch();
    setupEventListeners();
});

function setupEventListeners() {
    // Send button
    sendBtn.addEventListener('click', handleSend);

    // Enter to send
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // Sidebar navigation
    sidebarBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.dataset.view;
            switchView(view);
        });
    });
}

async function handleSend() {
    const query = queryInput.value.trim();
    if (!query || isLoading) return;

    // Add user message
    addUserMessage(query);
    queryInput.value = '';

    // Show loading
    showLoading(true);

    try {
        // Call Python backend
        if (window.pywebview && window.pywebview.api) {
            const result = await window.pywebview.api.research(query);
            if (result.success) {
                addResearchCard(result);
            } else {
                addErrorMessage(result.error || 'Research failed');
            }
        } else {
            // Fallback for testing without pywebview
            console.log('Would research:', query);
            setTimeout(() => {
                addResearchCard({
                    topic: query,
                    summary: 'This is a test summary for the query.',
                    key_insights: ['Test insight 1', 'Test insight 2'],
                    confidence: 0.8,
                    sources_found: 3,
                    research_id: Date.now(),
                    timestamp: new Date().toLocaleTimeString()
                });
            }, 1000);
        }
    } catch (error) {
        console.error('Research error:', error);
        addErrorMessage('Failed to complete research');
    } finally {
        showLoading(false);
    }
}

async function loadResearch() {
    try {
        if (window.pywebview && window.pywebview.api) {
            const research = await window.pywebview.api.get_research();
            if (research && research.length > 0) {
                research.forEach(r => addResearchCard(r, false));
            } else {
                showEmptyState();
            }
        } else {
            // Show empty state for testing
            showEmptyState();
        }
    } catch (error) {
        console.error('Load error:', error);
        showEmptyState();
    }
}

function addUserMessage(text) {
    const card = document.createElement('div');
    card.className = 'message-card query-card';
    card.innerHTML = `
        <div class="message-header">
            <span class="message-sender user">You</span>
        </div>
        <div class="message-content">${escapeHtml(text)}</div>
        <div class="message-time">${new Date().toLocaleTimeString()}</div>
    `;
    messagesContainer.appendChild(card);
    scrollToBottom();
}

function addResearchCard(data, animate = true) {
    const confidenceClass = data.confidence >= 0.8 ? '' : data.confidence >= 0.6 ? 'medium' : 'low';
    const confidencePercent = Math.round((data.confidence || 0) * 100);

    const card = document.createElement('div');
    card.className = 'message-card';
    if (!animate) card.style.animation = 'none';

    let insightsHtml = '';
    if (data.key_insights && data.key_insights.length > 0) {
        insightsHtml = `
            <div class="message-insights">
                <div class="insights-title">Key Insights</div>
                ${data.key_insights.map(i => `<div class="insight-item">${escapeHtml(i)}</div>`).join('')}
            </div>
        `;
    }

    card.innerHTML = `
        <div class="message-header">
            <span class="message-sender assistant">${escapeHtml(data.topic || 'Research')}</span>
            <span class="confidence-badge ${confidenceClass}">${confidencePercent}%</span>
        </div>
        <div class="message-content">${escapeHtml(data.summary || '')}</div>
        ${insightsHtml}
        <div class="message-footer">
            <div class="message-meta">
                <span>📚 ${data.sources_found || 0} sources</span>
                <span>🕐 ${data.timestamp || new Date().toLocaleTimeString()}</span>
            </div>
            <div class="star-rating" data-id="${data.research_id || 0}">
                ${[1,2,3,4,5].map(i => `
                    <button class="star-btn ${(data.rating || 0) >= i ? 'filled' : ''}" data-rating="${i}">★</button>
                `).join('')}
            </div>
        </div>
        <div class="agent-badge">AGENT: RESEARCH_AGENT</div>
    `;

    // Add rating handlers
    const starBtns = card.querySelectorAll('.star-btn');
    starBtns.forEach(btn => {
        btn.addEventListener('click', () => handleRating(data.research_id, parseInt(btn.dataset.rating), starBtns));
    });

    messagesContainer.appendChild(card);
    scrollToBottom();
}

function addErrorMessage(text) {
    const card = document.createElement('div');
    card.className = 'message-card';
    card.innerHTML = `
        <div class="message-header">
            <span class="message-sender assistant">Error</span>
        </div>
        <div class="message-content" style="color: #ef4444;">${escapeHtml(text)}</div>
    `;
    messagesContainer.appendChild(card);
    scrollToBottom();
}

async function handleRating(researchId, rating, starBtns) {
    // Update UI
    starBtns.forEach((btn, i) => {
        btn.classList.toggle('filled', i < rating);
    });

    // Save to backend
    try {
        if (window.pywebview && window.pywebview.api) {
            await window.pywebview.api.rate_research(researchId, rating);
        }
    } catch (error) {
        console.error('Rating error:', error);
    }
}

function showLoading(show) {
    isLoading = show;
    loadingIndicator.style.display = show ? 'flex' : 'none';
    sendBtn.disabled = show;
}

function showEmptyState() {
    messagesContainer.innerHTML = `
        <div class="empty-state">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            <h3>No research yet</h3>
            <p>Ask a question to start researching</p>
        </div>
    `;
}

function switchView(view) {
    currentView = view;
    sidebarBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === view);
    });

    // TODO: Implement view switching
    if (view !== 'chat') {
        messagesContainer.innerHTML = `
            <div class="empty-state">
                <h3>${view.charAt(0).toUpperCase() + view.slice(1)}</h3>
                <p>Coming soon</p>
            </div>
        `;
    } else {
        loadResearch();
    }
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Expose for Python callback
window.refreshResearch = loadResearch;
