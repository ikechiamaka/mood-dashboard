(() => {
  const STORAGE_KEY = 'neurosense.chatbot.history.v1';
  const QUICK_ACTIONS = [
    { label: 'Summarize this patient', prompt: 'Summarize the currently selected patient for handoff.' },
    { label: 'Show high-risk patients', prompt: 'Show high-risk patients in my current scope and why they need attention.' },
    { label: 'Bed occupancy status', prompt: 'Give me current bed occupancy and stale telemetry status.' },
    { label: 'Open alerts now', prompt: 'List open alerts and suggest immediate next actions.' },
  ];

  function escapeHtml(value) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
    return String(value ?? '').replace(/[&<>"']/g, ch => map[ch] || ch);
  }

  function renderSimpleMarkdown(text) {
    const lines = String(text || '').split(/\r?\n/);
    const html = [];
    let inList = false;
    for (const raw of lines) {
      const line = raw.trimEnd();
      const isList = /^[-*]\s+/.test(line);
      if (isList && !inList) {
        inList = true;
        html.push('<ul>');
      }
      if (!isList && inList) {
        inList = false;
        html.push('</ul>');
      }
      const safe = escapeHtml(isList ? line.replace(/^[-*]\s+/, '') : line)
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/`([^`]+)`/g, '<code>$1</code>');
      if (!line) {
        html.push('<div class="chatbot-spacer"></div>');
      } else if (isList) {
        html.push(`<li>${safe}</li>`);
      } else {
        html.push(`<p>${safe}</p>`);
      }
    }
    if (inList) {
      html.push('</ul>');
    }
    return html.join('');
  }

  class PatientChatbot {
    constructor() {
      this.messages = [];
      this.currentPatientId = null;
      this.currentFacilityId = null;
      this.currentPatientName = null;
      this.currentUserName = null;
      this.currentUserRole = null;
      this.isOpen = false;
      this.isSending = false;

      this.container = document.getElementById('chatbotContainer');
      this.trigger = document.getElementById('chatbotTrigger');
      this.windowEl = document.getElementById('chatbotWindow');
      this.closeBtn = document.getElementById('chatbotClose');
      this.clearBtn = document.getElementById('chatbotClear');
      this.messagesEl = document.getElementById('chatbotMessages');
      this.quickActionsEl = document.getElementById('chatbotQuickActions');
      this.inputEl = document.getElementById('chatbotInput');
      this.sendBtn = document.getElementById('chatbotSend');
      this.contextEl = document.getElementById('chatbotContext');

      if (!this.container || !this.trigger || !this.windowEl) {
        return;
      }

      this._hydrate();
      this._bind();
      this._renderQuickActions();
      this._renderMessages();
      this._setContextLabel();
    }

    _bind() {
      this.trigger.addEventListener('click', () => this.toggle());
      if (this.closeBtn) {
        this.closeBtn.addEventListener('click', () => this.close());
      }
      if (this.clearBtn) {
        this.clearBtn.addEventListener('click', () => this.clearChat());
      }
      if (this.sendBtn) {
        this.sendBtn.addEventListener('click', () => this._sendFromInput());
      }
      if (this.inputEl) {
        this.inputEl.addEventListener('keydown', evt => {
          if (evt.key === 'Enter' && !evt.shiftKey) {
            evt.preventDefault();
            this._sendFromInput();
          }
        });
      }
      if (this.quickActionsEl) {
        this.quickActionsEl.addEventListener('click', evt => {
          const btn = evt.target.closest('button[data-prompt]');
          if (!btn || this.isSending) return;
          const prompt = btn.getAttribute('data-prompt') || '';
          if (prompt) this.sendMessage(prompt);
        });
      }
      document.addEventListener('keydown', evt => {
        if (evt.key === 'Escape' && this.isOpen) {
          this.close();
        }
      });
    }

    _hydrate() {
      try {
        const raw = window.sessionStorage.getItem(STORAGE_KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        if (Array.isArray(parsed)) {
          this.messages = parsed
            .filter(item => item && (item.role === 'user' || item.role === 'assistant' || item.role === 'error'))
            .slice(-40);
        }
      } catch (_) {
        this.messages = [];
      }
      if (!this.messages.length) {
        this.messages.push(this._defaultAssistantMessage());
      }
    }

    _persist() {
      try {
        window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(this.messages.slice(-40)));
      } catch (_) {
      }
    }

    _setContextLabel() {
      if (!this.contextEl) return;
      const userPart = this.currentUserName
        ? `User: ${this.currentUserName}${this.currentUserRole ? ` (${this.currentUserRole})` : ''}`
        : 'User: session';
      const patientPart = this.currentPatientName
        ? `Patient: ${this.currentPatientName}`
        : (this.currentPatientId ? `Patient ID: ${this.currentPatientId}` : 'No patient selected');
      const facilityPart = this.currentFacilityId ? `Facility: ${this.currentFacilityId}` : 'Facility scope: auto';
      this.contextEl.textContent = `${userPart} | ${patientPart} | ${facilityPart}`;
    }

    _defaultAssistantMessage() {
      const name = this.currentUserName ? ` ${this.currentUserName}` : '';
      return {
        role: 'assistant',
        content: `Hi${name}. Ask me for patient summaries, high-risk triage, bed status, or open alerts.`,
      };
    }

    clearChat() {
      this.messages = [this._defaultAssistantMessage()];
      this._renderMessages();
      if (this.inputEl) {
        this.inputEl.focus();
      }
    }

    _renderQuickActions() {
      if (!this.quickActionsEl) return;
      this.quickActionsEl.innerHTML = QUICK_ACTIONS.map(action => (
        `<button type="button" class="chatbot-quick-btn" data-prompt="${escapeHtml(action.prompt)}">${escapeHtml(action.label)}</button>`
      )).join('');
    }

    _renderMessages() {
      if (!this.messagesEl) return;
      this.messagesEl.innerHTML = this.messages.map(msg => {
        const roleClass = msg.role === 'user' ? 'message-user' : (msg.role === 'error' ? 'message-error' : 'message-assistant');
        const label = msg.role === 'user' ? 'You' : (msg.role === 'error' ? 'Error' : 'Assistant');
        const body = msg.role === 'assistant'
          ? renderSimpleMarkdown(msg.content)
          : `<p>${escapeHtml(msg.content)}</p>`;
        return `
          <article class="chatbot-message ${roleClass}">
            <div class="chatbot-message-label">${label}</div>
            <div class="chatbot-message-body">${body}</div>
          </article>
        `;
      }).join('');
      this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
      this._persist();
    }

    _setLoading(loading) {
      this.isSending = Boolean(loading);
      if (this.sendBtn) {
        this.sendBtn.disabled = this.isSending;
      }
      if (this.clearBtn) {
        this.clearBtn.disabled = this.isSending;
      }
      if (this.inputEl) {
        this.inputEl.disabled = this.isSending;
      }
      const existing = this.messagesEl ? this.messagesEl.querySelector('.chatbot-typing') : null;
      if (!this.messagesEl) return;
      if (this.isSending && !existing) {
        const node = document.createElement('div');
        node.className = 'chatbot-typing';
        node.innerHTML = '<span></span><span></span><span></span>';
        this.messagesEl.appendChild(node);
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
      } else if (!this.isSending && existing) {
        existing.remove();
      }
    }

    _sendFromInput() {
      if (!this.inputEl) return;
      const value = (this.inputEl.value || '').trim();
      if (!value || this.isSending) return;
      this.inputEl.value = '';
      this.sendMessage(value);
    }

    updateContext(patientId, facilityId, patientName) {
      this.currentPatientId = patientId ? Number(patientId) : null;
      this.currentFacilityId = facilityId !== undefined && facilityId !== null && facilityId !== '' ? Number(facilityId) : null;
      this.currentPatientName = patientName || null;
      this._setContextLabel();
    }

    updateUserContext(userName, userRole) {
      this.currentUserName = userName || null;
      this.currentUserRole = userRole || null;
      this._setContextLabel();
      if (this.messages.length === 1 && this.messages[0].role === 'assistant') {
        this.messages[0] = this._defaultAssistantMessage();
        this._renderMessages();
      }
    }

    toggle() {
      if (this.isOpen) {
        this.close();
      } else {
        this.open();
      }
    }

    open() {
      this.isOpen = true;
      this.windowEl.classList.remove('d-none');
      this.container.classList.add('is-open');
      this.trigger.setAttribute('aria-expanded', 'true');
      if (this.inputEl) {
        setTimeout(() => this.inputEl.focus(), 120);
      }
    }

    close() {
      this.isOpen = false;
      this.container.classList.remove('is-open');
      this.trigger.setAttribute('aria-expanded', 'false');
      setTimeout(() => {
        if (!this.isOpen) {
          this.windowEl.classList.add('d-none');
        }
      }, 180);
    }

    async sendMessage(message) {
      this.messages.push({ role: 'user', content: message });
      this._renderMessages();
      this._setLoading(true);
      try {
        const headers = typeof withCsrf === 'function'
          ? withCsrf({ 'Content-Type': 'application/json' })
          : { 'Content-Type': 'application/json' };
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers,
          credentials: 'same-origin',
          body: JSON.stringify({
            message,
            patient_id: this.currentPatientId,
            facility_id: this.currentFacilityId,
          }),
        });
        if (response.status === 401) {
          window.location.href = '/login';
          return;
        }
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(data.error || `Request failed (${response.status})`);
        }
        this.messages.push({ role: 'assistant', content: String(data.reply || 'No response available.') });
        this._renderMessages();
      } catch (error) {
        this.messages.push({
          role: 'error',
          content: `Unable to complete request: ${error && error.message ? error.message : 'unknown error'}`,
        });
        this._renderMessages();
      } finally {
        this._setLoading(false);
      }
    }
  }

  window.PatientChatbot = PatientChatbot;
})();
