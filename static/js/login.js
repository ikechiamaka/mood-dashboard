(() => {
  const banner = document.getElementById('errorBanner');
  const hasError = document.body.dataset.error === 'true';
  if (hasError && banner) {
    banner.classList.add('is-visible');
  }
})();

(() => {
  const toggle = document.getElementById('togglePassword');
  const input = document.getElementById('password');
  if (!toggle || !input) return;
  toggle.addEventListener('click', () => {
    const isPassword = input.type === 'password';
    input.type = isPassword ? 'text' : 'password';
    const icon = toggle.querySelector('i');
    if (icon) {
      icon.classList.toggle('fa-eye');
      icon.classList.toggle('fa-eye-slash');
    }
    input.focus({ preventScroll: true });
  });
})();

(() => {
  const toastContainer = document.getElementById('loginToastContainer');
  function showToast(config) {
    const { title = 'Notice', message = '', variant = 'success' } = config || {};
    if (!toastContainer) {
      console.log(`[toast:${variant}] ${title}: ${message}`);
      return;
    }
    const wrapper = document.createElement('div');
    wrapper.className = `toast align-items-center text-bg-${variant === 'error' ? 'danger' : 'primary'} border-0`;
    wrapper.setAttribute('role', 'status');
    wrapper.setAttribute('aria-live', 'polite');
    wrapper.setAttribute('aria-atomic', 'true');

    const bodyWrap = document.createElement('div');
    bodyWrap.className = 'd-flex';

    const body = document.createElement('div');
    body.className = 'toast-body';
    const titleEl = document.createElement('strong');
    titleEl.className = 'd-block';
    titleEl.textContent = title;
    const messageEl = document.createElement('span');
    messageEl.textContent = message;
    body.appendChild(titleEl);
    body.appendChild(messageEl);

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'btn-close btn-close-white me-2 m-auto';
    closeBtn.setAttribute('data-bs-dismiss', 'toast');
    closeBtn.setAttribute('aria-label', 'Close');

    bodyWrap.appendChild(body);
    bodyWrap.appendChild(closeBtn);
    wrapper.appendChild(bodyWrap);
    toastContainer.appendChild(wrapper);
    const toast = new bootstrap.Toast(wrapper, { delay: 4000 });
    toast.show();
    wrapper.addEventListener('hidden.bs.toast', () => {
      wrapper.remove();
    });
  }

  async function submitSupportForm(formId, endpoint, successMessage) {
    const form = document.getElementById(formId);
    if (!form) return;
    const submitBtn = form.querySelector('button[type="submit"]');
    const submitLabel = submitBtn?.innerText;
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (!submitBtn) return;
      submitBtn.disabled = true;
      const loadingText = submitBtn.dataset.loadingText || 'Submitting...';
      submitBtn.innerText = loadingText;
      try {
        const formData = new FormData(form);
        const payload = Object.fromEntries(formData.entries());
        const resp = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!resp.ok) {
          const detail = await resp.json().catch(() => ({}));
          throw new Error(detail.error || 'Request failed');
        }
        form.reset();
        const modalEl = form.closest('.modal');
        if (modalEl && typeof bootstrap !== 'undefined') {
          const modal = bootstrap.Modal.getInstance(modalEl);
          modal?.hide();
        }
        showToast({ title: 'Support notified', message: successMessage, variant: 'success' });
      } catch (err) {
        console.error(err);
        showToast({ title: 'Unable to submit', message: err.message || 'Please try again shortly.', variant: 'error' });
      } finally {
        submitBtn.disabled = false;
        if (submitLabel) {
          submitBtn.innerText = submitLabel;
        }
      }
    });
  }

  submitSupportForm('requestAccessForm', '/support/request-access', 'Our team will reach out within one business day.');
  submitSupportForm('forgotPasswordForm', '/support/forgot-password', 'The care operations team will coordinate your reset shortly.');
})();
