from types import SimpleNamespace


def _install_openai_mock(monkeypatch, app_module, reply_text='ok', no_phi='0'):
    captured = {}
    monkeypatch.setenv('CHATBOT_LOCAL_ONLY', '0')
    monkeypatch.setenv('CHATBOT_NO_PHI', str(no_phi))

    class _Completions:
        def create(self, **kwargs):
            captured['kwargs'] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=reply_text))]
            )

    fake_openai = SimpleNamespace(
        api_key='test-key',
        chat=SimpleNamespace(completions=_Completions()),
    )
    monkeypatch.setattr(app_module, 'openai', fake_openai, raising=True)
    app_module._RATE_LIMITS.clear()
    return captured


def test_chat_requires_auth(app_db):
    flask_app, _db_module = app_db
    with flask_app.app.test_client() as client:
        resp = client.post('/api/chat', json={'message': 'hello'})
        assert resp.status_code == 401


def test_chat_includes_patient_context_for_allowed_staff(app_db, monkeypatch):
    flask_app, _db_module = app_db
    captured = _install_openai_mock(monkeypatch, flask_app, reply_text='summary ready')
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'staff1@demo.com'
            sess['role'] = 'staff'
            sess['facility_id'] = 1
            sess['assigned_patient_ids'] = [1]
        resp = client.post('/api/chat', json={'message': 'Need a concise status note for this patient', 'patient_id': 1, 'facility_id': 1})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body and body.get('reply') == 'summary ready'
        assert body.get('context', {}).get('patient_id') == 1
        assert body.get('context', {}).get('query_type') == 'general'
        assert body.get('context', {}).get('provider') == 'openai'

    system_prompt = captured['kwargs']['messages'][0]['content']
    assert 'Patient Context:' in system_prompt
    assert 'Blaine Cottrell' in system_prompt
    assert 'Facility Context:' in system_prompt


def test_chat_rejects_forbidden_patient_scope_for_staff(app_db, monkeypatch):
    flask_app, _db_module = app_db
    _install_openai_mock(monkeypatch, flask_app)
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'staff1@demo.com'
            sess['role'] = 'staff'
            sess['facility_id'] = 1
            sess['assigned_patient_ids'] = [1]
        resp = client.post('/api/chat', json={'message': 'Summarize patient 2', 'patient_id': 2})
        assert resp.status_code == 403
        body = resp.get_json()
        assert body and 'Forbidden patient scope' in (body.get('error') or '')


def test_chat_rejects_forbidden_facility_scope_for_facility_admin(app_db, monkeypatch):
    flask_app, _db_module = app_db
    _install_openai_mock(monkeypatch, flask_app)
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1
        resp = client.post('/api/chat', json={'message': 'Facility occupancy', 'facility_id': 2})
        assert resp.status_code == 403
        body = resp.get_json()
        assert body and 'Forbidden facility scope' in (body.get('error') or '')


def test_chat_local_only_mode_returns_local_provider(app_db, monkeypatch):
    flask_app, _db_module = app_db
    monkeypatch.setenv('CHATBOT_LOCAL_ONLY', '1')
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1
        resp = client.post('/api/chat', json={'message': 'Show bed occupancy', 'facility_id': 1})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body and isinstance(body.get('reply'), str)
        assert body.get('context', {}).get('provider') == 'local'
        assert body.get('context', {}).get('local_only') is True


def test_chat_local_about_prompt_returns_capability_text(app_db, monkeypatch):
    flask_app, _db_module = app_db
    monkeypatch.setenv('CHATBOT_LOCAL_ONLY', '1')
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1
        resp = client.post('/api/chat', json={'message': 'what is this chatbot about'})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body and 'NeuroSense Patient Assistant' in (body.get('reply') or '')


def test_chat_no_phi_mode_redacts_context_for_openai(app_db, monkeypatch):
    flask_app, _db_module = app_db
    captured = _install_openai_mock(monkeypatch, flask_app, reply_text='sanitized', no_phi='1')
    monkeypatch.setenv('CHATBOT_NO_PHI', '1')
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'staff1@demo.com'
            sess['role'] = 'staff'
            sess['facility_id'] = 1
            sess['assigned_patient_ids'] = [1]
        resp = client.post(
            '/api/chat',
            json={'message': 'Write a concise note about Blaine Cottrell and email me at test@example.com', 'patient_id': 1, 'facility_id': 1},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body and body.get('reply') == 'sanitized'
        assert body.get('context', {}).get('provider') == 'openai'
        assert body.get('context', {}).get('no_phi') is True

    system_prompt = captured['kwargs']['messages'][0]['content']
    user_prompt = captured['kwargs']['messages'][1]['content']
    assert 'Blaine Cottrell' not in system_prompt
    assert 'PAT-' in system_prompt
    assert '[redacted]' in system_prompt
    assert 'High-risk patient summaries:' in system_prompt
    assert 'test@example.com' not in user_prompt
    assert '[redacted-email]' in user_prompt


def test_chat_no_phi_reidentifies_aliases_in_response(app_db, monkeypatch):
    flask_app, _db_module = app_db
    monkeypatch.setenv('CHATBOT_LOCAL_ONLY', '0')
    monkeypatch.setenv('CHATBOT_NO_PHI', '1')
    monkeypatch.setenv('CHATBOT_REIDENTIFY_OUTPUT', '1')
    alias = flask_app._mask_identifier('Blaine Cottrell', 'PAT')
    captured = _install_openai_mock(monkeypatch, flask_app, reply_text=f"{alias} requires attention.", no_phi='1')
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'staff1@demo.com'
            sess['role'] = 'staff'
            sess['facility_id'] = 1
            sess['assigned_patient_ids'] = [1]
        resp = client.post('/api/chat', json={'message': 'Provide a quick note on the selected patient', 'patient_id': 1, 'facility_id': 1})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body and 'Blaine Cottrell requires attention.' == body.get('reply')
        assert body.get('context', {}).get('provider') == 'openai'

    system_prompt = captured['kwargs']['messages'][0]['content']
    assert 'PAT-' in system_prompt


def test_chat_prompt_enforces_template_rules(app_db, monkeypatch):
    flask_app, _db_module = app_db
    captured = _install_openai_mock(monkeypatch, flask_app, reply_text='ok', no_phi='1')
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'staff1@demo.com'
            sess['role'] = 'staff'
            sess['facility_id'] = 1
            sess['assigned_patient_ids'] = [1]
        resp = client.post('/api/chat', json={'message': 'Provide a concise operational note for the selected patient', 'patient_id': 1, 'facility_id': 1})
        assert resp.status_code == 200
    system_prompt = captured['kwargs']['messages'][0]['content']
    assert 'no markdown headings' in system_prompt
    assert 'Start with a direct answer' in system_prompt
    assert 'Keep output under 140 words' in system_prompt


def test_chat_reply_normalizer_removes_heading_tokens(app_db, monkeypatch):
    flask_app, _db_module = app_db
    reply = "### Summary\nPatient is stable.\n\n### Data Gaps\nNone"
    _install_openai_mock(monkeypatch, flask_app, reply_text=reply, no_phi='0')
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'staff1@demo.com'
            sess['role'] = 'staff'
            sess['facility_id'] = 1
            sess['assigned_patient_ids'] = [1]
        resp = client.post('/api/chat', json={'message': 'status update note', 'patient_id': 1, 'facility_id': 1})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body
        assert '###' not in (body.get('reply') or '')
        assert 'Summary' not in (body.get('reply') or '')


def test_chat_structured_queries_use_local_provider_even_when_openai_enabled(app_db, monkeypatch):
    flask_app, _db_module = app_db
    _install_openai_mock(monkeypatch, flask_app, reply_text='should not be used')
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'staff1@demo.com'
            sess['role'] = 'staff'
            sess['facility_id'] = 1
            sess['assigned_patient_ids'] = [1]
        resp = client.post('/api/chat', json={'message': 'Summarize the currently selected patient for handoff.', 'patient_id': 1, 'facility_id': 1})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body and body.get('context', {}).get('provider') == 'local'
        assert 'Blaine Cottrell' in (body.get('reply') or '')


def test_chat_patient_name_and_followup_facility_lookup(app_db, monkeypatch):
    flask_app, _db_module = app_db
    monkeypatch.setenv('CHATBOT_LOCAL_ONLY', '1')
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1
        r1 = client.post('/api/chat', json={'message': 'Show high-risk patients in my current scope and why they need attention.', 'facility_id': 1})
        assert r1.status_code == 200
        b1 = r1.get_json()
        assert b1 and 'Maria Green' in (b1.get('reply') or '')

        r2 = client.post('/api/chat', json={'message': 'which facilities are they in?', 'facility_id': 1})
        assert r2.status_code == 200
        b2 = r2.get_json()
        reply2 = b2.get('reply') or ''
        assert 'Blaine Cottrell' in reply2 and 'Maria Green' in reply2
        assert 'Facility' in reply2

        r3 = client.post('/api/chat', json={'message': 'what about maria green?', 'facility_id': 1})
        assert r3.status_code == 200
        b3 = r3.get_json()
        reply3 = b3.get('reply') or ''
        assert 'Maria Green' in reply3
        assert 'Bed B-5' in reply3

        r4 = client.post('/api/chat', json={'message': 'which facility is she in?', 'facility_id': 1})
        assert r4.status_code == 200
        b4 = r4.get_json()
        reply4 = b4.get('reply') or ''
        assert 'Maria Green' in reply4
        assert 'Facility' in reply4


def test_chat_count_queries_distinguish_access_vs_available_for_facilities(app_db, monkeypatch):
    flask_app, _db_module = app_db
    monkeypatch.setenv('CHATBOT_LOCAL_ONLY', '1')
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1

        r1 = client.post('/api/chat', json={'message': 'how many facilities do i have access to?'})
        assert r1.status_code == 200
        reply1 = (r1.get_json() or {}).get('reply') or ''
        assert 'access to 1 facility' in reply1.lower()

        r2 = client.post('/api/chat', json={'message': 'how many facilities are available?'})
        assert r2.status_code == 200
        reply2 = (r2.get_json() or {}).get('reply') or ''
        assert 'available' in reply2.lower()
        assert 'current scope' in reply2.lower()


def test_chat_count_queries_support_patients_and_beds(app_db, monkeypatch):
    flask_app, _db_module = app_db
    monkeypatch.setenv('CHATBOT_LOCAL_ONLY', '1')
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1
        r1 = client.post('/api/chat', json={'message': 'how many patients do i have access to?', 'facility_id': 1})
        assert r1.status_code == 200
        reply1 = (r1.get_json() or {}).get('reply') or ''
        assert 'access to' in reply1.lower() and 'patients' in reply1.lower()

        r2 = client.post('/api/chat', json={'message': 'how many beds are available?', 'facility_id': 1})
        assert r2.status_code == 200
        reply2 = (r2.get_json() or {}).get('reply') or ''
        assert 'beds available' in reply2.lower() or ('there are' in reply2.lower() and 'beds' in reply2.lower())
