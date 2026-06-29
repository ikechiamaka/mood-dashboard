def test_chat_requires_auth(app_db):
    flask_app, _db_module = app_db
    with flask_app.app.test_client() as client:
        resp = client.post('/api/chat', json={'message': 'hello'})
        assert resp.status_code == 401


def test_chat_rejects_forbidden_patient_scope_for_staff(app_db):
    flask_app, _db_module = app_db
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


def test_chat_rejects_forbidden_facility_scope_for_facility_admin(app_db):
    flask_app, _db_module = app_db
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1
        resp = client.post('/api/chat', json={'message': 'Facility occupancy', 'facility_id': 2})
        assert resp.status_code == 403
        body = resp.get_json()
        assert body and 'Forbidden facility scope' in (body.get('error') or '')


def test_chat_returns_local_provider_for_general_query(app_db):
    flask_app, _db_module = app_db
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'staff1@demo.com'
            sess['role'] = 'staff'
            sess['facility_id'] = 1
            sess['assigned_patient_ids'] = [1]
        resp = client.post('/api/chat', json={'message': 'Need a concise status note for this patient', 'patient_id': 1, 'facility_id': 1})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body and isinstance(body.get('reply'), str)
        assert body.get('context', {}).get('query_type') == 'general'
        assert body.get('context', {}).get('provider') == 'local'
        assert body.get('context', {}).get('local_only') is True
        assert body.get('context', {}).get('no_phi') is True


def test_chat_local_about_prompt_returns_capability_text(app_db):
    flask_app, _db_module = app_db
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1
        resp = client.post('/api/chat', json={'message': 'what is this chatbot about'})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body and 'MelX Health Patient Assistant' in (body.get('reply') or '')


def test_chat_reply_normalizer_removes_heading_tokens(app_db):
    flask_app, _db_module = app_db
    raw = "### Summary\nPatient is stable.\n\n### Data Gaps\nNone"
    normalized = flask_app._normalize_chatbot_reply_text(raw)
    assert '###' not in normalized
    assert 'Summary' not in normalized


def test_chat_structured_queries_use_local_provider(app_db):
    flask_app, _db_module = app_db
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


def test_chat_patient_name_and_followup_facility_lookup(app_db):
    flask_app, _db_module = app_db
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


def test_chat_count_queries_distinguish_access_vs_available_for_facilities(app_db):
    flask_app, _db_module = app_db
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


def test_chat_scope_info_handles_singular_facility_access_question(app_db):
    flask_app, _db_module = app_db
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1
        resp = client.post('/api/chat', json={'message': 'which facility do i have access to?'})
        assert resp.status_code == 200
        body = resp.get_json() or {}
        reply = (body.get('reply') or '').lower()
        assert body.get('context', {}).get('query_type') == 'scope_info'
        assert 'access to 1 facility' in reply
        assert 'facility 1' in reply


def test_chat_count_queries_support_patients_and_beds(app_db):
    flask_app, _db_module = app_db
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


def test_chat_count_patients_in_facility_queries_return_patient_counts(app_db):
    flask_app, _db_module = app_db
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1

        r1 = client.post('/api/chat', json={'message': 'how many patients are in facility 1?'})
        assert r1.status_code == 200
        b1 = r1.get_json() or {}
        reply1 = (b1.get('reply') or '').lower()
        assert b1.get('context', {}).get('query_type') == 'count_lookup'
        assert 'patients' in reply1
        assert 'facility available' not in reply1

        r2 = client.post('/api/chat', json={'message': 'which facility do i have access to?'})
        assert r2.status_code == 200
        r3 = client.post('/api/chat', json={'message': 'how many patients are in that facility?'})
        assert r3.status_code == 200
        reply3 = ((r3.get_json() or {}).get('reply') or '').lower()
        assert 'patients' in reply3
        assert 'facility available' not in reply3


def test_chat_all_patients_summary_returns_scope_wide_list(app_db):
    flask_app, _db_module = app_db
    with flask_app.app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user'] = 'fac1admin@demo.com'
            sess['role'] = 'facility_admin'
            sess['facility_id'] = 1
        resp = client.post(
            '/api/chat',
            json={'message': 'give me a summary of all the patients', 'patient_id': 1, 'facility_id': 1},
        )
        assert resp.status_code == 200
        body = resp.get_json() or {}
        reply = body.get('reply') or ''
        assert body.get('context', {}).get('query_type') == 'patients_summary'
        assert 'summary of' in reply.lower() and 'patients' in reply.lower()
        assert 'Blaine Cottrell' in reply
        assert 'Maria Green' in reply
