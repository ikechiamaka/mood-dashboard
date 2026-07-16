from werkzeug.security import check_password_hash


def _login_session(client, email, role, facility_id=None):
    with client.session_transaction() as sess:
        sess["user"] = email
        sess["role"] = role
        if facility_id is not None:
            sess["facility_id"] = facility_id


def test_user_can_change_own_password_with_current_password(app_db):
    flask_app, db_module = app_db
    with flask_app.app.test_client() as client:
        _login_session(client, "superadmin@demo.com", "super_admin")
        resp = client.post(
            "/api/me/password",
            json={"current_password": "admin123", "new_password": "new-admin-123"},
        )
        assert resp.status_code == 200

    user = db_module.db_get_user_by_email("superadmin@demo.com")
    assert user is not None
    assert check_password_hash(user["password_hash"], "new-admin-123")


def test_user_password_change_rejects_wrong_current_password(app_db):
    flask_app, _db_module = app_db
    with flask_app.app.test_client() as client:
        _login_session(client, "superadmin@demo.com", "super_admin")
        resp = client.post(
            "/api/me/password",
            json={"current_password": "wrong-password", "new_password": "new-admin-123"},
        )
        assert resp.status_code == 400
        assert "current password is incorrect" in ((resp.get_json() or {}).get("error") or "")


def test_admin_can_reset_user_password(app_db):
    flask_app, db_module = app_db
    with flask_app.app.test_client() as client:
        _login_session(client, "superadmin@demo.com", "super_admin")
        resp = client.patch(
            "/api/users/staff1@demo.com",
            json={"password": "reset-staff-123"},
        )
        assert resp.status_code == 200

    user = db_module.db_get_user_by_email("staff1@demo.com")
    assert user is not None
    assert check_password_hash(user["password_hash"], "reset-staff-123")
