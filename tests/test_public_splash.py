def test_primary_domain_renders_public_splash(app_db):
    app_module, _ = app_db
    client = app_module.app.test_client()

    response = client.get('/', base_url='https://melxhealth.com')

    assert response.status_code == 200
    assert b'Launching soon' in response.data
    assert b'Contact us' in response.data
    assert b'Healthcare Dashboard' not in response.data


def test_dashboard_subdomain_starts_at_login(app_db):
    app_module, _ = app_db
    client = app_module.app.test_client()

    response = client.get(
        '/',
        base_url='https://dashboard.melxhealth.com',
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers['Location'].endswith('/login')
