import app as app_module


if __name__ == '__main__':
    app_module._alert_cursor_id = app_module.db_get_latest_telemetry_id()
    app_module._alert_monitor_loop()
