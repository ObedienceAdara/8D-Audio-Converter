from spatial_audio_converter.api.app import create_app


def test_health_endpoint():
    app = create_app()
    app.testing = True
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
