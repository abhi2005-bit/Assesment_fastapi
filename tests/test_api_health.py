def test_health_endpoint(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "active_sessions" in data
