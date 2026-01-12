def test_full_assessment_flow(client):
    # start assessment
    start = client.post(
        "/api/assessment/start",
        json={"user_id": "flow_user"},
    )

    assert start.status_code == 200
    session_id = start.json()["session_id"]

    # get assessment
    get_session = client.get(f"/api/assessment/{session_id}")
    assert get_session.status_code == 200
