def test_generate_question(client):
    payload = {
        "user_context": {
            "user_id": "test_user",
            "current_score": 0.5,
            "questions_attempted": 0,
            "correct_answers": 0,
            "average_response_time": 30,
            "current_streak": 0,
            "weak_topics": [],
            "strong_topics": [],
            "confidence_level": 0.5,
            "engagement_level": 0.7,
            "stress_indicators": 0.3,
        }
    }

    response = client.post("/api/questions/generate", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "question" in data
    assert "options" in data
    assert len(data["options"]) == 4
