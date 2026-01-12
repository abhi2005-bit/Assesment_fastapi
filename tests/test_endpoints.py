"""Test coverage for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from fastapi_app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints."""

    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_providers_endpoint(self, client):
        response = client.get("/api/providers")
        assert response.status_code == 200
        assert "providers" in response.json()


class TestQuestionEndpoints:
    """Test question generation endpoints."""

    def test_batch_generate_questions(self, client):
        response = client.post("/api/questions/batch-generate", params={"count": 3, "topic": "basics"})
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert "questions" in data

    def test_batch_generate_default_params(self, client):
        response = client.post("/api/questions/batch-generate")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 5

    def test_simple_question(self, client):
        response = client.post("/api/questions/simple", params={"topic": "basics"})
        assert response.status_code == 200
        data = response.json()
        assert "question" in data
        assert "options" in data
        assert "answer" in data


class TestAnswerEndpoints:
    """Test answer submission and evaluation."""

    def test_submit_answer(self, client):
        # First start an assessment
        start = client.post("/api/assessment/start", json={"user_id": "answer_test_user"})
        session_id = start.json()["session_id"]

        # Generate a question
        q_response = client.post("/api/questions/generate", json={
            "user_context": {
                "user_id": "answer_test_user",
                "current_score": 0.5,
                "questions_attempted": 0,
                "correct_answers": 0,
                "average_response_time": 30,
                "current_streak": 0,
                "weak_topics": [],
                "strong_topics": [],
                "confidence_level": 0.5,
                "engagement_level": 0.7,
                "stress_indicators": 0.3
            },
            "visual_cues": None,
            "preferred_topics": ["basics"]
        })
        
        # Submit an answer
        answer_response = client.post("/api/answers/submit", json={
            "session_id": session_id,
            "question_id": "q_123",
            "user_answer": "A",
            "time_taken": 15.5
        })
        
        assert answer_response.status_code == 200
        data = answer_response.json()
        assert "correct" in data
        assert "total_answers" in data


class TestAnalyticsEndpoints:
    """Test analytics endpoints."""

    def test_performance_analytics(self, client):
        response = client.get("/api/analytics/performance")
        assert response.status_code == 200
        data = response.json()
        assert "total_questions" in data
        assert "correct_answers" in data
        assert "accuracy" in data
        assert "average_time" in data

    def test_analytics_summary(self, client):
        response = client.get("/api/analytics/summary")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "users" in data
        assert "system_status" in data
        assert data["system_status"] == "operational"


class TestUtilityEndpoints:
    """Test utility endpoints."""

    def test_validate_answer(self, client):
        response = client.get("/api/utils/validate-answer?answer=test")
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["length"] == 4

    def test_validate_empty_answer(self, client):
        response = client.get("/api/utils/validate-answer?answer=")
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

    def test_difficulty_levels(self, client):
        response = client.get("/api/utils/difficulty-levels")
        assert response.status_code == 200
        data = response.json()
        assert "levels" in data
        assert "easy" in data["levels"]
        assert "medium" in data["levels"]
        assert "hard" in data["levels"]

    def test_question_types(self, client):
        response = client.get("/api/utils/question-types")
        assert response.status_code == 200
        data = response.json()
        assert "types" in data
        assert "multiple_choice" in data["types"]

    def test_topics(self, client):
        response = client.get("/api/utils/topics")
        assert response.status_code == 200
        data = response.json()
        assert "topics" in data
        assert len(data["topics"]) > 0


class TestAssessmentManagement:
    """Test assessment lifecycle management."""

    def test_get_assessment(self, client):
        # Start assessment
        start = client.post("/api/assessment/start", json={"user_id": "get_test_user"})
        session_id = start.json()["session_id"]

        # Get assessment
        response = client.get(f"/api/assessment/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "get_test_user"

    def test_get_nonexistent_assessment(self, client):
        response = client.get("/api/assessment/nonexistent_id")
        assert response.status_code == 404

    def test_delete_assessment(self, client):
        # Start assessment
        start = client.post("/api/assessment/start", json={"user_id": "delete_test_user"})
        session_id = start.json()["session_id"]

        # Delete assessment
        response = client.delete(f"/api/assessment/{session_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        # Verify it's deleted
        response = client.get(f"/api/assessment/{session_id}")
        assert response.status_code == 404

    def test_delete_nonexistent_assessment(self, client):
        response = client.delete("/api/assessment/nonexistent_id")
        assert response.status_code == 404
