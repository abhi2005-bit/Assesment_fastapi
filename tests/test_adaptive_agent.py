"""Test coverage for adaptive agent functionality."""
import pytest
from adaptive_agent import (
    AdaptiveQuestionAgent,
    UserContext,
    VisualCues,
    DifficultyLevel,
    QuestionType,
    create_adaptive_question_agent,
)


class TestDifficultySelection:
    """Test difficulty selection logic."""

    def test_difficulty_high_stress(self):
        agent = AdaptiveQuestionAgent({})
        user = UserContext(
            user_id="test",
            current_score=0.9,
            questions_attempted=10,
            correct_answers=9,
            average_response_time=25,
            current_streak=5,
            weak_topics=[],
            strong_topics=["python"],
            confidence_level=0.9,
            engagement_level=0.8,
            stress_indicators=0.2,
        )
        visual = VisualCues(
            eye_contact=0.5,
            attention_level=0.4,
            stress_indicators=0.8,
            confidence_level=0.3,
            distraction_count=5,
            posture_score=0.3,
        )

        difficulty = agent._select_difficulty(user, visual)
        assert difficulty == DifficultyLevel.EASY

    def test_difficulty_high_score(self):
        agent = AdaptiveQuestionAgent({})
        user = UserContext(
            user_id="test",
            current_score=0.8,
            questions_attempted=20,
            correct_answers=16,
            average_response_time=20,
            current_streak=8,
            weak_topics=[],
            strong_topics=["python", "algorithms"],
            confidence_level=0.9,
            engagement_level=0.9,
            stress_indicators=0.1,
        )

        difficulty = agent._select_difficulty(user, None)
        assert difficulty == DifficultyLevel.HARD

    def test_difficulty_medium_score(self):
        agent = AdaptiveQuestionAgent({})
        user = UserContext(
            user_id="test",
            current_score=0.6,
            questions_attempted=15,
            correct_answers=9,
            average_response_time=28,
            current_streak=2,
            weak_topics=["recursion"],
            strong_topics=["basics"],
            confidence_level=0.6,
            engagement_level=0.7,
            stress_indicators=0.3,
        )

        difficulty = agent._select_difficulty(user, None)
        assert difficulty == DifficultyLevel.MEDIUM

    def test_difficulty_low_score(self):
        agent = AdaptiveQuestionAgent({})
        user = UserContext(
            user_id="test",
            current_score=0.3,
            questions_attempted=5,
            correct_answers=1,
            average_response_time=35,
            current_streak=0,
            weak_topics=["everything"],
            strong_topics=[],
            confidence_level=0.2,
            engagement_level=0.3,
            stress_indicators=0.6,
        )

        difficulty = agent._select_difficulty(user, None)
        assert difficulty == DifficultyLevel.EASY


class TestFallbackQuestionGeneration:
    """Test fallback question generation."""

    def test_fallback_easy_question(self):
        agent = AdaptiveQuestionAgent({})
        question = agent._generate_fallback_question("basics", DifficultyLevel.EASY)

        assert question is not None
        assert question.topic == "basics"
        assert question.difficulty == DifficultyLevel.EASY
        assert question.type == QuestionType.MULTIPLE_CHOICE
        assert len(question.options) == 4
        assert question.correct_answer in question.options
        assert question.question_text != ""
        assert question.explanation != ""
        assert question.points == 5

    def test_fallback_medium_question(self):
        agent = AdaptiveQuestionAgent({})
        question = agent._generate_fallback_question("algorithms", DifficultyLevel.MEDIUM)

        assert question.difficulty == DifficultyLevel.MEDIUM
        assert question.points == 10
        assert len(question.options) == 4

    def test_fallback_hard_question(self):
        agent = AdaptiveQuestionAgent({})
        question = agent._generate_fallback_question("advanced", DifficultyLevel.HARD)

        assert question.difficulty == DifficultyLevel.HARD
        assert question.points == 15
        assert len(question.options) == 4

    def test_fallback_uniqueness(self):
        """Test that fallback generates unique questions."""
        agent = AdaptiveQuestionAgent({})
        questions = [
            agent._generate_fallback_question("python", DifficultyLevel.MEDIUM)
            for _ in range(5)
        ]

        # Check that questions have unique IDs
        ids = [q.id for q in questions]
        assert len(ids) == len(set(ids)), "Generated questions should have unique IDs"


class TestAdaptiveQuestionGeneration:
    """Test adaptive question generation with fallback."""

    def test_question_generation_without_llm(self):
        """Test question generation falls back when LLM unavailable."""
        agent = AdaptiveQuestionAgent({})  # Empty LLM providers
        user = UserContext(
            user_id="test",
            current_score=0.5,
            questions_attempted=0,
            correct_answers=0,
            average_response_time=30,
            current_streak=0,
            weak_topics=["basics"],
            strong_topics=[],
            confidence_level=0.5,
            engagement_level=0.7,
            stress_indicators=0.3,
        )

        question = agent.generate_adaptive_question(user, None, ["python"])
        
        assert question is not None
        assert question.topic == "python"
        assert question.type == QuestionType.MULTIPLE_CHOICE
        assert len(question.options) == 4

    def test_question_generation_with_preferred_topics(self):
        agent = AdaptiveQuestionAgent({})
        user = UserContext(
            user_id="test",
            current_score=0.5,
            questions_attempted=0,
            correct_answers=0,
            average_response_time=30,
            current_streak=0,
            weak_topics=[],
            strong_topics=[],
            confidence_level=0.5,
            engagement_level=0.7,
            stress_indicators=0.3,
        )

        question = agent.generate_adaptive_question(
            user, None, ["databases", "oop", "algorithms"]
        )
        
        assert question.topic in ["databases", "oop", "algorithms"]

    def test_question_generation_no_preferred_topics(self):
        agent = AdaptiveQuestionAgent({})
        user = UserContext(
            user_id="test",
            current_score=0.5,
            questions_attempted=0,
            correct_answers=0,
            average_response_time=30,
            current_streak=0,
            weak_topics=[],
            strong_topics=[],
            confidence_level=0.5,
            engagement_level=0.7,
            stress_indicators=0.3,
        )

        question = agent.generate_adaptive_question(user, None, None)
        
        assert question.topic == "programming"


class TestAgentFactory:
    """Test agent factory function."""

    def test_create_agent_with_empty_providers(self):
        agent = create_adaptive_question_agent({})
        assert isinstance(agent, AdaptiveQuestionAgent)

    def test_create_agent_with_none_providers(self):
        agent = create_adaptive_question_agent(None)
        assert isinstance(agent, AdaptiveQuestionAgent)

    def test_create_agent_with_providers(self):
        providers = {"mock_llm": "test_provider"}
        agent = create_adaptive_question_agent(providers)
        assert isinstance(agent, AdaptiveQuestionAgent)
        assert agent._llm_providers == providers


class TestVisualCuesIntegration:
    """Test integration with visual cues."""

    def test_stress_detection_impacts_difficulty(self):
        agent = AdaptiveQuestionAgent({})
        
        # High performer with high stress
        user_high_perf = UserContext(
            user_id="test",
            current_score=0.9,
            questions_attempted=20,
            correct_answers=18,
            average_response_time=20,
            current_streak=8,
            weak_topics=[],
            strong_topics=["python"],
            confidence_level=0.9,
            engagement_level=0.9,
            stress_indicators=0.1,
        )
        
        visual_high_stress = VisualCues(
            eye_contact=0.3,
            attention_level=0.2,
            stress_indicators=0.85,
            confidence_level=0.2,
            distraction_count=10,
            posture_score=0.2,
        )
        
        # Should reduce difficulty due to stress despite high performance
        difficulty = agent._select_difficulty(user_high_perf, visual_high_stress)
        assert difficulty == DifficultyLevel.EASY
