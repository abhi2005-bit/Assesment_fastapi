from adaptive_agent import (
    AdaptiveQuestionAgent,
    UserContext,
)


def test_agent_generates_question():
    agent = AdaptiveQuestionAgent(llm_providers={})

    user = UserContext(
        user_id="test_user",
        current_score=0.6,
        questions_attempted=5,
        correct_answers=3,
        average_response_time=30,
        current_streak=1,
        weak_topics=[],
        strong_topics=[],
        confidence_level=0.5,
        engagement_level=0.7,
        stress_indicators=0.2,
    )

    question = agent.generate_adaptive_question(user)

    assert question.question_text
    assert len(question.options) == 4
    assert question.correct_answer in question.options
