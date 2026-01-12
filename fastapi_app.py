import os
import sys
import uuid
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq


# PATH SETUP

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))


# IMPORT AGENT

from adaptive_agent import (
    create_adaptive_question_agent,
    UserContext,
    VisualCues,
    DifficultyLevel
)


# ENV & LOGGING

load_dotenv(BASE_DIR / ".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("assessment-api")


# FASTAPI INIT

app = FastAPI(
    title="Assessment System API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# LLM PROVIDER

llm_providers = {}
if os.getenv("GROQ_API_KEY"):
    llm_providers["groq"] = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("Groq LLM enabled")
else:
    logger.warning("Groq API key missing → fallback only")

agent = create_adaptive_question_agent(llm_providers)


# IN-MEMORY STORAGE (REAL STATE)

sessions: Dict[str, Dict[str, Any]] = {}
users: Dict[str, Dict[str, Any]] = {}


# Pydantic MODELS

class UserContextModel(BaseModel):
    user_id: str
    current_score: float = 0.5
    questions_attempted: int = 0
    correct_answers: int = 0
    average_response_time: float = 30
    current_streak: int = 0
    weak_topics: List[str] = []
    strong_topics: List[str] = []
    confidence_level: float = 0.5
    engagement_level: float = 0.7
    stress_indicators: float = 0.3


class VisualCuesModel(BaseModel):
    eye_contact: float = 0.0
    attention_level: float = 0.0
    stress_indicators: float = 0.0
    confidence_level: float = 0.0
    distraction_count: int = 0
    posture_score: float = 0.0


class QuestionRequest(BaseModel):
    user_context: UserContextModel
    visual_cues: Optional[VisualCuesModel] = None
    preferred_topics: Optional[List[str]] = None


class AnswerSubmitRequest(BaseModel):
    session_id: str
    question_id: str
    user_answer: str
    time_taken: float



# HEALTH & DEBUG

@app.get("/")
def root():
    return {"status": "running", "docs": "/api/docs"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm_enabled": bool(llm_providers),
        "active_sessions": len(sessions)
    }


@app.get("/api/providers")
def providers():
    return {"providers": list(llm_providers.keys())}



# ASSESSMENT FLOW

@app.post("/api/assessment/start")
def start_assessment(user: UserContextModel):
    session_id = str(uuid.uuid4())

    users[user.user_id] = user.dict()

    sessions[session_id] = {
        "user_id": user.user_id,
        "questions": [],
        "answers": [],
        "started_at": time.time()
    }

    return {"session_id": session_id, "status": "started"}


@app.get("/api/assessment/{session_id}")
def get_assessment(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    return sessions[session_id]


@app.delete("/api/assessment/{session_id}")
def delete_assessment(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    del sessions[session_id]
    return {"status": "deleted"}



# QUESTION ENDPOINTS

@app.post("/api/questions/generate")
def generate_question(req: QuestionRequest):
    user_ctx = UserContext(**req.user_context.dict())
    visual = VisualCues(**req.visual_cues.dict()) if req.visual_cues else None

    question = agent.generate_adaptive_question(
        user_ctx, visual, req.preferred_topics
    )

    return {
        "id": question.id,
        "topic": question.topic,
        "difficulty": question.difficulty.value,
        "question": question.question_text,
        "options": question.options,
        "time_limit": question.time_limit,
        "points": question.points,
        "correct_answer": question.correct_answer  # needed for evaluation
    }


@app.post("/api/questions/simple")
def simple_question(topic: str = "basics"):
    q = agent._fallback_question(topic, DifficultyLevel.EASY)
    return {
        "question": q.question_text,
        "options": q.options,
        "answer": q.correct_answer
    }


@app.post("/api/questions/batch-generate")
def batch_generate(count: int = 5, topic: str = "basics"):
    questions = []
    for _ in range(count):
        q = agent._fallback_question(topic, DifficultyLevel.EASY)
        questions.append({
            "question": q.question_text,
            "options": q.options,
            "answer": q.correct_answer
        })
    return {"count": len(questions), "questions": questions}



# ANSWERS (REAL EVALUATION)

@app.post("/api/answers/submit")
def submit_answer(req: AnswerSubmitRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[req.session_id]
    last_question = session["questions"][-1] if session["questions"] else None

    is_correct = False
    if last_question:
        is_correct = req.user_answer == last_question["correct_answer"]

    session["answers"].append({
        "question_id": req.question_id,
        "answer": req.user_answer,
        "time_taken": req.time_taken,
        "correct": is_correct
    })

    return {
        "correct": is_correct,
        "total_answers": len(session["answers"])
    }


@app.post("/api/answers/evaluate")
def evaluate_answer(question_id: str, user_answer: str, correct_answer: str):
    return {
        "question_id": question_id,
        "correct": user_answer == correct_answer
    }



# ANALYTICS (REAL COMPUTATION)

@app.get("/api/analytics/performance")
def performance(user_id: Optional[str] = None):
    total = correct = 0
    times = []

    for s in sessions.values():
        for a in s["answers"]:
            total += 1
            if a["correct"]:
                correct += 1
            times.append(a["time_taken"])

    accuracy = (correct / total) * 100 if total else 0
    avg_time = sum(times) / len(times) if times else 0

    return {
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": round(accuracy, 2),
        "average_time": round(avg_time, 2)
    }


@app.get("/api/analytics/summary")
def analytics_summary():
    return {
        "sessions": len(sessions),
        "users": len(users),
        "system_status": "operational"
    }



# UTILITIES

@app.get("/api/utils/validate-answer")
def validate_answer(answer: str):
    return {"valid": bool(answer), "length": len(answer)}


@app.get("/api/utils/difficulty-levels")
def difficulty_levels():
    return {"levels": [d.value for d in DifficultyLevel]}


@app.get("/api/utils/question-types")
def question_types():
    return {"types": ["multiple_choice"]}


@app.get("/api/utils/topics")
def topics():
    return {
        "topics": ["basics", "data_structures", "algorithms", "databases", "oop"]
    }



# RUN

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host="127.0.0.1",
        port=8001,
        reload=True
    )
