import json
import logging
import time
import uuid
import hashlib
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import google.generativeai as genai

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# Try to import OpenAI (for OpenRouter and Perplexity)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Initialize multiple AI providers
def _initialize_providers():
    """Initialize all available AI providers"""
    providers = {}
    
    # Groq
    if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
        try:
            providers['groq'] = Groq(api_key=os.getenv("GROQ_API_KEY"))
            logging.info("Groq client initialized")
        except Exception as e:
            logging.error(f"Failed to initialize Groq: {e}")
    
    # OpenRouter
    if OPENAI_AVAILABLE and os.getenv("OPENROUTER_API_KEY"):
        try:
            providers['openrouter'] = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
            logging.info("OpenRouter client initialized")
        except Exception as e:
            logging.error(f"Failed to initialize OpenRouter: {e}")
    
    # Perplexity
    if OPENAI_AVAILABLE and os.getenv("PERPLEXITY_API_KEY"):
        try:
            providers['perplexity'] = OpenAI(
                api_key=os.getenv("PERPLEXITY_API_KEY"),
                base_url="https://api.perplexity.ai"
            )
            logging.info("Perplexity client initialized")
        except Exception as e:
            logging.error(f"Failed to initialize Perplexity: {e}")
    
    return providers

# Initialize providers globally
AI_PROVIDERS = _initialize_providers()

# Set page config as the first Streamlit command
st.set_page_config(page_title="Online Assessment (MVP)", layout="wide", initial_sidebar_state="collapsed")

# Multiple API key management
def _get_api_keys() -> list:
    """Get all available Gemini API keys"""
    keys = []
    for i in range(1, 6):  # Support up to 5 keys
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)
    
    # Also check default key
    default_key = os.getenv("GEMINI_API_KEY")
    if default_key and default_key not in keys:
        keys.insert(0, default_key)  # Prioritize default key
    
    return keys

def _get_current_api_key() -> str:
    """Get current working API key"""
    current_key_index = st.session_state.get('current_key_index', 0)
    keys = _get_api_keys()
    if keys and current_key_index < len(keys):
        return keys[current_key_index]
    return os.getenv("GEMINI_API_KEY", "")

def _rotate_api_key() -> str:
    """Rotate to next available API key"""
    keys = _get_api_keys()
    current_index = st.session_state.get('current_key_index', 0)
    
    if len(keys) <= 1:
        return _get_current_api_key()  # No rotation needed
    
    # Try next key
    next_index = (current_index + 1) % len(keys)
    st.session_state['current_key_index'] = next_index
    
    logging.info(f"Rotated to API key {next_index + 1}/{len(keys)}")
    return keys[next_index]

# Initialize API key rotation
if 'current_key_index' not in st.session_state:
    st.session_state['current_key_index'] = 0

GEMINI_API_KEY = _get_current_api_key()
if not GEMINI_API_KEY:
    st.error("❌ No GEMINI_API_KEY found in environment variables!")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)


@dataclass
class CandidateProfile:
    name: str
    branch: str
    passing_year: int
    university: str
    programming_language: str
    difficulty: str
    assessment_type: str


@dataclass
class TestConfig:
    total_questions: int
    duration_seconds: int




@dataclass
class Question:
    id: str
    type: str  # "mcq", "coding", "debugging"
    topic: str
    difficulty: str
    question: str
    options: list = None  # for MCQ
    answer_index: int = None  # for MCQ
    starter_code: str = None  # for coding/debugging
    solution: str = None  # for coding/debugging
    test_cases: list = None  # for coding/debugging


def _init_state() -> None:
    defaults = {
        "page": "registration",
        "profile": None,
        "accepted_rules": False,
        "rules_confirmed": False,
        "system_check_passed": False,
        "attempt_id": None,
        "test_config": None,
        "questions": None,
        "current_q": 0,
        "answers": {},
        "test_started_at": None,
        "test_submitted_at": None,
        "result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _set_page(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")




def _load_syllabus(language: str, difficulty: str) -> str:
    base = Path(__file__).parent / "data" / "knowledge_base"
    file_map = {
        "Python": "python_syllabus.txt",
        "Java": "java_syllabus.txt",
        "JavaScript": "javascript_syllabus.txt",
        "C++": "cpp_syllabus.txt",
        "SQL": "sql_syllabus.txt",
    }
    file_path = base / file_map.get(language, "python_syllabus.txt")
    if not file_path.exists():
        return ""
    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")
    level = difficulty.lower()
    filtered = []
    capture = False
    for line in lines:
        if line.strip().startswith("##") and level in line.lower():
            capture = True
        elif line.strip().startswith("##") and level not in line.lower():
            capture = False
        elif capture and line.strip():
            filtered.append(line.strip())
    return "\n".join(filtered)


def _load_used_hashes() -> set:
    base = Path(__file__).parent / "data" / "knowledge_base" / "used_question_hashes.json"
    if not base.exists():
        return set()
    try:
        data = json.loads(base.read_text(encoding="utf-8"))
        return set(data)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logging.error(f"Failed to load used hashes: {e}")
        return set()


def _save_used_hashes(hashes: set) -> None:
    base = Path(__file__).parent / "data" / "knowledge_base" / "used_question_hashes.json"
    base.write_text(json.dumps(sorted(hashes), ensure_ascii=False, indent=2), encoding="utf-8")


def _question_fingerprint(question: str, options: list) -> str:
    combined = question + "|" + "|".join(options)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def _get_available_gemini_model():
    try:
        models = list(genai.list_models())
        
        # Check if we've recently hit quota limits
        quota_status = _check_quota_status()
        
        if quota_status['recent_quota_hit']:
            logging.warning("Recent quota hit detected, using conservative model selection")
            # Use models with higher limits when quota is tight
            preferred_models = [
                'gemini-2.5-flash-lite',  # Highest daily limit (1000/day)
                'gemini-1.5-flash',       # Good balance
                'gemini-flash-latest'        # Fallback option
            ]
        else:
            # Normal model selection for full quota availability
            preferred_models = [
                'gemini-1.5-flash',
                'gemini-1.5-pro', 
                'gemini-pro',
                'gemini-pro-latest',
                'gemini-flash-latest'
            ]
        
        # Try preferred models first
        for preferred in preferred_models:
            for m in models:
                if preferred in m.name and "generateContent" in m.supported_generation_methods:
                    logging.info(f"Selected model: {preferred}")
                    return m.name.split("/")[-1]
        
        # Fallback to any available model
        for m in models:
            if "generateContent" in m.supported_generation_methods:
                logging.warning(f"Using fallback model: {m.name}")
                return m.name.split("/")[-1]
        return None
    except (ConnectionError, PermissionError, Exception) as e:
        st.error(f"Failed to list models: {e}")
        logging.error(f"Gemini API model listing failed: {e}")
        return None


def _check_quota_status() -> dict:
    """Check recent quota usage and provide recommendations"""
    quota_file = Path(__file__).parent / "data" / "quota_status.json"
    
    try:
        if quota_file.exists():
            data = json.loads(quota_file.read_text(encoding="utf-8"))
            last_hit = data.get('last_quota_hit', 0)
            hit_count = data.get('hit_count', 0)
            key_index = data.get('current_key_index', 0)
            
            # Check if quota was hit in last 2 hours
            time_since_hit = time.time() - last_hit
            recent_hit = time_since_hit < 7200  # 2 hours
            
            return {
                'recent_quota_hit': recent_hit,
                'hit_count': hit_count,
                'time_since_hit': time_since_hit,
                'current_key_index': key_index
            }
        else:
            return {'recent_quota_hit': False, 'hit_count': 0, 'time_since_hit': float('inf'), 'current_key_index': 0}
    except Exception as e:
        logging.error(f"Error checking quota status: {e}")
        return {'recent_quota_hit': False, 'hit_count': 0, 'time_since_hit': float('inf'), 'current_key_index': 0}


def _update_quota_status(hit: bool = True, key_index: int = None) -> None:
    """Update quota status tracking"""
    quota_file = Path(__file__).parent / "data" / "quota_status.json"
    quota_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if quota_file.exists():
            data = json.loads(quota_file.read_text(encoding="utf-8"))
        else:
            data = {'last_quota_hit': 0, 'hit_count': 0, 'current_key_index': 0}
        
        if hit:
            data['last_quota_hit'] = time.time()
            data['hit_count'] = data.get('hit_count', 0) + 1
        
        if key_index is not None:
            data['current_key_index'] = key_index
        
        quota_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logging.error(f"Error updating quota status: {e}")


def _handle_quota_exhaustion() -> str:
    """Handle quota exhaustion by rotating API keys"""
    keys = _get_api_keys()
    
    if len(keys) <= 1:
        st.warning("⚠️ Only one API key available. Waiting for quota reset.")
        return _get_current_api_key()
    
    # Try rotating to next key
    current_index = st.session_state.get('current_key_index', 0)
    next_index = (current_index + 1) % len(keys)
    
    # Check if next key also has recent quota hit
    quota_status = _check_quota_status()
    if quota_status['recent_quota_hit'] and next_index == quota_status.get('current_key_index', 0):
        st.warning(f"⚠️ All API keys exhausted. Using default questions.")
        return None
    
    # Rotate to next key
    new_key = _rotate_api_key()
    st.info(f"🔄 Rotated to API key {next_index + 1}/{len(keys)}")
    
    # Reconfigure with new key
    genai.configure(api_key=new_key)
    _update_quota_status(hit=False, key_index=next_index)
    
    return new_key


def _get_cache_key(profile: CandidateProfile, test_config: TestConfig) -> str:
    """Generate cache key for question requests"""
    import hashlib
    key_data = f"{profile.programming_language}|{profile.difficulty}|{test_config.total_questions}"
    return hashlib.md5(key_data.encode()).hexdigest()


def _load_cached_questions(cache_key: str) -> list:
    """Load questions from cache if available and fresh"""
    cache_file = Path(__file__).parent / "data" / "question_cache" / f"{cache_key}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        cache_time = data.get('timestamp', 0)
        current_time = time.time()
        
        # Cache is valid for 24 hours
        if current_time - cache_time < 86400:  # 24 hours
            logging.info(f"Loading {len(data.get('questions', []))} questions from cache")
            return data.get('questions', [])
        else:
            logging.info("Cache expired, regenerating questions")
            return None
    except Exception as e:
        logging.error(f"Error loading cache: {e}")
        return None


def _save_cached_questions(cache_key: str, questions: list) -> None:
    """Save questions to cache"""
    cache_dir = Path(__file__).parent / "data" / "question_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{cache_key}.json"
    
    try:
        cache_data = {
            'timestamp': time.time(),
            'questions': questions,
            'profile_hash': cache_key
        }
        cache_file.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info(f"Cached {len(questions)} questions for future use")
    except Exception as e:
        logging.error(f"Error saving cache: {e}")


def _generate_questions_with_groq(profile: CandidateProfile, test_config: TestConfig) -> list:
    """Generate questions using Groq API"""
    if 'groq' not in AI_PROVIDERS:
        return None
    
    try:
        prompt = f"""Generate exactly {test_config.total_questions} multiple-choice questions for {profile.programming_language} at {profile.difficulty.lower()} difficulty level.

Format as JSON array:
[
  {{
    "id": "q1",
    "topic": "basics",
    "difficulty": "easy",
    "question": "Your question here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer_index": 0
  }}
]

Requirements:
- Questions must be technical and relevant
- Include topics: data structures, algorithms, syntax, best practices
- Mix of difficulty levels
- Clear, unambiguous correct answers
- No duplicate questions"""

        response = AI_PROVIDERS['groq'].chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Check for empty response
        if not raw:
            logging.warning("Groq returned empty response")
            return None
            
        # Try to extract JSON from response
        try:
            questions = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
            else:
                logging.error(f"Groq response parsing failed: {raw[:200]}...")
                return None
        
        if isinstance(questions, list) and len(questions) > 0:
            logging.info(f"Generated {len(questions)} questions with Groq")
            return questions
        else:
            return None
            
    except Exception as e:
        logging.error(f"Groq generation failed: {e}")
        return None


def _generate_questions_with_openrouter(profile: CandidateProfile, test_config: TestConfig) -> list:
    """Generate questions using OpenRouter API"""
    if 'openrouter' not in AI_PROVIDERS:
        return None
    
    try:
        prompt = f"""Generate exactly {test_config.total_questions} multiple-choice questions for {profile.programming_language} at {profile.difficulty.lower()} difficulty level.

Format as JSON array:
[
  {{
    "id": "q1",
    "topic": "basics",
    "difficulty": "easy",
    "question": "Your question here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer_index": 0
  }}
]

Requirements:
- Questions must be technical and relevant
- Include topics: data structures, algorithms, syntax, best practices
- Mix of difficulty levels
- Clear, unambiguous correct answers
- No duplicate questions"""

        response = AI_PROVIDERS['openrouter'].chat.completions.create(
            model="meta-llama/llama-3.2-3b-instruct:free",  # Updated to valid free model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Check for empty response
        if not raw:
            logging.warning("OpenRouter returned empty response")
            return None
            
        # Try to extract JSON from response
        try:
            questions = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
            else:
                logging.error(f"OpenRouter response parsing failed: {raw[:200]}...")
                return None
        
        if isinstance(questions, list) and len(questions) > 0:
            logging.info(f"Generated {len(questions)} questions with OpenRouter")
            return questions
        else:
            return None
            
    except Exception as e:
        logging.error(f"OpenRouter generation failed: {e}")
        return None


def _generate_questions_with_perplexity(profile: CandidateProfile, test_config: TestConfig) -> list:
    """Generate questions using Perplexity API"""
    if 'perplexity' not in AI_PROVIDERS:
        return None
    
    try:
        prompt = f"""Generate exactly {test_config.total_questions} multiple-choice questions for {profile.programming_language} at {profile.difficulty.lower()} difficulty level.

Format as JSON array:
[
  {{
    "id": "q1",
    "topic": "basics",
    "difficulty": "easy",
    "question": "Your question here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer_index": 0
  }}
]

Requirements:
- Questions must be technical and relevant
- Include topics: data structures, algorithms, syntax, best practices
- Mix of difficulty levels
- Clear, unambiguous correct answers
- No duplicate questions"""

        response = AI_PROVIDERS['perplexity'].chat.completions.create(
            model="llama-3.1-sonar-small-128k-online",  # Updated to valid Perplexity model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Check for empty response
        if not raw:
            logging.warning("Perplexity returned empty response")
            return None
            
        # Try to extract JSON from response
        try:
            questions = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
            else:
                logging.error(f"Perplexity response parsing failed: {raw[:200]}...")
                return None
        
        if isinstance(questions, list) and len(questions) > 0:
            logging.info(f"Generated {len(questions)} questions with Perplexity")
            return questions
        else:
            return None
            
    except Exception as e:
        logging.error(f"Perplexity generation failed: {e}")
        return None


def _generate_questions_with_any_provider(profile: CandidateProfile, test_config: TestConfig) -> list:
    """Try all available AI providers in order of preference"""
    
    # Provider priority order
    providers = [
        ('groq', _generate_questions_with_groq),
        ('openrouter', _generate_questions_with_openrouter),
        ('perplexity', _generate_questions_with_perplexity),
        ('gemini', lambda p, t: _generate_mcqs_with_gemini(p, t, recursion_depth=0))
    ]
    
    for provider_name, provider_func in providers:
        try:
            logging.info(f"Trying {provider_name} for question generation...")
            questions = provider_func(profile, test_config)
            
            if questions and len(questions) > 0:
                logging.info(f"Successfully generated {len(questions)} questions using {provider_name}")
                st.info(f"🎯 Questions generated using {provider_name.title()}!")
                return questions
            else:
                logging.warning(f"{provider_name} returned no questions")
                
        except Exception as e:
            logging.error(f"{provider_name} failed: {e}")
            continue
    
    # If all providers fail, use default questions
    logging.warning("All AI providers failed, using default questions")
    return _get_default_questions(profile.programming_language, profile.difficulty)


def _generate_mcqs_with_gemini(profile: CandidateProfile, test_config: TestConfig, recursion_depth: int = 0) -> list:
    # Prevent infinite recursion
    if recursion_depth > 2:
        logging.error("Max recursion depth reached, using default questions")
        return _get_default_questions(profile.programming_language, profile.difficulty)
    
    # Check cache first
    cache_key = _get_cache_key(profile, test_config)
    cached_questions = _load_cached_questions(cache_key)
    if cached_questions:
        return cached_questions
    
    used_hashes = _load_used_hashes()
    syllabus = _load_syllabus(profile.programming_language, profile.difficulty)

    model_name = _get_available_gemini_model()
    if not model_name:
        st.error("❌ No compatible Gemini model found.")
        return []

    prompt = f"""You are an expert assessment designer. Generate exactly {test_config.total_questions} multiple-choice questions for {profile.programming_language} at {profile.difficulty.lower()} difficulty level.

CRITICAL RULES:
- Base EVERY question ONLY on the provided syllabus below. Do NOT invent topics.
- Each question must have 4 options (A, B, C, D)
- Clearly indicate the correct answer index (0-3)
- NEVER repeat questions. Use unique phrasing and concepts.
- Avoid generic questions like "What is Python?" — be specific to syllabus items.

SYLLABUS (use ONLY these topics):
{syllabus}

ANTI-REPETITION EXAMPLES (DO NOT use similar phrasing):
- Bad: "Which keyword defines a function?" → Good: "In Python, which keyword is used to define a function that does not return a value?"
- Bad: "How do you create a list?" → Good: "Which method adds an element to the end of a Python list?"

OUTPUT FORMAT (strict JSON array):
[
  {{
    "id": "q1",
    "topic": "topic_name_from_syllabus",
    "difficulty": "{profile.difficulty.lower()}",
    "question": "specific question text",
    "options": ["option A", "option B", "option C", "option D"],
    "answer_index": 0
  }},
  ...
]"""

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        raw = response.text.strip()
        
        # Log raw response for debugging
        logging.info(f"Gemini raw response: {raw[:200]}...")
        
        if not raw:
            logging.warning("Empty response from Gemini API")
            st.warning("⚠️ Using default questions due to API response issue")
            return _get_default_questions(profile.programming_language, profile.difficulty)
        
        questions = json.loads(raw)
        if not isinstance(questions, list):
            raise ValueError("Response is not a list")
        
        if len(questions) == 0:
            logging.warning("Empty questions list from Gemini")
            st.warning("⚠️ Using default questions due to empty response")
            return _get_default_questions(profile.programming_language, profile.difficulty)
        
        # Validate and fingerprint
        valid = []
        new_hashes = set()
        for q in questions:
            if all(k in q for k in ("id", "question", "options", "answer_index")) and isinstance(q["options"], list) and len(q["options"]) == 4:
                fp = _question_fingerprint(q["question"], q["options"])
                if fp not in used_hashes and fp not in new_hashes:
                    valid.append(q)
                    new_hashes.add(fp)
        
        if len(valid) < test_config.total_questions:
            logging.warning(f"Only {len(valid)} valid questions generated, supplementing with defaults")
            default_questions = _get_default_questions(profile.programming_language, profile.difficulty)
            needed = test_config.total_questions - len(valid)
            for dq in default_questions[:needed]:
                fp = _question_fingerprint(dq["question"], dq["options"])
                if fp not in used_hashes and fp not in new_hashes:
                    valid.append(dq)
                    new_hashes.add(fp)
        
        # Update global hash store
        updated = used_hashes.union(new_hashes)
        _save_used_hashes(updated)
        
        # Cache the successful result
        _save_cached_questions(cache_key, valid[: test_config.total_questions])
        
        return valid[: test_config.total_questions]
        
    except json.JSONDecodeError as e:
        st.error("❌ Failed to parse AI response. Using default questions.")
        logging.error(f"JSON parsing failed: {e}")
        return _get_default_questions(profile.programming_language, profile.difficulty)
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "limit" in error_msg or "exceeded" in error_msg:
            st.warning("⚠️ API quota exceeded. Attempting key rotation...")
            logging.warning(f"API quota exceeded: {e}")
            
            # Try rotating to next API key
            new_key = _handle_quota_exhaustion()
            if new_key:
                st.info("🔄 Retrying with new API key...")
                # Retry with new key (recursive call with new configuration)
                return _generate_mcqs_with_gemini(profile, test_config, recursion_depth + 1)
            else:
                st.warning("⚠️ All API keys exhausted. Using default questions.")
                _update_quota_status(hit=True)
                return _get_default_questions(profile.programming_language, profile.difficulty)
        else:
            st.error("❌ AI service unavailable. Using default questions.")
            logging.error(f"Question generation failed: {e}")
        return _get_default_questions(profile.programming_language, profile.difficulty)


def _get_default_questions(programming_language: str, difficulty: str):
    base = [
        {
            "id": "q1",
            "topic": "basics",
            "difficulty": "easy",
            "question": f"In {programming_language}, which keyword is commonly used to define a function?",
            "options": ["def", "func", "function", "lambda"],
            "answer_index": 0,
        },
        {
            "id": "q2",
            "topic": "types",
            "difficulty": "easy",
            "question": "Which data structure is typically used for key-value mapping?",
            "options": ["List", "Dictionary/Map", "Tuple", "Set"],
            "answer_index": 1,
        },
        {
            "id": "q3",
            "topic": "complexity",
            "difficulty": "medium",
            "question": "What is the average time complexity of hash table lookup?",
            "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
            "answer_index": 0,
        },
        {
            "id": "q4",
            "topic": "oop",
            "difficulty": "medium",
            "question": "What does OOP stand for?",
            "options": [
                "Object-Oriented Programming",
                "Order of Operations Processing",
                "Open Operation Protocol",
                "Object-Optional Pattern",
            ],
            "answer_index": 0,
        },
        {
            "id": "q5",
            "topic": "algorithms",
            "difficulty": "hard",
            "question": "Which algorithm is best suited for finding the shortest path in a weighted graph with non-negative weights?",
            "options": ["DFS", "BFS", "Dijkstra", "Kruskal"],
            "answer_index": 2,
        },
        {
            "id": "q6",
            "topic": "basics",
            "difficulty": "easy",
            "question": "Which operator is commonly used for equality comparison?",
            "options": ["=", "==", "!=", ">>"],
            "answer_index": 1,
        },
        {
            "id": "q7",
            "topic": "control_flow",
            "difficulty": "easy",
            "question": "Which loop is typically used when you know the number of iterations in advance?",
            "options": ["for", "while", "do-while", "foreach-only"],
            "answer_index": 0,
        },
        {
            "id": "q8",
            "topic": "data_structures",
            "difficulty": "easy",
            "question": "Which data structure does NOT allow duplicate elements?",
            "options": ["List", "Array", "Set", "String"],
            "answer_index": 2,
        },
        {
            "id": "q9",
            "topic": "strings",
            "difficulty": "medium",
            "question": "What is the typical time complexity of accessing a character by index in a string?",
            "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
            "answer_index": 0,
        },
        {
            "id": "q10",
            "topic": "algorithms",
            "difficulty": "medium",
            "question": "Which sorting algorithm is generally NOT stable in its classic in-place form?",
            "options": ["Merge sort", "Bubble sort", "Insertion sort", "Quick sort"],
            "answer_index": 3,
        },
        {
            "id": "q11",
            "topic": "algorithms",
            "difficulty": "medium",
            "question": "Binary search requires the input array/list to be:",
            "options": ["Sorted", "Reversed", "Unique", "Random"],
            "answer_index": 0,
        },
        {
            "id": "q12",
            "topic": "complexity",
            "difficulty": "medium",
            "question": "What is the time complexity of iterating over an array of size n once?",
            "options": ["O(1)", "O(log n)", "O(n)", "O(n^2)"],
            "answer_index": 2,
        },
        {
            "id": "q13",
            "topic": "databases",
            "difficulty": "easy",
            "question": "Which SQL clause is used to filter rows?",
            "options": ["WHERE", "GROUP BY", "ORDER BY", "JOIN"],
            "answer_index": 0,
        },
        {
            "id": "q14",
            "topic": "databases",
            "difficulty": "medium",
            "question": "Which SQL statement is used to remove rows from a table?",
            "options": ["DROP", "DELETE", "REMOVE", "TRUNCATE"],
            "answer_index": 1,
        },
        {
            "id": "q15",
            "topic": "oop",
            "difficulty": "easy",
            "question": "Encapsulation primarily means:",
            "options": [
                "Hiding internal state and requiring all interaction through methods",
                "Using multiple inheritance",
                "Writing code without classes",
                "Storing data only in global variables",
            ],
            "answer_index": 0,
        },
        {
            "id": "q16",
            "topic": "oop",
            "difficulty": "medium",
            "question": "Polymorphism allows:",
            "options": [
                "Objects to take multiple forms through a common interface",
                "A program to run without memory",
                "Only one method name per class",
                "Replacing variables with constants",
            ],
            "answer_index": 0,
        },
        {
            "id": "q17",
            "topic": "debugging",
            "difficulty": "easy",
            "question": "Which is typically the FIRST step in debugging a failing program?",
            "options": ["Rewrite the whole code", "Reproduce the issue consistently", "Deploy to production", "Disable all logs"],
            "answer_index": 1,
        },
        {
            "id": "q18",
            "topic": "security",
            "difficulty": "medium",
            "question": "Which practice helps protect user passwords?",
            "options": ["Store in plain text", "Encrypt with reversible key", "Hash with salt", "Email passwords to users"],
            "answer_index": 2,
        },
        {
            "id": "q19",
            "topic": "networks",
            "difficulty": "easy",
            "question": "HTTP is primarily a:",
            "options": ["Programming language", "Network protocol", "Database", "Operating system"],
            "answer_index": 1,
        },
        {
            "id": "q20",
            "topic": "version_control",
            "difficulty": "easy",
            "question": "Git is primarily used for:",
            "options": ["Image editing", "Version control", "Database indexing", "Email automation"],
            "answer_index": 1,
        },
    ]

    level = difficulty.lower()
    if level == "easy":
        picked = [q for q in base if q["difficulty"] == "easy"]
    elif level == "medium":
        picked = [q for q in base if q["difficulty"] in {"easy", "medium"}]
    else:
        picked = base

    if len(picked) < 20:
        picked_ids = {q["id"] for q in picked}
        picked.extend([q for q in base if q["id"] not in picked_ids])

    return picked


def _ensure_test_seeded() -> None:
    if st.session_state.questions is not None:
        return

    profile: CandidateProfile = st.session_state.profile
    test_config: TestConfig = st.session_state.test_config

    questions = _generate_questions_with_any_provider(profile, test_config)

    st.session_state.questions = questions
    st.session_state.current_q = 0
    st.session_state.answers = {}


def _remaining_seconds() -> int:
    if st.session_state.test_started_at is None:
        return 0
    test_config: TestConfig = st.session_state.test_config
    elapsed = int(time.time() - st.session_state.test_started_at)
    return max(0, test_config.duration_seconds - elapsed)


def _finalize_attempt(reason: str) -> None:
    if st.session_state.test_submitted_at is not None:
        return

    st.session_state.test_submitted_at = time.time()

    questions = st.session_state.questions or []
    answers = st.session_state.answers or {}

    correct = 0
    details = []
    for q in questions:
        qid = q["id"]
        selected = answers.get(qid)
        is_correct = selected == q["answer_index"]
        if is_correct:
            correct += 1
        details.append(
            {
                "id": qid,
                "question": q["question"],
                "options": q["options"],
                "correct_index": q["answer_index"],
                "selected_index": selected,
                "is_correct": is_correct,
                "topic": q.get("topic"),
                "difficulty": q.get("difficulty"),
            }
        )

    total = len(questions)
    score_percent = (correct / total * 100.0) if total else 0.0

    profile: CandidateProfile = st.session_state.profile
    test_config: TestConfig = st.session_state.test_config

    started_at = st.session_state.test_started_at
    submitted_at = st.session_state.test_submitted_at
    time_taken_seconds = int((submitted_at or time.time()) - (started_at or time.time()))

    result = {
        "attempt_id": st.session_state.attempt_id,
        "created_at": _now_iso(),
        "finalized_reason": reason,
        "candidate": asdict(profile) if profile else None,
        "test_config": asdict(test_config) if test_config else None,
        "metrics": {
            "total_questions": total,
            "correct": correct,
            "incorrect": max(0, total - correct),
            "percentage": round(score_percent, 2),
            "time_taken_seconds": time_taken_seconds,
        },
        "questions": details,
    }

    st.session_state.result = result
    _persist_result(result)


def _persist_result(result: dict) -> None:
    base = Path(__file__).parent
    out_dir = base / "data" / "results"
    candidates_dir = out_dir / "candidates"
    questions_dir = out_dir / "questions"
    scores_dir = out_dir / "scores"

    for d in (candidates_dir, questions_dir, scores_dir):
        d.mkdir(parents=True, exist_ok=True)

    attempt_id = result.get("attempt_id") or str(uuid.uuid4())

    candidate_file = candidates_dir / f"{attempt_id}.json"
    questions_file = questions_dir / f"{attempt_id}.json"
    score_file = scores_dir / f"{attempt_id}.json"

    candidate_data = {
        "attempt_id": result.get("attempt_id"),
        "created_at": result.get("created_at"),
        "finalized_reason": result.get("finalized_reason"),
        "candidate": result.get("candidate"),
        "test_config": result.get("test_config"),
    }

    questions_data = {
        "attempt_id": result.get("attempt_id"),
        "questions": result.get("questions"),
    }

    score_data = {
        "attempt_id": result.get("attempt_id"),
        "created_at": result.get("created_at"),
        "finalized_reason": result.get("finalized_reason"),
        "metrics": result.get("metrics"),
    }

    candidate_file.write_text(json.dumps(candidate_data, ensure_ascii=False, indent=2), encoding="utf-8")
    questions_file.write_text(json.dumps(questions_data, ensure_ascii=False, indent=2), encoding="utf-8")
    score_file.write_text(json.dumps(score_data, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Automatically combine data after persisting
    _combine_attempt_data(attempt_id, result)


def _combine_attempt_data(attempt_id: str, result: dict) -> None:
    """Combine candidate data, score, and questions into single JSON file"""
    try:
        base = Path(__file__).parent / "data" / "results"
        combined_dir = base / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        
        # Load individual data files
        candidate_file = base / "candidates" / f"{attempt_id}.json"
        score_file = base / "scores" / f"{attempt_id}.json"
        questions_file = base / "questions" / f"{attempt_id}.json"
        
        candidate_data = {}
        if candidate_file.exists():
            candidate_data = json.loads(candidate_file.read_text(encoding="utf-8"))
        
        score_data = {}
        if score_file.exists():
            score_data = json.loads(score_file.read_text(encoding="utf-8"))
        
        questions_data = {}
        if questions_file.exists():
            questions_data = json.loads(questions_file.read_text(encoding="utf-8"))
        
        # Combine all data
        combined_data = {
            "attempt_id": attempt_id,
            "created_at": candidate_data.get("created_at", ""),
            "finalized_reason": candidate_data.get("finalized_reason", ""),
            "candidate": candidate_data.get("candidate", {}),
            "test_config": candidate_data.get("test_config", {}),
            "metrics": score_data.get("metrics", {}),
            "questions": questions_data.get("questions", [])
        }
        
        # Save combined data
        combined_file = combined_dir / f"{attempt_id}.json"
        combined_file.write_text(
            json.dumps(combined_data, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        
    except Exception as e:
        # Log error for debugging but don't break the main flow
        logging.error(f"Error combining attempt data for {attempt_id}: {e}")
        # Continue without combined data - not critical for main functionality


def _inject_custom_css():
    st.markdown("""
    <style>
    /* Modern, Clean & Elegant Design */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: white;
        font-weight: 300;
        letter-spacing: -0.5px;
    }
    
    .card-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a202c;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .form-label {
        font-weight: 500;
        color: #4a5568;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* Clean Input Styling */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: #2d3748;
        font-weight: 400;
        transition: all 0.2s ease;
        font-size: 1rem;
    }
    
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus,
    .stNumberInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    /* Modern Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        cursor: pointer;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a67d8, #6b46c1);
    }
    
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Clean Radio Buttons */
    .stRadio>div {
        background: rgba(255, 255, 255, 0.5);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stRadio label {
        font-weight: 500;
        color: #4a5568;
        padding: 0.5rem 0;
        transition: color 0.2s ease;
    }
    
    .stRadio label:hover {
        color: #667eea;
    }
    
    /* Modern Alerts */
    .stAlert {
        border-radius: 12px;
        font-weight: 500;
        border: none;
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    .stInfo {
        background: rgba(102, 126, 234, 0.1);
        color: #4c51bf;
        border-left: 4px solid #667eea;
    }
    
    .stSuccess {
        background: rgba(72, 187, 120, 0.1);
        color: #2f855a;
        border-left: 4px solid #48bb78;
    }
    
    .stWarning {
        background: rgba(237, 137, 54, 0.1);
        color: #c05621;
        border-left: 4px solid #ed8936;
    }
    
    .stError {
        background: rgba(245, 101, 101, 0.1);
        color: #c53030;
        border-left: 4px solid #f56565;
    }
    
    /* Hide Streamlit Branding */
    .stDeployButton {
        display: none;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .card-container {
            padding: 1.5rem;
            margin: 0.5rem 0;
        }
        
        .main-header {
            padding: 1rem 0;
            font-size: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def _render_header() -> None:
    _inject_custom_css()
    st.markdown("""
    <style>
    .stApp > div {
        padding-top: 0;
    }
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; margin-bottom:3rem;'>
        <div style='font-size:3rem; font-weight:700; background:linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.5rem;'>
            🚀 Online Assessment System
        </div>
        <div style='color:#94a3b8; font-size:1.1rem; font-weight:400;'>Enterprise-grade AI-powered assessments</div>
    </div>
    """, unsafe_allow_html=True)


def _render_progress_indicator(step: int):
    steps = ["Candidate Details", "Rules", "Assessment", "Result"]
    icons = ["👤", "📋", "📝", "🎉"]
    cols = st.columns(len(steps))
    for i, (col, s, icon) in enumerate(zip(cols, steps, icons)):
        with col:
            if i <= step:
                color = "#60a5fa"
                bg = "rgba(96, 165, 250, 0.15)"
                border = "1px solid rgba(96, 165, 250, 0.3)"
            else:
                color = "#64748b"
                bg = "rgba(100, 116, 139, 0.1)"
                border = "1px solid rgba(100, 116, 139, 0.2)"
            st.markdown(f"""
            <div style='text-align:center; padding:0.75rem; background:{bg}; border:{border}; border-radius:12px; margin-bottom:0.5rem; transition:all 0.3s;'>
                <div style='font-size:1.8rem; margin-bottom:0.25rem;'>{icon}</div>
                <div style='font-weight:600; color:{color}; font-size:0.85rem; letter-spacing:-0.01em;'>{s}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_registration() -> None:
    _render_progress_indicator(0)
    
    st.markdown("""
    <div class="main-header">
        <h1>👤 Candidate Registration</h1>
        <p>Please provide your details to begin the assessment</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("registration_form"):
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        
        # Personal Information
        st.markdown('<h2 class="section-title">Personal Information</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", key="reg_name", placeholder="John Doe")
            branch = st.text_input("Branch", key="reg_branch", placeholder="Computer Science")
        with col2:
            passing_year = st.number_input("Passing Year", min_value=1980, max_value=2100, value=2025, step=1, key="reg_passing_year")
            university = st.text_input("University", key="reg_university", placeholder="MIT University")

        # Assessment Preferences
        st.markdown('<h2 class="section-title" style="margin-top: 2rem;">Assessment Preferences</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            programming_language = st.selectbox("Programming Language", ["Python", "Java", "JavaScript", "C++", "SQL"], key="reg_lang")
        with col2:
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key="reg_difficulty")
        with col3:
            assessment_type = st.selectbox(
                "Assessment Type",
                ["MCQ only", "MCQ + Coding", "MCQ + Debugging", "Conceptual / Aptitude"],
                key="reg_type",
            )

        # Test Configuration Info
        st.markdown("""
        <div style="background: rgba(102, 126, 234, 0.1); border-radius: 12px; padding: 1.5rem; margin: 2rem 0;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 2rem;">📋</div>
                <div>
                    <div style="font-weight: 600; color: #667eea; margin-bottom: 0.25rem;">Assessment Details</div>
                    <div style="color: #4a5568; font-size: 0.9rem;">20 questions • 20 minutes • Fixed duration</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Submit Button
        st.markdown('</div>', unsafe_allow_html=True)  # Close card-container
        submitted = st.form_submit_button("Continue →")

    if submitted:
        if not name.strip() or not branch.strip() or not university.strip():
            st.error("Please fill all required fields.")
            return

        st.session_state.profile = CandidateProfile(
            name=name.strip(),
            branch=branch.strip(),
            passing_year=int(passing_year),
            university=university.strip(),
            programming_language=programming_language,
            difficulty=difficulty,
            assessment_type=assessment_type,
        )
        st.session_state.test_config = TestConfig(
            total_questions=20,
            duration_seconds=20 * 60,
        )
        st.session_state.attempt_id = str(uuid.uuid4())
        st.session_state.accepted_rules = False
        st.session_state.system_check_passed = True
        st.session_state.questions = None
        st.session_state.current_q = 0
        st.session_state.answers = {}
        st.session_state.test_started_at = None
        st.session_state.test_submitted_at = None
        st.session_state.result = None

        _set_page("rules")


def _render_rules() -> None:
    _render_progress_indicator(1)
    
    st.markdown("""
    <div class="main-header">
        <h1>📋 Assessment Rules</h1>
        <p>Please read and agree to the following guidelines</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">📌 Assessment Guidelines</div>', unsafe_allow_html=True)
    
    # Guidelines Grid
    guidelines = [
        ("⏱️", "Duration", "20 minutes total"),
        ("📝", "Questions", "20 multiple choice questions"),
        ("✅", "No Negative Marking", "Wrong answers won't reduce your score"),
        ("🔄", "Complete in One Sitting", "Must finish without interruptions"),
        ("👁️", "Stay on This Page", "Navigation away may affect your test")
    ]
    
    for icon, title, description in guidelines:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 1rem; padding: 1rem; background: rgba(102, 126, 234, 0.05); border-radius: 12px; margin-bottom: 1rem;">
            <div style="font-size: 1.5rem;">{icon}</div>
            <div>
                <div style="font-weight: 600; color: #2d3748;">{title}</div>
                <div style="color: #4a5568;">{description}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    accepted = st.checkbox("I have read and agree to the rules", key="rules_accept")
    
    if accepted:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(72, 187, 120, 0.1), rgba(34, 197, 94, 0.1)); border: 1px solid rgba(72, 187, 120, 0.3); border-radius: 16px; padding: 2rem; margin: 2rem 0; text-align: center;">
            <div style="color: #2f855a; font-weight: 700; font-size: 1.3rem; margin-bottom: 1rem;">🚀 Ready to Start</div>
            <div style="color: #2f855a; font-size: 1rem; margin-bottom: 1.5rem;">You have read and agreed to the assessment rules.</div>
            <div style="color: #4a5568; font-size: 0.9rem;">The assessment will begin once you click the button below.</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back", key="confirm_back"):
                st.session_state.rules_confirmed = False
                st.rerun()
        with col2:
            if st.button("Start Assessment →", key="confirm_start", help="Begin the assessment now"):
                st.session_state.accepted_rules = True
                st.session_state.rules_confirmed = True
                st.session_state.system_check_passed = True
                st.session_state.test_started_at = time.time()
                _ensure_test_seeded()
                _set_page("test")
    
    st.markdown('</div>', unsafe_allow_html=True)


def _render_test() -> None:
    if not st.session_state.profile or not st.session_state.accepted_rules:
        st.error("Test cannot start before completing registration and rules acceptance.")
        if st.button("Go to Registration", key="test_to_reg"):
            _set_page("registration")
        return

    _ensure_test_seeded()

    remaining = _remaining_seconds()
    if remaining <= 0:
        _finalize_attempt("timeout")
        _set_page("result")
        return

    _render_progress_indicator(2)
    st.subheader("📝 Assessment")

    # Timer and progress
    start_ms = int((st.session_state.test_started_at or time.time()) * 1000)
    duration_ms = int((st.session_state.test_config.duration_seconds if st.session_state.test_config else 0) * 1000)
    components.html(
        f"""
        <div style="font-size: 16px; font-weight: 600; margin-bottom: 8px;">Time Remaining: <span id="timer">--:--</span></div>
        <script>
        (function() {{
            const startMs = {start_ms};
            const durationMs = {duration_ms};

            function pad(n) {{ return String(n).padStart(2, '0'); }}

            function tick() {{
                const now = Date.now();
                const remaining = Math.max(0, (startMs + durationMs) - now);
                const totalSeconds = Math.floor(remaining / 1000);
                const mins = Math.floor(totalSeconds / 60);
                const secs = totalSeconds % 60;
                const el = document.getElementById('timer');
                if (el) el.textContent = `${{pad(mins)}}:${{pad(secs)}}`;

                if (remaining <= 0) {{
                    window.location.reload();
                }}
            }}

            tick();
            setInterval(tick, 1000);
        }})();
        </script>
        """,
        height=50,
    )

    questions = st.session_state.questions or []
    idx = int(st.session_state.current_q or 0)

    # Handle case where no questions are available
    if len(questions) == 0:
        st.error("❌ No questions available. Please try again or contact support.")
        st.info("🔄 This might be due to API quota limits. The system will use default questions.")
        # Try to generate questions with default fallback
        profile = st.session_state.profile
        test_config = st.session_state.test_config
        default_questions = _get_default_questions(profile.programming_language, profile.difficulty)
        st.session_state.questions = default_questions[:test_config.total_questions]
        st.rerun()
        return

    # Only redirect to result if we have questions AND we've gone past the last one
    if len(questions) > 0 and idx >= len(questions):
        _finalize_attempt("completed")
        _set_page("result")
        return

    q = questions[idx]
    qid = q["id"]

    # Progress bar
    progress = (idx + 1) / len(questions)
    st.markdown(f"""
    <div style='background:rgba(100, 116, 139, 0.2); border-radius:12px; height:10px; margin-bottom:1.5rem; overflow:hidden;'>
        <div style='background:linear-gradient(90deg, #3b82f6, #8b5cf6); width:{progress*100}%; height:100%; border-radius:12px; transition:width 0.5s ease;'></div>
    </div>
    <div style='display:flex; justify-content:space-between; margin-bottom:2rem; color:#94a3b8; font-size:0.95rem; font-weight:500;'>
        <span>Question {idx + 1} of {len(questions)}</span>
        <span>{progress*100:.0f}% Complete</span>
    </div>
    """, unsafe_allow_html=True)

    # Question card with clean, focused design
    with st.container():
        st.markdown(f"""
        <div style='background:linear-gradient(135deg, rgba(30, 41, 59, 0.98), rgba(51, 65, 85, 0.98)); border:1px solid rgba(148, 163, 184, 0.3); border-left:4px solid #3b82f6; padding:2.5rem; border-radius:20px; margin-bottom:2rem; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);'>
            <div style='display: flex; align-items: center; gap:1.5rem; margin-bottom:1.5rem;'>
                <div style='background:linear-gradient(135deg, #3b82f6, #8b5cf6); color:white; font-size:1.3rem; font-weight:700; width:3.5rem; height:3.5rem; border-radius:50%; display:flex; align-items:center; justify-content:center; box-shadow:0 4px 12px rgba(59, 130, 246, 0.3);'>{idx + 1}</div>
                <div>
                    <div style='font-weight:700; color:#f1f5f9; font-size:1.2rem; margin-bottom:0.5rem;'>Question {idx + 1}</div>
                    <div style='color:#60a5fa; font-size:0.9rem; font-weight:600;'>Select the correct answer</div>
                </div>
            </div>
            <div style='font-size:1.3rem; color:#f8fafc; line-height:1.7; font-weight:500;'>{q["question"]}</div>
        </div>
        """, unsafe_allow_html=True)

        existing = st.session_state.answers.get(qid)
        
        # Custom radio buttons with better visual feedback
        st.markdown("**Select your answer:**")
        
        for i, option in enumerate(q["options"]):
            is_selected = existing == i
            
            # Clean option styling
            if is_selected:
                border_color = "#3b82f6"
                bg_color = "rgba(59, 130, 246, 0.1)"
                radio_icon = "🔘"
                text_color = "#3b82f6"
            else:
                border_color = "rgba(148, 163, 184, 0.3)"
                bg_color = "rgba(15, 23, 42, 0.8)"
                radio_icon = "⭕"
                text_color = "#e2e8f0"
            
            # Clean option layout
            col1, col2 = st.columns([0.08, 0.92])
            with col1:
                st.markdown(f"""
                <div style='display: flex; align-items: center; justify-content: center; margin-top: 0.5rem;'>
                    <span style='font-size: 1.2rem;'>{radio_icon}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                button_key = f"q_{qid}_opt_{i}"
                
                # Clean button styling
                st.markdown(f"""
                <style>
                div[data-testid="stVerticalBlock"] > div:has(button[data-testid*="{button_key}"]) {{
                    margin-bottom: 0.75rem !important;
                }}
                
                button[data-testid*="{button_key}"] {{
                    background: {bg_color} !important;
                    border: 2px solid {border_color} !important;
                    border-radius: 12px !important;
                    padding: 1rem 1.25rem !important;
                    color: {text_color} !important;
                    font-weight: 500 !important;
                    text-align: left !important;
                    width: 100% !important;
                    transition: all 0.2s ease !important;
                    cursor: pointer !important;
                }}
                
                button[data-testid*="{button_key}"]:hover {{
                    background: rgba(59, 130, 246, 0.15) !important;
                    border-color: #3b82f6 !important;
                    color: #3b82f6 !important;
                    transform: translateY(-1px) !important;
                }}
                </style>
                """, unsafe_allow_html=True)
                
                if st.button(option, key=button_key, help="Click to select this answer"):
                    st.session_state.answers[qid] = i
                    st.rerun()
        
        # Clean selection feedback
        if existing is not None:
            st.markdown(f"""
            <div style='background:rgba(34, 197, 94, 0.1); border:1px solid rgba(34, 197, 94, 0.3); border-radius:12px; padding:1rem 1.25rem; margin-top:1.5rem;'>
                <div style='display: flex; align-items: center; gap:0.75rem;'>
                    <div style='font-size: 1.2rem;'>✅</div>
                    <div>
                        <div style='color:#86efac; font-weight:600; font-size:0.95rem;'>Selected: {q['options'][existing]}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Clean navigation
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Submit Test", key="test_submit", help="Complete the assessment and view your results"):
            _finalize_attempt("manual_submit")
            _set_page("result")
    with col2:
        # Simple progress indicator
        st.markdown(f"""
        <div style='text-align: center; padding: 0.75rem;'>
            <div style='color:#60a5fa; font-weight:600; font-size:1rem;'>Question {idx + 1} of {len(questions)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        btn_label = "Next →" if idx < len(questions) - 1 else "Finish"
        if st.button(btn_label, key=f"test_nav_{idx}", help="Go to next question"):
            st.session_state.current_q = idx + 1
            st.rerun()


def _render_result() -> None:
    _render_progress_indicator(3)
    st.subheader("🎉 Assessment Result")

    result = st.session_state.result
    if not result:
        st.error("No result found.")
        if st.button("Go to Registration", key="result_to_reg"):
            _set_page("registration")
        return

    metrics = result.get("metrics", {})
    score = metrics.get("percentage", 0.0)

    # Score header
    st.markdown(f"""
    <div style='text-align:center; margin-bottom:3rem; padding:2rem; background:linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(51, 65, 85, 0.8)); border:1px solid rgba(148, 163, 184, 0.1); border-radius:20px; backdrop-filter:blur(10px);'>
        <div style='font-size:4rem; font-weight:700; background:linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.5rem; letter-spacing:-0.02em;'>{score:.0f}%</div>
        <div style='color:#94a3b8; font-size:1.2rem; font-weight:500;'>Your Score</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:2rem; font-weight:700; color:#22c55e;'>{metrics.get('correct', 0)}</div>
            <div style='color:#94a3b8; font-size:0.9rem; font-weight:500;'>Correct</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:2rem; font-weight:700; color:#ef4444;'>{metrics.get('incorrect', 0)}</div>
            <div style='color:#94a3b8; font-size:0.9rem; font-weight:500;'>Incorrect</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:2rem; font-weight:700; color:#3b82f6;'>{metrics.get('total_questions', 0)}</div>
            <div style='color:#94a3b8; font-size:0.9rem; font-weight:500;'>Total</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:2rem; font-weight:700; color:#a855f7;'>{metrics.get('time_taken_seconds', 0)}s</div>
            <div style='color:#94a3b8; font-size:0.9rem; font-weight:500;'>Time</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📄 Test Details")
    st.info(f"Attempt ID: `{result.get('attempt_id')}` | Finalized: {result.get('finalized_reason')}")
    
    st.markdown("---")
    if st.button("🔄 Start New Attempt", key="result_restart"):
        for k in [
            "page",
            "profile",
            "accepted_rules",
            "system_check_passed",
            "attempt_id",
            "test_config",
            "questions",
            "current_q",
            "answers",
            "test_started_at",
            "test_submitted_at",
            "result",
        ]:
            if k in st.session_state:
                del st.session_state[k]
        _init_state()
        _set_page("registration")


def main() -> None:
    _init_state()
    _render_header()

    page = st.session_state.page
    if page == "registration":
        _render_registration()
    elif page == "rules":
        _render_rules()
    elif page == "test":
        _render_test()
    else:
        _render_result()


if __name__ == "__main__":
    main()
