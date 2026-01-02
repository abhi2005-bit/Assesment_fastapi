import json
import os
from pathlib import Path

def combine_candidate_data():
    """Combine candidate details, scores, and questions into single JSON files"""
    
    # Base directory
    base_dir = Path(__file__).parent / "data" / "results"
    candidates_dir = base_dir / "candidates"
    scores_dir = base_dir / "scores" 
    questions_dir = base_dir / "questions"
    combined_dir = base_dir / "combined"
    
    # Create combined directory if it doesn't exist
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all attempt IDs from candidates directory
    candidate_files = list(candidates_dir.glob("*.json"))
    
    for candidate_file in candidate_files:
        attempt_id = candidate_file.stem
        
        # Load candidate data
        candidate_data = {}
        if candidate_file.exists():
            candidate_data = json.loads(candidate_file.read_text(encoding="utf-8"))
        
        # Load score data
        score_file = scores_dir / f"{attempt_id}.json"
        score_data = {}
        if score_file.exists():
            score_data = json.loads(score_file.read_text(encoding="utf-8"))
        
        # Load questions data
        questions_file = questions_dir / f"{attempt_id}.json"
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
        
        print(f"✅ Combined data saved for attempt: {attempt_id}")

if __name__ == "__main__":
    combine_candidate_data()
