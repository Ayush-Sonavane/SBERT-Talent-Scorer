import pytest
from src.scorer import compute_title_fit, compute_experience_fit, compute_skill_fit, compute_behavioral_modifier, score_candidate

def test_title_fit():
    assert compute_title_fit("Senior AI Engineer") == 1.0
    assert compute_title_fit("Machine Learning Engineer") == 1.0
    assert compute_title_fit("Data Scientist") == 0.7
    assert compute_title_fit("Frontend Developer") == 0.0

def test_experience_fit():
    assert compute_experience_fit(7) == 1.0   # 5-9 sweet spot
    assert compute_experience_fit(10) == 0.8  # slightly outside
    assert compute_experience_fit(1) == 0.3   # way too junior

def test_skill_fit():
    # Candidate with vector DB, Python, and NDCG/eval metrics
    skills = [
        {"name": "Python", "proficiency": "expert", "duration_months": 36},
        {"name": "Pinecone", "proficiency": "expert", "duration_months": 24},
        {"name": "NDCG", "proficiency": "intermediate", "duration_months": 12}
    ]
    score = compute_skill_fit(skills)
    assert score >= 0.75  # Should cover multiple must-have categories

def test_behavioral_modifier():
    # Active, responsive candidate
    active_cand = {
        "redrob_signals": {
            "recruiter_response_rate": 0.9,
            "open_to_work_flag": True,
            "notice_period_days": 15,
            "interview_completion_rate": 0.85
        }
    }
    # Passive, unresponsive candidate
    passive_cand = {
        "redrob_signals": {
            "recruiter_response_rate": 0.05,
            "open_to_work_flag": False,
            "notice_period_days": 90,
            "interview_completion_rate": 0.10
        }
    }
    mod_active = compute_behavioral_modifier(active_cand)
    mod_passive = compute_behavioral_modifier(passive_cand)
    
    assert mod_active > 1.0
    assert mod_passive < 0.9
    assert mod_active > mod_passive

def test_composite_scoring():
    candidate = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "current_title": "Senior AI Engineer",
            "years_of_experience": 6
        },
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 36},
            {"name": "Pinecone", "proficiency": "expert", "duration_months": 24},
            {"name": "NDCG", "proficiency": "intermediate", "duration_months": 12}
        ],
        "redrob_signals": {
            "recruiter_response_rate": 0.9,
            "open_to_work_flag": True,
            "notice_period_days": 15,
            "interview_completion_rate": 0.85
        }
    }
    score = score_candidate(candidate, semantic_similarity=0.85)
    assert 0.0 <= score <= 1.0
    assert score > 0.7  # Strong candidate should have high score
