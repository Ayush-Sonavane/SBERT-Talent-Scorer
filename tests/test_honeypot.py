import pytest
from src.honeypot import is_honeypot

def test_valid_candidate():
    candidate = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "years_of_experience": 5
        },
        "career_history": [
            {
                "duration_months": 36,
                "is_current": False
            },
            {
                "duration_months": 24,
                "is_current": True
            }
        ],
        "skills": [
            {
                "name": "Python",
                "proficiency": "expert",
                "duration_months": 48
            }
        ]
    }
    assert is_honeypot(candidate) is False

def test_honeypot_negative_durations():
    # End date before start date would result in negative duration or duration_months < 0
    candidate = {
        "candidate_id": "CAND_0000002",
        "profile": {
            "years_of_experience": 5
        },
        "career_history": [
            {
                "duration_months": -10,
                "is_current": False
            }
        ],
        "skills": []
    }
    assert is_honeypot(candidate) is True

def test_honeypot_zero_duration_expert_skills():
    # "Expert" proficiency with 0 duration months for many skills
    candidate = {
        "candidate_id": "CAND_0000003",
        "profile": {
            "years_of_experience": 5
        },
        "career_history": [
            {
                "duration_months": 60,
                "is_current": True
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 0},
            {"name": "PyTorch", "proficiency": "expert", "duration_months": 0},
            {"name": "ML", "proficiency": "expert", "duration_months": 0},
            {"name": "NLP", "proficiency": "expert", "duration_months": 0},
            {"name": "BERT", "proficiency": "expert", "duration_months": 0},
            {"name": "Transformers", "proficiency": "expert", "duration_months": 0},
            {"name": "LangChain", "proficiency": "expert", "duration_months": 0},
            {"name": "Vector DB", "proficiency": "expert", "duration_months": 0},
            {"name": "SQL", "proficiency": "expert", "duration_months": 0}
        ]
    }
    assert is_honeypot(candidate) is True

def test_honeypot_duration_vs_experience_mismatch():
    # 10 years experience claimed, but sum of durations is only 3 months
    candidate = {
        "candidate_id": "CAND_0000004",
        "profile": {
            "years_of_experience": 10
        },
        "career_history": [
            {
                "duration_months": 3,
                "is_current": False
            }
        ],
        "skills": []
    }
    assert is_honeypot(candidate) is True
