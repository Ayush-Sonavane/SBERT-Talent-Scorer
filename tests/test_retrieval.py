import pytest
from src.retrieval import matches_high_recall_keywords, is_service_company
from src.honeypot import is_honeypot

def test_retrieval_keeps_valid_candidate():
    candidate = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "years_of_experience": 5,
            "headline": "Senior ML Engineer",
            "current_title": "Machine Learning Engineer"
        },
        "career_history": [
            {"company": "TechStart", "title": "ML Engineer", "duration_months": 24}
        ],
        "skills": [{"name": "PyTorch", "proficiency": "expert", "duration_months": 24}]
    }
    # Should not be a honeypot
    assert not is_honeypot(candidate)
    # Should have a non-service career step
    assert not all(is_service_company(job.get("company", "")) for job in candidate["career_history"])
    # Should match high-recall keywords
    assert matches_high_recall_keywords(candidate)

def test_retrieval_excludes_consulting_only():
    candidate = {
        "candidate_id": "CAND_0000003",
        "profile": {
            "years_of_experience": 6,
            "headline": "Software Engineer",
            "current_title": "Senior Consultant"
        },
        "career_history": [
            {"company": "Tata Consultancy Services", "title": "Developer", "duration_months": 36},
            {"company": "Infosys", "title": "Senior Consultant", "duration_months": 36}
        ],
        "skills": [{"name": "Python", "proficiency": "expert", "duration_months": 60}]
    }
    # Should be identified as consulting-only
    assert all(is_service_company(job.get("company", "")) for job in candidate["career_history"])

def test_retrieval_keeps_consulting_then_product():
    candidate = {
        "candidate_id": "CAND_0000004",
        "profile": {
            "years_of_experience": 6,
            "headline": "ML Engineer",
            "current_title": "Lead ML Engineer"
        },
        "career_history": [
            {"company": "Infosys", "title": "Developer", "duration_months": 36},
            {"company": "AI Product Corp", "title": "ML Engineer", "duration_months": 36}
        ],
        "skills": [{"name": "PyTorch", "proficiency": "expert", "duration_months": 36}]
    }
    # Should not be identified as consulting-only
    assert not all(is_service_company(job.get("company", "")) for job in candidate["career_history"])
    assert matches_high_recall_keywords(candidate)

def test_retrieval_excludes_unrelated_roles():
    candidate = {
        "candidate_id": "CAND_0000005",
        "profile": {
            "years_of_experience": 6,
            "headline": "Marketing Manager",
            "current_title": "Marketing Manager"
        },
        "career_history": [
            {"company": "RetailCorp", "title": "Marketing Associate", "duration_months": 36},
            {"company": "SalesCorp", "title": "Marketing Manager", "duration_months": 36}
        ],
        "skills": [{"name": "SEO", "proficiency": "expert", "duration_months": 60}]
    }
    # Should not match high-recall keywords
    assert not matches_high_recall_keywords(candidate)

def test_retrieval_excludes_substring_traps():
    candidate = {
        "candidate_id": "CAND_0000006",
        "profile": {
            "years_of_experience": 5,
            "headline": "Web Developer",
            "current_title": "Frontend Engineer"
        },
        "career_history": [
            {"company": "WebCorp", "title": "Developer", "description": "Maintain databases and write HTML and CSS.", "duration_months": 60}
        ],
        "skills": [{"name": "HTML", "proficiency": "expert", "duration_months": 48}]
    }
    # Should not match high-recall keywords on titles (only description contains 'maintain' which doesn't match 'main' title keywords, and web developer/frontend engineer don't match AI keywords)
    assert not matches_high_recall_keywords(candidate)
