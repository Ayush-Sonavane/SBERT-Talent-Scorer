import pytest
from src.semantic import SemanticMatcher

def test_semantic_matching():
    # Initialize the matcher using our local model
    # (Since Phase 2 downloaded it to models/all-MiniLM-L6-v2, this will run locally)
    matcher = SemanticMatcher(model_path="models/all-MiniLM-L6-v2")
    
    jd_text = (
        "Senior AI Engineer to own the intelligence layer. Must have experience with "
        "production embeddings, vector databases like Pinecone/Weaviate/Milvus, hybrid search, "
        "and evaluation frameworks like NDCG/MAP. Experience with fine-tuning LLMs is a plus."
    )
    
    # Candidate 1: Strong AI Engineer matching JD
    cand_1 = {
        "profile": {
            "headline": "Senior AI & Search Engineer",
            "summary": "Building production search and retrieval systems using dense embeddings and Milvus vector DB. Fine-tuned Llama 3 models."
        },
        "career_history": [
            {"title": "AI Engineer", "description": "Built vector search retrieval pipeline using PyTorch, FAISS, and Qdrant. Evaluated with NDCG@10."}
        ]
    }
    
    # Candidate 2: Unrelated role (e.g. Frontend developer)
    cand_2 = {
        "profile": {
            "headline": "Senior Frontend Developer",
            "summary": "Experienced React and TypeScript engineer. Building responsive user interfaces and optimizing web performance."
        },
        "career_history": [
            {"title": "UI Engineer", "description": "Led frontend architecture, migrated codebase to Next.js, and improved accessibility scores."}
        ]
    }
    
    # Encode JD
    matcher.fit_jd(jd_text)
    
    # Score candidates
    score_1 = matcher.score_candidate(cand_1)
    score_2 = matcher.score_candidate(cand_2)
    
    print(f"Candidate 1 (AI Engineer) semantic score: {score_1:.4f}")
    print(f"Candidate 2 (Frontend) semantic score: {score_2:.4f}")
    
    # Candidate 1 must have higher similarity to JD than Candidate 2
    assert score_1 > score_2
    assert 0.0 <= score_1 <= 1.0
    assert 0.0 <= score_2 <= 1.0

def test_semantic_matching_batch():
    matcher = SemanticMatcher(model_path="models/all-MiniLM-L6-v2")
    jd_text = "Senior AI Engineer"
    matcher.fit_jd(jd_text)
    
    candidates = [
        {
            "profile": {"headline": "AI Engineer", "summary": "Dense retrieval models"},
            "career_history": [],
            "skills": []
        },
        {
            "profile": {"headline": "UI Designer", "summary": "Figma layout design"},
            "career_history": [],
            "skills": []
        }
    ]
    
    scores = matcher.score_candidates_batch(candidates)
    assert len(scores) == 2
    assert scores[0] > scores[1]
    assert 0.0 <= scores[0] <= 1.0
    assert 0.0 <= scores[1] <= 1.0
