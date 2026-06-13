#!/usr/bin/env python3
import os
import sys
import argparse
import csv
import glob
from src.utils import load_docx_text, stream_jsonl
from src.honeypot import is_honeypot
from src.retrieval import matches_high_recall_keywords, is_service_company
from src.semantic import SemanticMatcher
from src.scorer import score_candidate

def parse_args():
    parser = argparse.ArgumentParser(description="Rank candidates for Senior AI Engineer.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to output submission.csv")
    return parser.parse_args()

def find_file_recursively(filename):
    matches = glob.glob(f"**/{filename}", recursive=True)
    return matches[0] if matches else None

def generate_recruiter_reasoning(cand, score, semantic_score):
    profile = cand.get("profile", {})
    title = profile.get("current_title", "Software Engineer")
    years = profile.get("years_of_experience", 0)
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    base = f"{title} with {years:.1f} yrs experience"
    
    vector_dbs = {"pinecone", "weaviate", "qdrant", "milvus", "faiss"}
    match_db = None
    match_embed = None
    match_metric = None
    
    for s in skills:
        s_name = s.get("name", "").lower()
        if not s_name:
            continue
        if not match_db and any(db in s_name for db in vector_dbs):
            match_db = s.get("name")
        if not match_embed and any(em in s_name for em in ["embed", "sentence-transformer", "bert"]):
            match_embed = s.get("name")
        if not match_metric and any(m in s_name for m in ["ndcg", "mrr", "map", "evaluation"]):
            match_metric = s.get("name")
            
    matching_skills = [s for s in [match_db, match_embed, match_metric] if s]

    skill_part = ""
    if matching_skills:
        skill_part = f"; matches on {', '.join(matching_skills[:2])}"
        
    resp_rate = signals.get("recruiter_response_rate")
    notice = signals.get("notice_period_days")
    
    signal_part = ""
    if resp_rate is not None:
        signal_part = f"; {int(resp_rate * 100)}% response rate"
    if notice is not None and notice <= 30:
        signal_part += f", quick joiner ({notice}d notice)"
        
    fit_desc = "Strong candidate for dense retrieval & ranking pipelines."
    if semantic_score > 0.8:
        fit_desc = "Excellent match for Series A intelligence layer."
    elif score > 0.8:
        fit_desc = "Highly active candidate with solid ML background."
        
    return f"{base}{skill_part}{signal_part}. {fit_desc}"

def main():
    args = parse_args()
    
    if not os.path.exists(args.candidates):
        print(f"Error: Candidate file '{args.candidates}' not found.")
        sys.exit(1)
        
    jd_path = find_file_recursively("job_description.docx")
    jd_fallback_text = (
        "Senior AI Engineer. Own retrieval and intelligence layer. "
        "Experience with dense embeddings, sentence-transformers, vector search (FAISS, Milvus, Qdrant), "
        "ranking evaluation frameworks (NDCG, MAP), and Python. Nice to have: LLM fine-tuning."
    )
    
    if not jd_path or not os.path.exists(jd_path):
        print("Warning: job_description.docx not found. Using default fallback JD requirements.")
        jd_text = jd_fallback_text
    else:
        print(f"Loading job description from '{jd_path}'...")
        try:
            jd_text = load_docx_text(jd_path)
        except Exception as e:
            print(f"Warning: Failed to load '{jd_path}' ({e}). Using default fallback JD requirements.", file=sys.stderr)
            jd_text = jd_fallback_text
        
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "all-MiniLM-L6-v2")
    if not os.path.exists(model_path):
        print(f"Error: Local model weights not found at '{model_path}'. Run download_model.py first.")
        sys.exit(1)
        
    matcher = SemanticMatcher(model_path=model_path)
    print(f"Initializing semantic matcher ({os.path.basename(matcher.model_file)})...")
    matcher.fit_jd(jd_text)
    
    print("Running Stage 1 Filtering...")
    retrieved = []
    total_count = 0
    
    for cand in stream_jsonl(args.candidates):
        total_count += 1
        
        if is_honeypot(cand):
            continue
            
        profile = cand.get("profile", {})
        years = profile.get("years_of_experience")
        if years is not None and (years < 2 or years > 15):
            continue

        career_history = cand.get("career_history", [])
        if len(career_history) > 0:
            all_service = True
            for job in career_history:
                company = job.get("company", "")
                if not is_service_company(company):
                    all_service = False
                    break
            if all_service:
                continue
                
        if not matches_high_recall_keywords(cand):
            continue
            
        retrieved.append(cand)
        
    print(f"Parsed {total_count} candidates. Retrieved {len(retrieved)} after Stage 1 filter.")
    
    print("Running Stage 2 Scoring...")
    scored_candidates = []
    
    if retrieved:
        print(f"Batch encoding {len(retrieved)} candidates...")
        sem_scores = matcher.score_candidates_batch(retrieved)
        
        for cand, sem_score in zip(retrieved, sem_scores):
            final_score = score_candidate(cand, semantic_similarity=sem_score)
            scored_candidates.append({
                "cand": cand,
                "score": final_score,
                "semantic_score": sem_score
            })
        
    scored_candidates.sort(key=lambda x: (-x["score"], x["cand"]["candidate_id"]))
    
    print(f"Writing top 100 shortlist to '{args.out}'...")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) if os.path.dirname(args.out) else ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            
            for rank, item in enumerate(scored_candidates[:100], 1):
                cand = item["cand"]
                score = item["score"]
                sem_score = item["semantic_score"]
                reasoning = generate_recruiter_reasoning(cand, score, sem_score)
                writer.writerow([cand["candidate_id"], rank, f"{score:.4f}", reasoning])
        print("Shortlist ranking completed successfully.")
    except Exception as e:
        print(f"Error: Failed to write shortlist to '{args.out}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
