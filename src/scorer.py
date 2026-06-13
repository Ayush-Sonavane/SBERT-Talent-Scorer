def compute_title_fit(title: str) -> float:
    if not title:
        return 0.0
    title_lower = title.lower()
    
    unrelated_keywords = ["frontend", "front-end", "ui", "ux", "designer", "graphic", "marketing", "hr", "writer", "sales", "recruiter", "accountant"]
    for kw in unrelated_keywords:
        if kw in title_lower:
            return 0.0
            
    core_keywords = ["ai engineer", "ml engineer", "machine learning engineer", "nlp engineer", "search engineer", "information retrieval", "research engineer"]
    for kw in core_keywords:
        if kw in title_lower:
            return 1.0
            
    adjacent_keywords = ["data scientist", "computer vision engineer", "deep learning engineer"]
    for kw in adjacent_keywords:
        if kw in title_lower:
            return 0.7
            
    general_keywords = ["software engineer", "backend", "fullstack", "developer", "tech lead", "engineering manager"]
    for kw in general_keywords:
        if kw in title_lower:
            return 0.4
            
    return 0.0

def compute_experience_fit(years: float) -> float:
    if years is None:
        return 0.0
    if 5 <= years <= 9:
        return 1.0
    elif years == 4 or years == 10:
        return 0.8
    elif years == 3 or years == 11:
        return 0.6
    else:
        return 0.3

def compute_skill_fit(skills: list) -> float:
    if not skills:
        return 0.0
        
    categories = {
        "vector_search": ["pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "opensearch", "vector search", "vector db", "hybrid search"],
        "dense_embeddings": ["sentence-transformers", "embeddings", "bert", "sbert", "e5", "bge", "dense retrieval", "retrieval"],
        "evaluation": ["ndcg", "mrr", "map", "evaluation", "recall", "precision"],
        "programming": ["python"]
    }
    
    bonus_category = ["lora", "qlora", "peft", "fine-tuning", "llama", "llm"]
    
    matched_categories = set()
    has_bonus = False
    
    for skill in skills:
        skill_name = skill.get("name", "").lower()
        if not skill_name:
            continue
            
        for cat_name, keywords in categories.items():
            for kw in keywords:
                if kw in skill_name:
                    matched_categories.add(cat_name)
                    break
                    
        for kw in bonus_category:
            if kw in skill_name:
                has_bonus = True
                break
                
    score = len(matched_categories) * 0.25
    if has_bonus:
        score += 0.1
        
    return min(1.0, score)

def compute_behavioral_modifier(candidate: dict) -> float:
    signals = candidate.get("redrob_signals", {})
    if not signals:
        return 1.0
        
    modifier = 1.0
    
    response_rate = signals.get("recruiter_response_rate")
    if response_rate is not None:
        if response_rate >= 0.8:
            modifier += 0.05
        elif response_rate < 0.2:
            modifier -= 0.15
            
    notice_period = signals.get("notice_period_days")
    if notice_period is not None:
        if notice_period <= 30:
            modifier += 0.05
        elif notice_period >= 90:
            modifier -= 0.10
            
    open_to_work = signals.get("open_to_work_flag")
    if open_to_work is True:
        modifier += 0.05
        
    completion_rate = signals.get("interview_completion_rate")
    if completion_rate is not None:
        if completion_rate >= 0.8:
            modifier += 0.05
        elif completion_rate < 0.4:
            modifier -= 0.10
            
    return max(0.5, min(1.2, modifier))

def score_candidate(candidate: dict, semantic_similarity: float) -> float:
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "")
    years = profile.get("years_of_experience", 0)
    skills = candidate.get("skills", [])
    
    t_fit = compute_title_fit(title)
    e_fit = compute_experience_fit(years)
    s_fit = compute_skill_fit(skills)
    
    core_score = 0.30 * t_fit + 0.15 * e_fit + 0.25 * s_fit + 0.30 * semantic_similarity
    modifier = compute_behavioral_modifier(candidate)
    final_score = core_score * modifier
    
    return round(max(0.0, min(1.0, final_score)), 4)
