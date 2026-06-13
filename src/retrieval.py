import re

TITLE_KEYWORDS = [
    'ai', 'ml', 'nlp', 'cv', 'llm', 'rag', 'peft', 'lora', 'qlora',
    'deep learning', 'machine learning', 'data science', 'data scientist',
    'vector search', 'vector db', 'vector database', 'hybrid search',
    'neural network', 'neural networks', 'sentence-transformers', 'embeddings',
    'pytorch', 'tensorflow', 'ndcg', 'mrr', 'map', 'transformers',
    'information retrieval', 'recommendation', 'recommender', 'fine-tuning'
]

TITLE_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in TITLE_KEYWORDS) + r')\b',
    re.IGNORECASE
)

SERVICE_COMPANIES = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture", "cognizant", 
    "capgemini", "hcl", "tech mahindra", "l&t infotech", "lnt infotech", "mindtree", 
    "deloitte", "ey", "pwc", "kpmg", "capgemini india", "tata consultancy"
}

def is_service_company(company_name: str) -> bool:
    if not company_name:
        return False
    name_lower = company_name.lower()
    for svc in SERVICE_COMPANIES:
        if svc in name_lower:
            return True
    return False

def matches_high_recall_keywords(candidate: dict) -> bool:
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    
    title_texts = []
    if profile.get("headline"):
        title_texts.append(profile.get("headline"))
    if profile.get("current_title"):
        title_texts.append(profile.get("current_title"))
        
    for job in career_history:
        if job.get("title"):
            title_texts.append(job.get("title"))
            
    combined_titles = " ".join(title_texts)
    return bool(TITLE_PATTERN.search(combined_titles))
