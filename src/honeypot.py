def is_honeypot(candidate: dict) -> bool:
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    
    for job in career_history:
        duration = job.get("duration_months")
        if duration is not None and duration < 0:
            return True
            
    expert_zero_duration_count = 0
    for skill in skills:
        proficiency = skill.get("proficiency", "").lower()
        duration = skill.get("duration_months")
        if proficiency == "expert" and (duration is None or duration == 0):
            expert_zero_duration_count += 1
            
    if expert_zero_duration_count >= 8:
        return True
        
    years_of_exp = profile.get("years_of_experience")
    if years_of_exp is not None and years_of_exp >= 2:
        total_duration_months = sum(job.get("duration_months", 0) for job in career_history if job.get("duration_months", 0) > 0)
        expected_months = years_of_exp * 12
        if total_duration_months < (expected_months * 0.15):
            return True
            
    return False
