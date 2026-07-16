"""
Loads and caches SKILL.md content for each agent's system prompt.

Usage:
    from skills_loader import load_skill
    prompt = load_skill("candidate_search")   # reads skills/candidate_search/SKILL.md
"""

_SKILL_CACHE: dict[str, str] = {}


def load_skill(skill_name: str) -> str:
    if skill_name not in _SKILL_CACHE:
        with open(f"skills/{skill_name}/SKILL.md", encoding="utf-8") as f:
            _SKILL_CACHE[skill_name] = f.read()
    return _SKILL_CACHE[skill_name]
