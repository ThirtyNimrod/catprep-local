import re
from core.utils import contains_phrase

def extract_timeframe(text: str) -> str | None:
    match = re.search(r'\b(\d+)\s*(day|days|week|weeks|month|months)\b', text.lower())
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None

def extract_focus_area(text: str) -> str | None:
    categories = {
        "QA": ["qa", "quantitative ability", "quant", "math"],
        "VA-RC": ["varc", "va rc", "verbal", "reading comprehension"],
        "LR-DI": ["lrdi", "lr di", "logical reasoning", "data interpretation", "lr", "di"]
    }
    
    found = []
    for category, keywords in categories.items():
        if any(contains_phrase(text, kw) for kw in keywords):
            found.append(category)
            
    if found:
        return "/".join(found)
    return None
