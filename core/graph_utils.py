import re

def normalize_entity_name(name: str) -> str:
    """Normalize entity names: lowercase, strip, & -> and, strip punctuation, collapse whitespace."""
    name = name.lower()
    name = name.replace("&", "and")
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name
