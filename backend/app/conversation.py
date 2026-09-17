"""Conversation layer.

Holds per-session state (in-memory dict keyed by session_id) so multi-turn
conversations accumulate known environmental fields across messages instead
of re-asking. Also does light-weight extraction of structured fields out of
free text, so a user typing "soil organic carbon is 0.3%, rainfall is low,
monoculture wheat in a semi-arid region" doesn't have to use the JSON
endpoint to be understood.

This is deliberately simple pattern matching (not an LLM call) so the whole
system stays free / self-hostable with no external API dependency. Swap-in
note: `extract_fields_from_text` is the single seam where a local LLM call
could later be substituted for extraction without touching session storage
or the reasoning layer.
"""
import re
from typing import Dict, Any

# session_id -> known fields dict
SESSIONS: Dict[str, Dict[str, Any]] = {}

_NUMERIC_FIELD_PATTERNS = {
    "soil_organic_carbon": r"(?:soil organic carbon|soc|organic carbon)[^\d]{0,10}(\d+(?:\.\d+)?)\s*%?",
    "soil_ph": r"(?:soil )?ph[^\d]{0,10}(\d+(?:\.\d+)?)",
}

_CATEGORICAL_FIELD_PATTERNS = {
    "rainfall": r"rainfall\D{0,10}(low|erratic|medium|moderate|high)",
    "temperature": r"temperature\D{0,10}(stable|rising|extreme|hot|cold)",
    "land_use": r"(monoculture|agroforestry|intercrop\w*|pasture|cleared|deforested|forest(?:ed)?)",
    "species_richness": r"species richness\D{0,10}(low|medium|high)",
    "habitat_diversity": r"habitat diversity\D{0,10}(low|medium|high)",
    "habitat_fragmentation": r"(?:habitat )?fragmentation\D{0,10}(low|medium|high)",
    "pollution_level": r"pollution\D{0,10}(low|medium|high)",
    "deforestation_rate": r"deforestation\D{0,10}(low|medium|high)",
    "region": r"(semi-arid|arid|tropical|temperate|subtropical|tundra|mediterranean)",
}

_LAND_USE_NORMALIZE = {
    "intercropping": "agroforestry",
    "intercrop": "agroforestry",
    "deforested": "cleared",
    "forested": "forest",
}


def extract_fields_from_text(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    found: Dict[str, Any] = {}

    for field, pattern in _NUMERIC_FIELD_PATTERNS.items():
        m = re.search(pattern, text_lower)
        if m:
            try:
                found[field] = float(m.group(1))
            except ValueError:
                pass

    for field, pattern in _CATEGORICAL_FIELD_PATTERNS.items():
        m = re.search(pattern, text_lower)
        if m:
            val = m.group(1)
            found[field] = _LAND_USE_NORMALIZE.get(val, val)

    return found


def get_session(session_id: str) -> Dict[str, Any]:
    return SESSIONS.setdefault(session_id, {})


def update_session(session_id: str, new_fields: Dict[str, Any]) -> Dict[str, Any]:
    session = get_session(session_id)
    for k, v in new_fields.items():
        if v is not None:
            session[k] = v
    return session


def reset_session(session_id: str) -> None:
    SESSIONS.pop(session_id, None)
