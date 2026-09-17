"""Reasoning layer.

Turns (a) retrieved knowledge entries and (b) the user's known structured
variables into ranked, evidence-backed recommendations. This is where the
system explicitly connects multiple environmental variables (soil <-> water
<-> biodiversity <-> land use), rather than answering off a single field.

Design:
1. Build a natural-language query from whatever fields are known, and use
   the retriever to shortlist candidate knowledge entries.
2. For every candidate, evaluate its structured `trigger_condition` against
   the known fields (a tiny sandboxed expression evaluator - no `eval` of
   arbitrary code, only comparisons against a fixed variable namespace).
3. Score = retrieval similarity + a bonus if the structured trigger
   condition is actually satisfied by the user's numbers/categories, plus a
   bonus for each additional distinct environmental *category* the
   recommendation touches (its own category + trigger_metrics span +
   linked_variables) - this is what rewards multi-variable, non-obvious
   answers over single-variable ones.
4. Return the top N, deduplicated by knowledge-base id.
"""
from typing import Dict, Any, List
import re

from .retrieval import retriever
from .knowledge_base import knowledge_base

# Fields that describe each environmental "category" for multi-metric scoring
CATEGORY_FIELDS = {
    "soil_health": {"soil_organic_carbon", "soil_ph", "soil_moisture"},
    "climate_factors": {"rainfall", "temperature"},
    "land_use": {"land_use", "habitat_fragmentation"},
    "biodiversity_indicators": {"species_richness", "habitat_diversity"},
    "human_impact": {"pollution_level", "deforestation_rate"},
}

REQUIRED_MINIMUM_FIELDS = ["soil_organic_carbon_or_ph", "rainfall_or_temperature", "land_use"]

# Every field the conversation layer knows how to elicit, in priority order.
ELICITABLE_FIELDS = [
    ("soil_organic_carbon", "What is the approximate soil organic carbon (%)? A lab test or rough estimate both work."),
    ("rainfall", "How would you describe rainfall in the area — low, erratic, medium, or high?"),
    ("land_use", "What is the current land use / land cover — e.g. monoculture, agroforestry, pasture, cleared, forest?"),
    ("species_richness", "How would you rate species richness on the land — low, medium, or high?"),
    ("region", "Which region or climate zone is this (e.g. semi-arid, tropical, temperate)?"),
]

MINIMUM_FIELDS_FOR_REASONING = ["soil_organic_carbon", "rainfall", "land_use"]


def _safe_eval_condition(condition: str, known: Dict[str, Any]) -> bool:
    """Evaluate a simple boolean trigger_condition string against known
    fields. Only allows the known variable names plus comparison/boolean
    operators - no builtins, no attribute access, no calls.
    """
    # Only allow tokens that are: known field names, python literals/operators, whitespace
    allowed_names = set(known.keys())
    tokens = re.findall(r"[A-Za-z_]+", condition)
    for tok in tokens:
        if tok in ("and", "or", "not", "in", "None", "True", "False"):
            continue
        if tok not in allowed_names:
            return False  # referenced field is unknown -> condition can't be confirmed
    try:
        return bool(eval(condition, {"__builtins__": {}}, dict(known)))
    except Exception:
        return False


def missing_required_fields(known: Dict[str, Any]) -> List[str]:
    missing = []
    for field, question in ELICITABLE_FIELDS[:3]:  # the mandatory minimum 3 variables
        if field not in known or known[field] is None:
            missing.append(field)
    return missing


def _category_span(entry: Dict[str, Any]) -> set:
    """Which of the 5 environmental categories this entry's reasoning touches."""
    span = {entry.get("category")}
    for var in entry.get("linked_variables", []):
        span.add(var)
    return span


def build_query_text(known: Dict[str, Any], free_text: str) -> str:
    parts = [free_text]
    for k, v in known.items():
        parts.append(f"{k} {v}")
    return " ".join(str(p) for p in parts)


def generate_recommendations(known: Dict[str, Any], free_text: str = "", top_k: int = 4) -> List[Dict[str, Any]]:
    query_text = build_query_text(known, free_text)
    candidates = retriever.query(query_text, top_k=10)

    scored = []
    seen_ids = set()
    for entry, sim_score in candidates:
        if entry["id"] in seen_ids:
            continue
        seen_ids.add(entry["id"])

        condition_met = _safe_eval_condition(entry.get("trigger_condition", ""), known)
        multi_metric_bonus = 0.15 * len(_category_span(entry))
        trigger_bonus = 0.5 if condition_met else 0.0
        final_score = round(sim_score + trigger_bonus + multi_metric_bonus, 4)

        scored.append({
            "entry": entry,
            "score": final_score,
            "condition_met": condition_met,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]

    results = []
    for item in top:
        e = item["entry"]
        results.append({
            "title": e["title"],
            "what_to_do": e["intervention"],
            "why_it_works": e["mechanism"],
            "impacted_metrics": e["impacted_metrics"],
            "expected_impact": e["expected_impact"],
            "time_horizon": e["time_horizon"],
            "confidence": e["confidence"] if item["condition_met"] else _downgrade_confidence(e["confidence"]),
            "source": e["source"],
            "linked_variables": sorted(_category_span(e)),
            "retrieval_score": item["score"],
        })
    return results


def _downgrade_confidence(confidence: str) -> str:
    """If the structured trigger condition couldn't be confirmed against the
    user's known fields (e.g. they only gave free text), we still surface
    the recommendation via semantic retrieval but flag lower confidence
    rather than silently asserting the same certainty.
    """
    order = ["high", "medium", "low"]
    idx = order.index(confidence) if confidence in order else 1
    return order[min(idx + 1, len(order) - 1)]
