"""
Checker output schema and validation.

Defines the structured JSON contract between the Checker agent and the
policy layer.
"""

from __future__ import annotations

CHECKER_DIMENSIONS = [
    "emotional_appropriateness",
    "safety_boundaries",
    "uncertainty_grounding",
    "escalation_need",
]

VALID_DECISIONS = {"approve", "revise", "abstain", "escalate"}
VALID_SCORES = list(range(1, 6))  # 1–5


def validate_checker_output(parsed: dict) -> dict | None:
    """Validate a parsed checker JSON response.

    Returns a cleaned dict or None if structurally invalid.
    """
    if not isinstance(parsed, dict):
        return None

    # --- scores ---
    scores = parsed.get("scores", {})
    if not isinstance(scores, dict):
        return None
    for dim in CHECKER_DIMENSIONS:
        val = scores.get(dim)
        if not isinstance(val, (int, float)):
            return None
        if int(val) not in VALID_SCORES:
            return None

    # --- decision ---
    decision = parsed.get("decision", "")
    if decision not in VALID_DECISIONS:
        return None

    # --- optional fields ---
    flags = parsed.get("flags", [])
    if not isinstance(flags, list):
        flags = []
    flags = [str(f)[:200] for f in flags]

    revision_guidance = str(parsed.get("revision_guidance", ""))[:500]
    escalation_resources = str(parsed.get("escalation_resources", ""))[:500]

    return {
        "scores": {dim: int(scores[dim]) for dim in CHECKER_DIMENSIONS},
        "decision": decision,
        "flags": flags,
        "revision_guidance": revision_guidance,
        "escalation_resources": escalation_resources,
    }
