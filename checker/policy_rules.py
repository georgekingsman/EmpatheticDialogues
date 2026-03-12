"""
Policy layer — maps Checker verdicts to system actions.

Actions:
    approve   → pass Maker response through unchanged
    revise    → return revision guidance; caller re-prompts Maker
    abstain   → system declines to answer
    escalate  → switch to crisis-oriented response with resources
"""

from __future__ import annotations

from pathlib import Path

from checker.checker_schema import CHECKER_DIMENSIONS, validate_checker_output

# ---------------------------------------------------------------------------
# Default crisis resources
# ---------------------------------------------------------------------------

DEFAULT_CRISIS_RESOURCES = (
    "If you are in crisis, please reach out:\n"
    "• National Suicide Prevention Lifeline: 988 (US)\n"
    "• Crisis Text Line: Text HOME to 741741 (US)\n"
    "• Samaritans: 116 123 (UK)\n"
    "• IASP Crisis Centres: https://www.iasp.info/resources/Crisis_Centres/"
)


# ---------------------------------------------------------------------------
# Policy decision logic
# ---------------------------------------------------------------------------

def apply_policy(checker_result: dict) -> dict:
    """Determine final action from validated checker output.

    Parameters
    ----------
    checker_result : dict
        Output of ``validate_checker_output()``.

    Returns
    -------
    dict with keys:
        action          — approve | revise | abstain | escalate
        maker_response  — whether to pass through the original response
        revision_note   — guidance for Maker if action is revise
        user_message    — message to show user (for abstain / escalate)
        resources       — crisis resources string (for escalate)
    """
    decision = checker_result["decision"]
    scores = checker_result["scores"]

    # --- hard override: any escalation signal → escalate ---
    if scores.get("escalation_need", 5) <= 2:
        decision = "escalate"

    # --- hard override: dangerous safety score → abstain ---
    if scores.get("safety_boundaries", 5) <= 1:
        decision = "abstain"

    # --- build result ---
    if decision == "approve":
        return {
            "action": "approve",
            "pass_through": True,
            "revision_note": "",
            "user_message": "",
            "resources": "",
        }

    if decision == "revise":
        return {
            "action": "revise",
            "pass_through": False,
            "revision_note": checker_result.get("revision_guidance", ""),
            "user_message": "",
            "resources": "",
        }

    if decision == "abstain":
        return {
            "action": "abstain",
            "pass_through": False,
            "revision_note": "",
            "user_message": (
                "I want to be thoughtful here. This topic is beyond what I "
                "can helpfully address. I'd encourage you to speak with a "
                "qualified professional who can give you the support you "
                "deserve."
            ),
            "resources": "",
        }

    # escalate
    resources = checker_result.get("escalation_resources", "")
    if not resources.strip():
        resources = DEFAULT_CRISIS_RESOURCES

    return {
        "action": "escalate",
        "pass_through": False,
        "revision_note": "",
        "user_message": (
            "It sounds like you may be going through something really "
            "serious, and I want to make sure you get the right support. "
            "Please consider reaching out to one of these resources — "
            "you don't have to go through this alone."
        ),
        "resources": resources,
    }


# ---------------------------------------------------------------------------
# Visible-checker explanation builder
# ---------------------------------------------------------------------------

def build_visible_explanation(checker_result: dict, policy_result: dict) -> dict:
    """Build user-facing explanation for Condition C (Visible Checker).

    Returns
    -------
    dict with:
        indicator   — compact one-line string always shown
        detail      — expandable explanation (shown on click)
    """
    action = policy_result["action"]

    indicator_map = {
        "approve": "✓ Safety reviewed — This response has been reviewed for safety and appropriateness.",
        "revise": "✓ Safety reviewed — This response was adjusted after a safety review.",
        "abstain": "✓ Safety reviewed — We want to be careful with this topic.",
        "escalate": "✓ Safety reviewed — This response includes professional resources based on a safety review.",
    }

    indicator = indicator_map.get(action, indicator_map["approve"])

    detail_lines = [
        "🔍 Safety Review Details",
        "",
        "Our system uses a two-step process to support you:",
        "  1. A support agent drafts a caring response.",
        "  2. An independent safety reviewer checks it before you see it.",
        "",
    ]

    if action == "approve":
        detail_lines.append("Review result: Approved — no concerns found.")
    elif action == "revise":
        detail_lines.append("Review result: Adjusted — minor improvements were made.")
        guidance = checker_result.get("revision_guidance", "")
        if guidance:
            detail_lines.append(f"The original response was refined: {guidance}")
    elif action == "abstain":
        detail_lines.append("Review result: Caution — we want to be careful here.")
        detail_lines.append(
            "We weren't confident we could respond helpfully, so we suggest "
            "speaking with someone who can help more directly."
        )
    elif action == "escalate":
        detail_lines.append("Review result: We care about your safety.")
        detail_lines.append("")
        detail_lines.append(policy_result.get("resources", DEFAULT_CRISIS_RESOURCES))
        detail_lines.append("")
        detail_lines.append("You don't have to go through this alone.")

    return {
        "indicator": indicator,
        "detail": "\n".join(detail_lines),
    }
