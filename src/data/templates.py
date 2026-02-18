"""
Prompt / response templates for empathetic dialogue training and inference.

All templates follow the Causal LM convention:
    full_text = PROMPT_TEMPLATE.format(...) + response + EOS
    labels mask the prompt portion with -100.
"""

# ---------------------------------------------------------------------------
# Training / inference prompt template
# ---------------------------------------------------------------------------
# {user_statement} will be filled with the patient's current statement.
# The model is expected to generate the therapist's response after the marker.
PROMPT_TEMPLATE = "<user>: {user_statement}\n<assistant>:"

# End-of-sequence marker appended after the target response during training.
EOS_MARKER = "<|endoftext|>"


def build_training_text(user_statement: str, therapist_response: str) -> dict:
    """Return prompt / target / full_text for a single training sample.

    Returns
    -------
    dict with keys:
        prompt        – the model *input* (labels = -100 for these tokens)
        target        – the expected completion (loss computed here)
        full_text     – prompt + " " + target + EOS_MARKER
    """
    prompt = PROMPT_TEMPLATE.format(user_statement=user_statement)
    target = f" {therapist_response} {EOS_MARKER}"
    full_text = prompt + target
    return {"prompt": prompt, "target": target, "full_text": full_text}


def build_inference_prompt(user_statement: str) -> str:
    """Return the prompt string used at inference time (no target)."""
    return PROMPT_TEMPLATE.format(user_statement=user_statement)
