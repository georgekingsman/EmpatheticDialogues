## Visible Checker — User-Facing Explanation Templates

These templates are used in **Condition C** (Double AI, Visible Checker) to
generate the safety-review indicator and optional expanded explanation shown to
the user.

---

### Compact indicator (always shown)

```
✓ Safety reviewed — {summary}
```

Where `{summary}` is one of:
- "This response has been reviewed for safety and appropriateness."
- "This response was adjusted after a safety review."
- "This response includes professional resources based on a safety review."

---

### Expansion template (shown on click / tap)

```
🔍 Safety Review Details

Our system uses a two-step process to support you:
  1. A support agent drafts a caring response.
  2. An independent safety reviewer checks it before you see it.

Review result: {decision_label}

{detail_sections}
```

---

### Decision-specific labels and details

#### approve
```
decision_label: "Approved — no concerns found."
detail_sections: ""
```

#### revise
```
decision_label: "Adjusted — minor improvements were made."
detail_sections:
  "The original response was refined to better match your situation.
   Specifically: {revision_summary}"
```

#### abstain
```
decision_label: "Caution — we want to be careful here."
detail_sections:
  "We weren't confident we could respond helpfully to this topic.
   Rather than risk giving unhelpful advice, we'd like to suggest
   speaking with someone who can help more directly.
   {referral_suggestion}"
```

#### escalate
```
decision_label: "We care about your safety."
detail_sections:
  "It sounds like you may be going through something serious.
   We want to make sure you have the right support:

   {escalation_resources}

   You don't have to go through this alone."
```

---

### Variable definitions

| Variable | Source | Description |
|----------|--------|-------------|
| `{summary}` | Policy layer | One-line indicator text |
| `{decision_label}` | Checker `decision` field | Human-readable label |
| `{revision_summary}` | Checker `revision_guidance` | What was changed |
| `{referral_suggestion}` | Policy layer | Professional resource suggestion |
| `{escalation_resources}` | Checker `escalation_resources` | Crisis hotlines etc. |

---

### Default crisis resources (English)

```
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741 (US)
- Samaritans: 116 123 (UK)
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
```

---

### Design rationale

The visible checker serves two purposes:
1. **Transparency** — users understand that responses are reviewed, which may
   improve trust calibration.
2. **Boundary setting** — the system models appropriate help-seeking by openly
   acknowledging its limitations and suggesting professional resources.

The compact indicator is always visible; the expansion is opt-in to avoid
cognitive overload.
