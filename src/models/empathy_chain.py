"""
Chain-of-Empathy enhanced GPT-2 model.

This module is retained as an **ablation / weak-empathy baseline** within the
evaluation framework.  The primary research contribution is the evaluation &
calibration pipeline, not this model itself.

Architecture
------------
Input → GPT-2 (output_hidden_states=True)
         ↓ hidden_states[-1]
         ↓ mean pooling → (batch, hidden_dim)
         ↓ ChainOfEmpathy (5-stage reasoning)
         ↓ broadcast-add back to hidden_states
         ↓ gpt2.lm_head → logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Chain-of-Empathy sub-module
# ---------------------------------------------------------------------------

class ChainOfEmpathy(nn.Module):
    """Five-stage empathy reasoning chain:
    scenario → emotion → cause → goal → response, with emotion–scenario fusion.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scenario_layer = nn.Linear(hidden_dim, hidden_dim)
        self.emotion_layer = nn.Linear(hidden_dim, hidden_dim)
        self.cause_layer = nn.Linear(hidden_dim, hidden_dim)
        self.goal_layer = nn.Linear(hidden_dim, hidden_dim)
        self.response_layer = nn.Linear(hidden_dim, hidden_dim)
        self.emotion_fusion_layer = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scenario_rep = F.relu(self.scenario_layer(x))
        emotion_rep = F.relu(self.emotion_layer(scenario_rep))
        cause_rep = F.relu(self.cause_layer(emotion_rep))
        goal_rep = F.relu(self.goal_layer(cause_rep))
        response_rep = self.response_layer(goal_rep)

        fused_rep = torch.cat((emotion_rep, scenario_rep), dim=-1)
        fused_rep = F.relu(self.emotion_fusion_layer(fused_rep))

        return response_rep + fused_rep


def _init_chain_weights(chain: ChainOfEmpathy) -> None:
    """Xavier-uniform initialization for chain layers."""
    for name, param in chain.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CBTEmpatheticModel(nn.Module):
    """GPT-2 + Chain-of-Empathy for training and generation."""

    def __init__(self, model_name: str = "gpt2", hidden_dim: int = 768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.gpt2 = AutoModelForCausalLM.from_pretrained(model_name)
        self.chain = ChainOfEmpathy(hidden_dim)
        _init_chain_weights(self.chain)

        self.chain_weight = 0.5  # fusion strength at inference

    # ------------------------------------------------------------------
    # Training forward (labels-aware)
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]               # (B, T, H)

        pooled = hidden_states.mean(dim=1)                      # (B, H)
        chain_out = self.chain(pooled)                          # (B, H)
        chain_expanded = chain_out.unsqueeze(1).expand_as(hidden_states)

        combined = hidden_states + chain_expanded               # additive fusion
        logits = self.gpt2.lm_head(combined)                   # (B, T, V)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, **kwargs) -> torch.Tensor:
        return self.gpt2.generate(**kwargs)

    @torch.no_grad()
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """Generate a response using chain-conditioned input embeddings."""
        device = next(self.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Get chain-conditioned embeddings
        input_embeds = self.gpt2.get_input_embeddings()(input_ids)
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        pooled = hidden_states.mean(dim=1)
        chain_out = self.chain(pooled)
        chain_expanded = chain_out.unsqueeze(1).expand_as(input_embeds)
        conditioned_embeds = input_embeds + self.chain_weight * chain_expanded

        generated = self.gpt2.generate(
            inputs_embeds=conditioned_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # When inputs_embeds is used, newer transformers may only return new
        # tokens (no echo of the prompt ids).  Handle both cases:
        prompt_len = input_ids.shape[1]
        gen_len = generated.shape[1]
        if gen_len > prompt_len:
            new_tokens = generated[0][prompt_len:]
        else:
            new_tokens = generated[0]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
