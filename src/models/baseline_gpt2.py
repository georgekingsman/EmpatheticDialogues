"""
GPT-2 Baseline model — pure causal LM without the Chain-of-Empathy module.

Used as the primary control / ablation in the evaluation framework.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPT2BaselineModel(nn.Module):
    """Thin wrapper around HuggingFace GPT-2 for training and generation."""

    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Standard Causal LM forward.

        If ``labels`` is provided (with -100 masking on prompt tokens),
        the model computes the cross-entropy loss internally.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, **kwargs) -> torch.Tensor:
        return self.model.generate(**kwargs)

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
        """Generate a single response string from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(self.parameters()).device)
        attention_mask = inputs["attention_mask"].to(next(self.parameters()).device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Only decode the *new* tokens (strip prompt)
        new_tokens = outputs[0][input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
