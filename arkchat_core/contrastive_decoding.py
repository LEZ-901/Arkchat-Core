#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evidence-aware contrastive decoding for text-only causal language models.

This is inspired by Attention Contrastive Decoding (ACD), but operates on two
text branches because this package has no visual tokens:

  grounded branch   = persona + retrieved evidence + conversation
  ungrounded branch = persona + conversation

Both branches decode the same generated tokens. Their KV caches remain
independent, while the next-token distribution is combined at logit level.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

try:  # Keep CPU-only routing/prompt tests importable without CUDA runtimes.
    import torch
    _TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on the host install.
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc


def _require_torch() -> None:
    if torch is None:
        detail = f": {_TORCH_IMPORT_ERROR}" if _TORCH_IMPORT_ERROR else ""
        raise RuntimeError(
            "contrastive decoding requires a working PyTorch installation"
            + detail
        )


@dataclass
class ContrastiveStats:
    strategy: str = "cad"
    steps: int = 0
    applied_steps: int = 0
    mean_alpha: float = 0.0
    max_evidence_margin: float = 0.0
    grounded_prompt_tokens: int = 0
    ungrounded_prompt_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def adaptive_contrast_logits(
    grounded_logits: torch.Tensor,
    ungrounded_logits: torch.Tensor,
    alpha: float = 0.8,
    gate: float = 0.08,
    plausibility_beta: float = 0.05,
) -> tuple[torch.Tensor, float, float]:
    """Combine two next-token distributions without promoting implausible tokens.

    The classic context-aware rule is
    ``grounded + alpha * (grounded - ungrounded)``. We apply it in log-probability
    space, restrict candidates to tokens that remain plausible in the grounded
    branch, and ramp alpha up only when the evidence creates a measurable margin.
    """
    _require_torch()
    if grounded_logits.shape != ungrounded_logits.shape:
        raise ValueError("grounded and ungrounded logits must have the same shape")
    if not 0 < plausibility_beta <= 1:
        raise ValueError("plausibility_beta must be in (0, 1]")
    if alpha < 0 or gate < 0:
        raise ValueError("alpha and gate must be non-negative")

    grounded = torch.log_softmax(grounded_logits.float(), dim=-1)
    ungrounded = torch.log_softmax(ungrounded_logits.float(), dim=-1)
    floor = grounded.amax(dim=-1, keepdim=True) + math.log(plausibility_beta)
    plausible = grounded >= floor
    delta = grounded - ungrounded

    evidence_margin = delta.masked_fill(~plausible, -torch.inf).amax(dim=-1)
    # A short linear ramp avoids switching from standard decoding to full
    # contrast on a tiny numerical difference between the two branches.
    ramp = ((evidence_margin - gate) / 0.5).clamp(0.0, 1.0)
    effective_alpha = alpha * ramp
    combined = grounded + effective_alpha.unsqueeze(-1) * delta
    combined = combined.masked_fill(~plausible, -torch.inf)
    return combined, float(effective_alpha.mean().item()), float(
        evidence_margin.max().item()
    )


def apply_repetition_penalty(
    logits: torch.Tensor, token_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Apply the same sign-aware repetition penalty used by Transformers."""
    _require_torch()
    if penalty <= 0:
        raise ValueError("repetition penalty must be positive")
    if penalty == 1.0 or token_ids.numel() == 0:
        return logits
    out = logits.clone()
    for batch in range(out.shape[0]):
        used = torch.unique(token_ids[batch])
        scores = out[batch, used]
        scores = torch.where(scores < 0, scores * penalty, scores / penalty)
        out[batch, used] = scores
    return out


def _eos_ids(eos_token_id) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return {int(token_id) for token_id in eos_token_id}


def contrastive_generate(
    model,
    grounded_ids: torch.Tensor,
    ungrounded_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id=None,
    alpha: float = 0.8,
    gate: float = 0.08,
    plausibility_beta: float = 0.05,
    repetition_penalty: float = 1.08,
) -> tuple[torch.Tensor, dict]:
    """Greedily decode a shared continuation from grounded/ungrounded branches."""
    _require_torch()
    if grounded_ids.shape[0] != 1 or ungrounded_ids.shape[0] != 1:
        raise ValueError("contrastive_generate currently supports batch size 1")
    if max_new_tokens <= 0:
        return grounded_ids, ContrastiveStats(
            grounded_prompt_tokens=grounded_ids.shape[1],
            ungrounded_prompt_tokens=ungrounded_ids.shape[1],
        ).to_dict()

    grounded_mask = torch.ones_like(grounded_ids)
    ungrounded_mask = torch.ones_like(ungrounded_ids)
    grounded_out = model(
        input_ids=grounded_ids,
        attention_mask=grounded_mask,
        use_cache=True,
        return_dict=True,
        logits_to_keep=1,
    )
    ungrounded_out = model(
        input_ids=ungrounded_ids,
        attention_mask=ungrounded_mask,
        use_cache=True,
        return_dict=True,
        logits_to_keep=1,
    )
    grounded_cache = grounded_out.past_key_values
    ungrounded_cache = ungrounded_out.past_key_values
    generated = grounded_ids
    eos_ids = _eos_ids(eos_token_id)
    alphas = []
    margins = []

    for _ in range(max_new_tokens):
        combined, effective_alpha, evidence_margin = adaptive_contrast_logits(
            grounded_out.logits[:, -1, :],
            ungrounded_out.logits[:, -1, :],
            alpha=alpha,
            gate=gate,
            plausibility_beta=plausibility_beta,
        )
        combined = apply_repetition_penalty(
            combined, generated, repetition_penalty
        )
        next_token = combined.argmax(dim=-1, keepdim=True).to(generated.device)
        generated = torch.cat((generated, next_token), dim=-1)
        alphas.append(effective_alpha)
        margins.append(evidence_margin)
        if int(next_token.item()) in eos_ids:
            break

        grounded_mask = torch.cat(
            (grounded_mask, torch.ones_like(next_token)), dim=-1
        )
        ungrounded_mask = torch.cat(
            (ungrounded_mask, torch.ones_like(next_token)), dim=-1
        )
        grounded_out = model(
            input_ids=next_token,
            attention_mask=grounded_mask,
            past_key_values=grounded_cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        ungrounded_out = model(
            input_ids=next_token,
            attention_mask=ungrounded_mask,
            past_key_values=ungrounded_cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        grounded_cache = grounded_out.past_key_values
        ungrounded_cache = ungrounded_out.past_key_values

    stats = ContrastiveStats(
        steps=len(alphas),
        applied_steps=sum(value > 0 for value in alphas),
        mean_alpha=round(sum(alphas) / max(1, len(alphas)), 4),
        max_evidence_margin=round(max(margins, default=0.0), 4),
        grounded_prompt_tokens=grounded_ids.shape[1],
        ungrounded_prompt_tokens=ungrounded_ids.shape[1],
    )
    return generated, stats.to_dict()
