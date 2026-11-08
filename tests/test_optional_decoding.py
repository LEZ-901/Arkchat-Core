import pytest

torch = pytest.importorskip("torch")

from arkchat_core.contrastive_decoding import adaptive_contrast_logits


def test_evidence_can_change_next_token_choice():
    grounded = torch.tensor([[4.0, 3.9, 0.0]])
    ungrounded = torch.tensor([[4.2, 1.0, 0.0]])
    combined, effective_alpha, margin = adaptive_contrast_logits(
        grounded, ungrounded, alpha=1.0, gate=0.0
    )
    assert grounded.argmax().item() == 0
    assert combined.argmax().item() == 1
    assert effective_alpha > 0
    assert margin > 0
