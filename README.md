# ArkChat Core

ArkChat Core is a small, reusable extraction of the ArkChat v9 control plane. It provides reviewable long-term memory, metadata-only tool-routing experience, model artifact/version utilities, and optional evidence-aware text contrastive decoding. It does **not** include the original character application, copyrighted persona/corpus, pretrained weights, adapters, or user conversations.

This repository is an **initial source release candidate**. It is not the full ArkChat v9 chat application. The project owner states that they are the sole maintainer and copyright holder of the included code and chooses **ArkChat** as the public credit. The package name should be checked before publishing to a package index or public Git host.

## Requirements and quick start

Python 3.10 or newer. The base package uses the standard library. PyTorch is needed only for contrastive decoding.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
python -m pytest -q
```

For decoding functionality, install `python -m pip install -e '.[decoding,test]'` in a compatible PyTorch environment. The package does not download a model automatically.

```python
from arkchat_core.conversation_memory import propose_user_memories

history = [
    {"role": "user", "content": "我喜欢红茶。"},
    {"role": "assistant", "content": "收到。"},
    {"role": "user", "content": "今天聊聊别的。"},
]
candidates = propose_user_memories(history)
print(candidates)  # review candidates; no automatic persistent write
```

## Components

| Module | Responsibility | Model required? |
| --- | --- | --- |
| `conversation_memory` | Candidate extraction, approval, correction, expiration, recall | No |
| `v9_metaagent` | Bounded tool choice and metadata-only feedback | No |
| `v9_infra` | Generation telemetry and versioned adapter registry | No for metadata operations |
| `contrastive_decoding` | Two-branch evidence-aware token scoring | PyTorch and caller-supplied causal LM |

These modules are extracted from a research prototype. Their APIs may change before 1.0. The memory store may contain user data; applications must provide their own access controls, retention policy, and deletion policy. Retrieval evidence quality and model output quality require separate evaluation.

## Scope and provenance

The source modules were copied from ArkChat v9 and only the role-specific labels in `conversation_memory` and a module comment were generalized. The package has no pretrained model or runnable role-chat CLI. See [RELEASE_AUDIT.md](RELEASE_AUDIT.md) for the source boundary and publication checklist. MIT applies to the code in this directory; it does not relicense models, datasets, character assets, or files in the parent workspace.

## Development

Run `python -m pytest -q`. Please include a focused test with behavior changes. See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues should follow [SECURITY.md](SECURITY.md).
