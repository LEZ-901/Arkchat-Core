# Release boundary and audit

This directory is a standalone code candidate extracted from the ArkChat workspace. It includes only four reusable Python modules and small synthetic tests. It deliberately excludes `penance.py`, `prompt_spec.py`, role personas, training corpora, RAG indexes, model weights, adapters, generated reports, cluster account details, and live conversation data. It is **not** a drop-in replacement for the complete v9 application.

Before public publication:

1. The project owner states that they are the sole maintainer and copyright holder of the included code and authorizes preparing it as an open-source project. The chosen public credit is `ArkChat`; the MIT notice and package metadata use that name.
2. Confirm the package and repository names do not conflict with an existing published package.
3. Review the exact Git staging area for secrets, user data, model weights, and character-specific text. `.gitignore` is a guardrail, not a substitute for reviewing tracked files or commit history.
4. Run tests in a clean Python environment. Test optional PyTorch behavior separately on a compatible machine.
5. Enable GitHub private vulnerability reporting or add a maintainer-approved contact for private security reports, and choose a release/version policy.

Do not publish the parent workspace as-is. Its `data/`, `models/`, `output/`, `persona/`, `dist/`, `logs/`, and reports contain materials with different provenance or publication requirements. This repository targets `https://github.com/LEZ-901/Arkchat-Core`; it is not uploaded to a package index.
