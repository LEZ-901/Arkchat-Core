from pathlib import Path

from arkchat_core.conversation_memory import propose_user_memories
from arkchat_core.v9_infra import sha256_file
from arkchat_core.v9_metaagent import InMemoryExperienceStore, MetaAgentController


def test_memory_proposal_uses_explicit_user_statement():
    history = [
        {"role": "user", "content": "我喜欢红茶。"},
        {"role": "assistant", "content": "我喜欢咖啡。"},
        {"role": "user", "content": "今天聊聊别的。"},
    ]
    candidates = propose_user_memories(history)
    assert len(candidates) == 1
    assert candidates[0]["content"] == "我喜欢红茶。"
    assert candidates[0]["source_role"] == "user"


def test_required_tool_is_not_suppressed():
    controller = MetaAgentController(InMemoryExperienceStore())
    plan = controller.plan("fact", "source", required=("rag",), optional=("memory",))
    assert "rag" in plan.selected


def test_artifact_hash_changes_with_bytes(tmp_path: Path):
    path = tmp_path / "artifact.txt"
    path.write_text("first", encoding="utf-8")
    first = sha256_file(path)
    path.write_text("second", encoding="utf-8")
    assert sha256_file(path) != first
