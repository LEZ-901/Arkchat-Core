#!/usr/bin/env python3
"""Bounded MetaAgent-style tool learning for a single conversation.

The store contains routing outcomes and explicit feedback, never conversation
text, tool payloads, generated answers, or model-authored factual "lessons".
Tool provenance is checked separately from answer correctness. Only feedback
can teach the controller that a response was good or bad.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence
import uuid


SCHEMA = "arkchat-v9-metaagent-1"
KEY = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
MAX_EVENTS = 256


@dataclass(frozen=True)
class Tool:
    name: str
    run: Callable[[], Any]
    verify_provenance: Callable[[Any], bool]

    def __post_init__(self) -> None:
        if not KEY.fullmatch(self.name):
            raise ValueError("tool name must be a short identifier")


@dataclass(frozen=True)
class HelpPlan:
    task_key: str
    gap: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    selected: tuple[str, ...]
    recalled_lessons: tuple[str, ...]


@dataclass(frozen=True)
class ToolObservation:
    name: str
    result: Any
    evidence_count: int
    provenance_verified: bool


class MetaExperienceStore:
    """Small, conversation-scoped, metadata-only experience log."""

    def __init__(self, root: str | Path, conversation_id: str):
        if not conversation_id:
            raise ValueError("meta experience requires a conversation id")
        digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()
        self.path = Path(root) / f"{digest}.meta-v9.json"

    def read(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        if payload.get("schema") != SCHEMA or not isinstance(payload.get("events"), list):
            raise ValueError("invalid meta experience schema")
        return payload["events"][-MAX_EVENTS:]

    def append(self, event: Mapping[str, Any]) -> None:
        events = self.read()
        events.append(dict(event))
        payload = {"schema": SCHEMA, "events": events[-MAX_EVENTS:]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True),
                                 encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


class InMemoryExperienceStore:
    """Session-only experience when no persistent conversation id is supplied."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def read(self) -> list[dict[str, Any]]:
        return list(self.events)

    def append(self, event: Mapping[str, Any]) -> None:
        self.events = (self.events + [dict(event)])[-MAX_EVENTS:]

    def clear(self) -> None:
        self.events.clear()


class MetaAgentController:
    """Detect a help gap, route bounded tools, verify evidence, reuse lessons.

    The caller supplies task classes and trusted tools. The controller may
    explore an optional tool once per task class, and then learns whether it
    produces verifiable evidence. Required tools are never suppressed. Explicit
    user feedback can strengthen or weaken an optional route on later turns.
    """

    def __init__(self, store: MetaExperienceStore, *, max_tools: int = 2):
        if not 1 <= max_tools <= 4:
            raise ValueError("max_tools must be between 1 and 4")
        self.store = store
        self.max_tools = max_tools
        self.last_episode_id: str | None = next(
            (event["id"] for event in reversed(store.read())
             if event.get("kind") == "episode"), None
        )

    def _scores(self, task_key: str) -> tuple[dict[str, float], set[str]]:
        scores: dict[str, float] = {}
        tried: set[str] = set()
        episodes: dict[str, dict[str, Any]] = {}
        feedback: dict[str, tuple[bool, str | None]] = {}
        for event in self.store.read():
            if event.get("kind") == "episode":
                episodes[event["id"]] = event
            elif event.get("kind") == "feedback":
                feedback[event["episode_id"]] = (
                    event["positive"], event.get("tool"))
        for episode_id, event in episodes.items():
            if event.get("task_key") != task_key:
                continue
            for item in event.get("tools", []):
                name = item["name"]
                tried.add(name)
                if not event.get("response_gate_passed"):
                    continue
                if item["evidence_count"] == 0:
                    scores[name] = scores.get(name, 0.0) - 1.0
                elif item["provenance_verified"]:
                    scores[name] = scores.get(name, 0.0) + 1.0
                user_feedback = feedback.get(episode_id)
                if user_feedback and user_feedback[1] in (None, name):
                    # Generic negative feedback is not enough to blame a tool.
                    if (user_feedback[0] and item["evidence_count"] > 0
                            and item["provenance_verified"]):
                        scores[name] = scores.get(name, 0.0) + 2.0
                    elif user_feedback[1] == name:
                        scores[name] = scores.get(name, 0.0) - 2.0
        return scores, tried

    def plan(self, task_key: str, gap: str, *, required: Sequence[str] = (),
             optional: Sequence[str] = ()) -> HelpPlan:
        if not KEY.fullmatch(task_key) or not KEY.fullmatch(gap):
            raise ValueError("task and gap must be coarse identifiers, not user text")
        required = tuple(dict.fromkeys(required))
        optional = tuple(name for name in dict.fromkeys(optional) if name not in required)
        if any(not KEY.fullmatch(name) for name in required + optional):
            raise ValueError("invalid tool name")
        if len(required) > self.max_tools:
            raise ValueError("required tools exceed budget")
        scores, tried = self._scores(task_key)
        remaining = self.max_tools - len(required)
        preferred = sorted((name for name in optional if scores.get(name, 0) > 0),
                           key=lambda name: (-scores[name], name))
        selected_optional = preferred[:remaining]
        if len(selected_optional) < remaining:
            # Bounded exploration is what lets a new route earn experience.
            fresh = next((name for name in optional
                          if name not in tried and name not in selected_optional), None)
            if fresh:
                selected_optional.append(fresh)
        lessons = tuple(
            f"{name}:source_available" if scores[name] > 0 else f"{name}:no_evidence"
            for name in sorted(scores) if scores[name] != 0
        )
        return HelpPlan(task_key, gap, required, optional,
                        required + tuple(selected_optional), lessons)

    def execute(self, plan: HelpPlan,
                tools: Mapping[str, Tool]) -> dict[str, ToolObservation]:
        observations: dict[str, ToolObservation] = {}
        for name in plan.selected:
            tool = tools.get(name)
            if tool is None or tool.name != name:
                raise ValueError(f"unregistered help tool: {name}")
            try:
                result = tool.run()
                count = len(result) if isinstance(result, (list, tuple)) else int(result is not None)
                verified = bool(count and tool.verify_provenance(result))
            except Exception:
                if name in plan.required:
                    raise
                result = []
                count = 0
                verified = False
            observations[name] = ToolObservation(name, result, count, verified)
        return observations

    def reflect(self, plan: HelpPlan, observations: Mapping[str, ToolObservation],
                *, response_gate_passed: bool) -> str:
        """Persist only bounded strategy metadata after a completed turn."""
        if set(observations) != set(plan.selected):
            raise ValueError("tool observations do not match the help plan")
        episode_id = uuid.uuid4().hex
        self.store.append({
            "kind": "episode", "id": episode_id, "task_key": plan.task_key,
            "gap": plan.gap, "response_gate_passed": bool(response_gate_passed),
            "tools": [
                {"name": name, "evidence_count": item.evidence_count,
                 "provenance_verified": item.provenance_verified}
                for name, item in observations.items()
            ],
        })
        self.last_episode_id = episode_id
        return episode_id

    def feedback(self, *, positive: bool, tool: str | None = None) -> None:
        """Accept external feedback; never treat the model's answer as its judge."""
        if type(positive) is not bool or (tool is not None and not KEY.fullmatch(tool)):
            raise ValueError("feedback must be explicit and tool must be an identifier")
        if not self.last_episode_id:
            raise ValueError("no completed meta episode")
        events = self.store.read()
        episode = next((event for event in reversed(events)
                        if event.get("kind") == "episode"
                        and event.get("id") == self.last_episode_id), None)
        if episode is None or (tool is not None and
                               tool not in {item["name"] for item in episode["tools"]}):
            raise ValueError("feedback tool was not used in the last episode")
        self.store.append({
            "kind": "feedback", "episode_id": self.last_episode_id,
            "positive": positive, "tool": tool,
        })

    def snapshot(self) -> dict[str, Any]:
        events = self.store.read()
        return {"episodes": sum(event.get("kind") == "episode" for event in events),
                "feedback": sum(event.get("kind") == "feedback" for event in events),
                "last_episode_id": self.last_episode_id}

    def improvement_cases(self) -> list[dict[str, str]]:
        """Expose compact failures to the existing offline candidate workbench."""
        events = self.store.read()
        feedback = {
            event["episode_id"]: event for event in events
            if event.get("kind") == "feedback"
        }
        cases: list[dict[str, str]] = []
        for episode in events:
            if episode.get("kind") != "episode":
                continue
            reasons = []
            if not episode["response_gate_passed"]:
                reasons.append(("response_gate_failed", ""))
            for item in episode["tools"]:
                if item["evidence_count"] == 0:
                    reasons.append(("tool_no_evidence", item["name"]))
            review = feedback.get(episode["id"])
            if review and review["positive"] is False:
                reasons.append(("negative_feedback", review.get("tool") or ""))
            for reason, tool in reasons:
                identity = f"{episode['id']}:{reason}:{tool}"
                cases.append({
                    "case_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    "task_key": episode["task_key"],
                    "gap": episode["gap"],
                    "reason": reason,
                    "tool": tool,
                })
        return cases
