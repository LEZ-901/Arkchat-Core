#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent conversation history and bounded long-term recall."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


VALID_ROLES = {"user", "assistant"}
WORD = re.compile(r"[a-zA-Z0-9_]{2,}|[\u4e00-\u9fff]")
SALIENT_USER_FACT = re.compile(
    r"(我叫|我的.{0,10}(?:是|叫)|我(?:最)?喜欢|我不喜欢|我讨厌|"
    r"我住在|我来自|我的生日|记住|别忘|以后|约定|答应我|"
    r"我打算|我要在|我会在)"
)
SENSITIVE_MEMORY = re.compile(
    r"(密码|口令|验证码|身份证|护照|银行卡|信用卡|私钥|api[ _-]?key|"
    r"\b\d{15,19}\b|\b1[3-9]\d{9}\b|"
    r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})",
    re.IGNORECASE,
)
MEMORY_SCHEMA_VERSION = "arkchat-conversation-memory-v9-1"
MAX_MEMORY_CHARS = 300


def _clean_message(message: object) -> dict | None:
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    content = message.get("content")
    if role not in VALID_ROLES or not isinstance(content, str):
        return None
    content = content.strip()
    if not content:
        return None
    return {"role": role, "content": content}


def _terms(text: str) -> set[str]:
    chars = WORD.findall(text.lower())
    terms = set(chars)
    chinese = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
    terms.update(chinese[i:i + 2] for i in range(max(0, len(chinese) - 1)))
    return terms


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _normal_text(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def _memory_category(text: str) -> str:
    if re.search(r"(喜欢|不喜欢|讨厌|偏好)", text):
        return "preference"
    if re.search(r"(打算|计划|我要在|我会在|约定|答应我|以后)", text):
        return "commitment"
    return "user_fact"


def _memory_priority(text: str) -> float:
    """Estimate review priority from explicit future utility, not sentiment."""
    score = 0.45
    if re.search(r"(记住|别忘|以后|约定|答应我)", text):
        score += 0.35
    category = _memory_category(text)
    if category == "preference":
        score += 0.12
    elif category == "commitment":
        score += 0.08
    # Prefer compact notes when usefulness is otherwise similar.
    score -= min(0.15, max(0, len(_normal_text(text)) - 80) / 1000.0)
    return round(min(1.0, max(0.0, score)), 4)


def propose_user_memories(
    history: list[dict], *, recent_messages: int = 2, limit: int = 8
) -> list[dict]:
    """Create review candidates from explicit user statements only.

    This deterministic pass never persists or injects candidates by itself.
    Assistant statements, sensitive strings, and oversized messages are skipped.
    """
    if not history or limit <= 0:
        return []
    cutoff = max(0, len(history) - max(0, recent_messages))
    seen: set[str] = set()
    candidates: list[dict] = []
    for index in range(cutoff - 1, -1, -1):
        clean = _clean_message(history[index])
        if not clean or clean["role"] != "user":
            continue
        content = clean["content"]
        normal = _normal_text(content)
        if (not normal or len(content) > MAX_MEMORY_CHARS
                or SENSITIVE_MEMORY.search(content)
                or not SALIENT_USER_FACT.search(content)
                or normal in seen):
            continue
        seen.add(normal)
        candidates.append({
            "source_turn": index // 2 + 1,
            "category": _memory_category(content),
            "content": content,
            "reason": "explicit_user_statement",
            "source_role": "user",
            "source_confidence": 1.0,
            "priority": _memory_priority(content),
            "storage_cost_chars": len(content),
        })
        if len(candidates) >= limit:
            break
    return list(reversed(candidates))


class ReviewableMemoryStore:
    """Versioned long-term memories kept separate from raw conversation history.

    Candidates require explicit approval before retrieval. Updates supersede
    prior versions, and deleting a memory removes that curated record only;
    the source conversation remains until the conversation itself is reset.
    """

    def __init__(self, root: Path, *, max_pending: int = 64,
                 max_records: int = 512, recall_char_budget: int = 2400):
        if min(max_pending, max_records, recall_char_budget) <= 0:
            raise ValueError("curated memory budgets must be positive")
        self.root = Path(root)
        self.max_pending = max_pending
        self.max_records = max_records
        self.recall_char_budget = recall_char_budget

    def _path(self, conversation_id: str) -> Path:
        digest = hashlib.sha256(str(conversation_id).encode("utf-8")).hexdigest()
        return self.root / f"{digest}.memory-v9.json"

    @staticmethod
    def _empty() -> dict:
        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "pending": [],
            "records": [],
            "suppressed": [],
        }

    @staticmethod
    def _suppression_key(conversation_id: str, content: str) -> str:
        # A deduplication tombstone, not an anonymity guarantee. The original
        # conversation remains separately managed until the user resets it.
        return hashlib.sha256(
            f"{conversation_id}:{_normal_text(content)}".encode("utf-8")
        ).hexdigest()

    def _read(self, conversation_id: str) -> dict:
        if not conversation_id:
            return self._empty()
        try:
            value = json.loads(self._path(conversation_id).read_text(encoding="utf-8"))
            if (not isinstance(value, dict)
                    or value.get("schema_version") != MEMORY_SCHEMA_VERSION
                    or not isinstance(value.get("pending"), list)
                    or not isinstance(value.get("records"), list)):
                return self._empty()
            value["pending"] = [row for row in value["pending"] if isinstance(row, dict)]
            value["records"] = [row for row in value["records"] if isinstance(row, dict)]
            value["suppressed"] = [
                key for key in value.get("suppressed", [])
                if isinstance(key, str) and re.fullmatch(r"[0-9a-f]{64}", key)
            ]
            return value
        except (FileNotFoundError, OSError, ValueError, TypeError):
            return self._empty()

    def _write(self, conversation_id: str, value: dict) -> None:
        if not conversation_id:
            return
        path = self._path(conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._empty()
        payload.update(value)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)

    def propose(self, conversation_id: str | None, history: list[dict],
                *, recent_messages: int = 2, limit: int = 8) -> list[dict]:
        if not conversation_id:
            return []
        value = self._read(conversation_id)
        now = _utc_now()
        previous_pending_count = len(value["pending"])
        value["pending"] = [
            item for item in value["pending"]
            if not self._expired(item, now)
        ]
        known = {
            _normal_text(item.get("content", ""))
            for item in value["pending"] + value["records"]
        }
        suppressed = set(value["suppressed"])
        added = []
        candidates = propose_user_memories(history, recent_messages=recent_messages, limit=limit)
        candidates.sort(key=lambda row: (row["priority"], row["source_turn"]), reverse=True)
        for candidate in candidates:
            normal = _normal_text(candidate["content"])
            if (normal in known or self._suppression_key(
                    conversation_id, candidate["content"]) in suppressed):
                continue
            if len(value["pending"]) >= self.max_pending:
                break
            candidate = {
                **candidate,
                "memory_id": hashlib.sha256(
                    f"{conversation_id}:{candidate['source_turn']}:{normal}".encode("utf-8")
                ).hexdigest()[:16],
                "status": "pending",
                "created_at": _iso(_utc_now()),
                "expires_at": _iso(_utc_now() + timedelta(days=7)),
            }
            value["pending"].append(candidate)
            known.add(normal)
            added.append(candidate)
        if added or previous_pending_count != len(value["pending"]):
            self._write(conversation_id, value)
        return added

    @staticmethod
    def _expired(item: dict, now: datetime) -> bool:
        expires_at = item.get("expires_at")
        if not expires_at:
            return False
        try:
            expiry = datetime.fromisoformat(expires_at)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            return expiry <= now
        except (TypeError, ValueError):
            return True

    def list_records(self, conversation_id: str | None, *,
                     status: str = "active") -> list[dict]:
        if not conversation_id:
            return []
        value = self._read(conversation_id)
        records = value["pending"] if status == "pending" else value["records"]
        now = _utc_now()
        result = []
        for item in records:
            if status != "pending" and item.get("status") != status:
                continue
            if status in {"active", "pending"} and self._expired(item, now):
                continue
            result.append(dict(item))
        result.sort(key=lambda item: (
            item.get("updated_at", item.get("created_at", "")),
            item.get("source_turn", 0),
        ), reverse=True)
        return result

    def approve(self, conversation_id: str | None, memory_id: str,
                *, ttl_days: int | None = None) -> dict | None:
        if not conversation_id:
            return None
        if ttl_days is not None and (type(ttl_days) is not int or ttl_days <= 0):
            raise ValueError("ttl_days must be positive")
        value = self._read(conversation_id)
        for index, item in enumerate(value["pending"]):
            if item.get("memory_id") != memory_id:
                continue
            now = _utc_now()
            if self._expired(item, now):
                value["pending"].pop(index)
                self._write(conversation_id, value)
                return None
            if len(value["records"]) >= self.max_records:
                raise ValueError("curated memory capacity reached; forget a record first")
            record = {
                **item,
                "status": "active",
                "provenance": "user_approved",
                "approved_at": _iso(now),
                "updated_at": _iso(now),
                "expires_at": _iso(now + timedelta(days=ttl_days))
                if ttl_days is not None else None,
            }
            value["pending"].pop(index)
            value["records"].append(record)
            self._write(conversation_id, value)
            return record
        return None

    def update(self, conversation_id: str | None, memory_id: str,
               content: str) -> dict | None:
        content = str(content or "").strip()
        if not content or len(content) > MAX_MEMORY_CHARS:
            raise ValueError(f"memory content must be 1–{MAX_MEMORY_CHARS} characters")
        if SENSITIVE_MEMORY.search(content):
            raise ValueError("sensitive information cannot be stored as long-term memory")
        if not conversation_id:
            return None
        value = self._read(conversation_id)
        for old in value["records"]:
            if old.get("memory_id") != memory_id or old.get("status") != "active":
                continue
            if self._expired(old, _utc_now()):
                return None
            if _normal_text(old.get("content", "")) == _normal_text(content):
                return dict(old)
            if len(value["records"]) >= self.max_records:
                raise ValueError("curated memory capacity reached; forget a record first")
            now = _iso(_utc_now())
            new_id = uuid.uuid4().hex[:16]
            old["status"] = "superseded"
            old["superseded_by"] = new_id
            old["updated_at"] = now
            record = {
                **old,
                "memory_id": new_id,
                "content": content,
                "category": _memory_category(content),
                "priority": _memory_priority(content),
                "storage_cost_chars": len(content),
                "status": "active",
                "provenance": "user_correction",
                "supersedes": memory_id,
                "superseded_by": None,
                "created_at": now,
                "updated_at": now,
            }
            value["records"].append(record)
            self._write(conversation_id, value)
            return record
        return None

    def forget(self, conversation_id: str | None, memory_id: str) -> bool:
        if not conversation_id:
            return False
        value = self._read(conversation_id)
        related = {memory_id}
        changed = True
        while changed:
            changed = False
            for item in value["records"]:
                current = item.get("memory_id")
                links = {item.get("supersedes"), item.get("superseded_by")}
                if current in related or links & related:
                    before_size = len(related)
                    related.add(current)
                    related.update(link for link in links if link)
                    changed = changed or len(related) != before_size
        before = len(value["records"]) + len(value["pending"])
        removed_records = [
            item for item in value["records"] + value["pending"]
            if item.get("memory_id") in related
        ]
        suppressed = set(value["suppressed"])
        suppressed.update(self._suppression_key(conversation_id, item.get("content", ""))
                          for item in removed_records)
        value["suppressed"] = sorted(suppressed)
        value["records"] = [
            item for item in value["records"]
            if item.get("memory_id") not in related
        ]
        value["pending"] = [
            item for item in value["pending"]
            if item.get("memory_id") != memory_id
        ]
        if before == len(value["records"]) + len(value["pending"]):
            return False
        self._write(conversation_id, value)
        return True

    def retrieve(self, conversation_id: str | None, query: str,
                 *, limit: int = 8, max_chars: int | None = None) -> list[str]:
        if not conversation_id or limit <= 0:
            return []
        query_terms = _terms(query)
        now = _utc_now()
        ranked = []
        for index, item in enumerate(self.list_records(conversation_id)):
            terms = _terms(item.get("content", ""))
            shared = query_terms & terms
            # A lone common Chinese character (e.g. "请") is not evidence that
            # a tea preference is relevant to an identity-boundary question.
            meaningful = {term for term in shared if len(term) >= 2}
            if query_terms and not meaningful:
                continue
            score = len(meaningful) / math.sqrt(max(1, len(query_terms) * len(terms)))
            score += 0.02 * float(item.get("priority", 0.0) or 0.0)
            if item.get("category") == "preference":
                score += 0.03
            updated = item.get("updated_at", item.get("created_at", ""))
            try:
                age_days = max(0.0, (now - datetime.fromisoformat(updated)).total_seconds() / 86400)
                score += 0.05 / (1.0 + age_days / 30.0)
            except (TypeError, ValueError):
                pass
            ranked.append((score, index, item.get("content", "")))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        remaining = self.recall_char_budget if max_chars is None else max(0, max_chars)
        recalled = []
        for _, _, content in ranked:
            note = f"用户曾明确提供：{content}"
            if len(note) > remaining:
                continue
            recalled.append(note)
            remaining -= len(note)
            if len(recalled) >= limit:
                break
        return recalled

    def reset(self, conversation_id: str | None) -> None:
        if conversation_id:
            self._path(conversation_id).unlink(missing_ok=True)


def select_user_memories(history: list[dict], query: str,
                         recent_messages: int = 8,
                         limit: int = 8) -> list[str]:
    """Recall older user statements without treating assistant text as fact."""
    if limit <= 0 or not history:
        return []
    cutoff = max(0, len(history) - max(0, recent_messages))
    older = history[:cutoff]
    query_terms = _terms(query)
    ranked = []
    total = max(1, len(older))
    for index, message in enumerate(older):
        clean = _clean_message(message)
        if not clean or clean["role"] != "user":
            continue
        content = clean["content"]
        terms = _terms(content)
        overlap = len(query_terms & terms)
        relevant = overlap > 0
        salient = bool(SALIENT_USER_FACT.search(content))
        if not relevant and not salient:
            continue
        overlap_score = overlap / math.sqrt(max(1, len(query_terms) * len(terms)))
        score = overlap_score + (0.75 if salient else 0.0) + index / total * 0.05
        ranked.append((score, index, content))
    selected = sorted(ranked, reverse=True)[:limit]
    selected.sort(key=lambda item: item[1])
    return [f"用户曾说：{content}" for _, _, content in selected]


class ConversationMemoryStore:
    def __init__(self, root: Path, max_messages: int = 400):
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        self.root = Path(root)
        self.max_messages = max_messages

    def _path(self, conversation_id: str) -> Path:
        digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def load(self, conversation_id: str) -> list[dict]:
        path = self._path(conversation_id)
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return []
        if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list):
            return []
        messages = [clean for item in raw["messages"]
                    if (clean := _clean_message(item)) is not None]
        return messages[-self.max_messages:]

    def save(self, conversation_id: str, messages: list[dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        clean = [valid for item in messages
                 if (valid := _clean_message(item)) is not None]
        clean = clean[-self.max_messages:]
        path = self._path(conversation_id)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        payload = {"version": 1, "messages": clean}
        try:
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)

    def delete(self, conversation_id: str) -> None:
        self._path(conversation_id).unlink(missing_ok=True)
