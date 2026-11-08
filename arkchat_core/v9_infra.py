#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, dependency-light v9 AI-infra primitives.

The module deliberately keeps model loading out of the control plane.  It
provides two pieces that can be tested on CPU and used by a GPU runner:

* generation telemetry with separate input/prefill and decode timing fields;
* an atomic registry for fast surface-weight (for example, LoRA) activation.

Base weights remain immutable.  A surface-weight switch is a versioned,
rollbackable pointer update and is only applied at a turn boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import threading
import time
from typing import Any, Mapping
import uuid



INFRA_SCHEMA_VERSION = "arkchat-v9-infra-1"
SURFACE_SCHEMA_VERSION = "arkchat-v9-surface-weight-1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_artifact(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file or adapter directory, including relative names and bytes."""
    path = Path(path).resolve()
    if path.is_file():
        return sha256_file(path, chunk_size=chunk_size)
    if not path.is_dir():
        raise ValueError(f"artifact is missing or not a file/directory: {path}")
    digest = hashlib.sha256()
    entries = list(path.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise ValueError("artifact directory contains a symlink")
    files = sorted(item for item in entries if item.is_file())
    if not files:
        raise ValueError(f"artifact directory contains no files: {path}")
    for item in files:
        resolved = item.resolve()
        if not resolved.is_relative_to(path) or item.is_symlink():
            raise ValueError("artifact directory contains a symlink outside its root")
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


@dataclass
class GenerationTelemetry:
    """Serializable generation timing and token counters."""

    schema_version: str = INFRA_SCHEMA_VERSION
    input_tokens: int = 0
    output_tokens: int = 0
    prefill_ms: float | None = None
    decode_ms: float | None = None
    total_ms: float | None = None
    tokens_per_second: float | None = None
    peak_memory_mib: int | None = None
    gpu_synchronized: bool = False
    timing_source: str = "host_wall"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenerationTimer:
    """Measure a generation call without forcing GPU synchronization by default."""

    def __init__(self, *, synchronize_gpu: bool = False, torch_module: Any = None):
        self.synchronize_gpu = bool(synchronize_gpu)
        self.torch = torch_module
        self.started = 0.0
        self.finished = 0.0
        self._did_sync = False

    def _sync(self) -> None:
        if not self.synchronize_gpu or self.torch is None:
            return
        cuda = getattr(self.torch, "cuda", None)
        if cuda is not None and bool(cuda.is_available()):
            cuda.synchronize()
            self._did_sync = True

    def __enter__(self) -> "GenerationTimer":
        self._sync()
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._sync()
        self.finished = time.perf_counter()

    def finish(self, *, input_tokens: int, output_tokens: int,
               prefill_ms: float | None = None,
               peak_memory_mib: int | None = None) -> GenerationTelemetry:
        total_ms = max(0.0, (self.finished - self.started) * 1000.0)
        decode_ms = None
        if prefill_ms is not None:
            decode_ms = max(0.0, total_ms - float(prefill_ms))
        rate = None
        # Host wall-clock throughput is still useful when CUDA synchronization
        # is disabled. It remains approximate in that mode, but is preferable
        # to reporting no rate for ordinary CPU and asynchronous runs.
        if output_tokens > 0 and total_ms > 0:
            rate = output_tokens / (total_ms / 1000.0)
        return GenerationTelemetry(
            input_tokens=max(0, int(input_tokens)),
            output_tokens=max(0, int(output_tokens)),
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            total_ms=total_ms,
            tokens_per_second=rate,
            peak_memory_mib=peak_memory_mib,
            gpu_synchronized=self._did_sync,
            timing_source=("synchronized_host_wall" if self._did_sync
                           else "unsynchronized_host_wall"),
        )


def peak_memory_mib(torch_module: Any = None) -> int | None:
    """Read the optional CUDA allocator peak without requiring torch at import."""
    if torch_module is None:
        return None
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        return None
    try:
        return int(cuda.max_memory_allocated() // (1024 * 1024))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


@dataclass(frozen=True)
class SurfaceWeightSpec:
    version: str
    artifact: str
    sha256: str
    base_fingerprint: str
    kind: str = "lora"
    rank: int | None = None
    parameter_count: int | None = None
    created_at: str = ""
    parent_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SURFACE_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SurfaceWeightSpec":
        if value.get("schema_version") not in (None, SURFACE_SCHEMA_VERSION):
            raise ValueError("unsupported surface-weight schema")
        required = ("version", "artifact", "sha256", "base_fingerprint")
        if any(not isinstance(value.get(key), str) or not value[key] for key in required):
            raise ValueError("surface-weight manifest is missing required fields")
        rank = value.get("rank")
        if rank is not None and (type(rank) is not int or rank <= 0):
            raise ValueError("surface-weight rank must be a positive integer")
        return cls(
            version=value["version"], artifact=value["artifact"],
            sha256=value["sha256"], base_fingerprint=value["base_fingerprint"],
            kind=str(value.get("kind", "lora")), rank=rank,
            parameter_count=value.get("parameter_count"),
            created_at=str(value.get("created_at", "")),
            parent_version=value.get("parent_version"),
        )


class SurfaceWeightRegistry:
    """Register and atomically activate immutable surface-weight artifacts."""

    def __init__(self, root: Path, *, base_fingerprint: str):
        self.root = Path(root).resolve()
        self.base_fingerprint = str(base_fingerprint)
        if not self.base_fingerprint:
            raise ValueError("base_fingerprint is required")
        self.manifest_dir = self.root / "manifests"
        self.pointer_path = self.root / "active.json"

    def _manifest_path(self, version: str) -> Path:
        safe = "".join(ch for ch in str(version) if ch.isalnum() or ch in "._-")
        if not safe or safe != version or safe in {".", ".."} or len(safe) > 128:
            raise ValueError("invalid surface-weight version")
        return self.manifest_dir / f"{safe}.json"

    def register(self, *, version: str, artifact: Path, kind: str = "lora",
                 rank: int | None = None, parameter_count: int | None = None,
                 parent_version: str | None = None) -> SurfaceWeightSpec:
        artifact = Path(artifact).resolve()
        if not (artifact.is_file() or artifact.is_dir()):
            raise ValueError(f"surface-weight artifact is missing or empty: {artifact}")
        artifact_hash = sha256_artifact(artifact)
        manifest_path = self._manifest_path(version)
        if artifact.is_dir() and self.root.is_relative_to(artifact):
            raise ValueError("registry cannot be placed inside the source artifact")
        if rank is not None and (type(rank) is not int or rank <= 0):
            raise ValueError("rank must be a positive integer")
        if parameter_count is not None and (type(parameter_count) is not int or parameter_count < 0):
            raise ValueError("parameter_count must be a non-negative integer")
        if parent_version is not None:
            if parent_version == version:
                raise ValueError("surface-weight version cannot parent itself")
            self.load(parent_version)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = manifest_path.with_suffix(".lock")
        # Exclusive creation prevents two processes from rewriting one version.
        lock = lock_path.open("x", encoding="utf-8")
        try:
            if manifest_path.exists():
                existing = self.load(version)
                if (existing.sha256, existing.kind, existing.rank,
                        existing.parameter_count, existing.parent_version) != (
                        artifact_hash, kind, rank, parameter_count, parent_version):
                    raise FileExistsError("surface-weight versions are immutable")
                return existing
            artifacts = self.root / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            destination = artifacts / version
            if not destination.exists():
                with tempfile.TemporaryDirectory(prefix=".register-", dir=artifacts) as temp:
                    staging = Path(temp) / "snapshot"
                    staging.mkdir()
                    if artifact.is_dir():
                        shutil.copytree(artifact, staging / "artifact")
                    else:
                        shutil.copy2(artifact, staging / "artifact")
                    if sha256_artifact(staging / "artifact") != artifact_hash:
                        raise ValueError("artifact changed during registration")
                    staging.rename(destination)
            snapshot = destination / "artifact"
            if sha256_artifact(snapshot) != artifact_hash:
                raise FileExistsError("surface-weight snapshot is immutable")
            spec = SurfaceWeightSpec(
                version=version, artifact=str(snapshot), sha256=artifact_hash,
                base_fingerprint=self.base_fingerprint, kind=kind, rank=rank,
                parameter_count=parameter_count, created_at=utc_now(),
                parent_version=parent_version,
            )
            _atomic_json(manifest_path, spec.to_dict())
            return spec
        finally:
            lock.close()
            lock_path.unlink(missing_ok=True)

    def load(self, version: str) -> SurfaceWeightSpec:
        value = json.loads(self._manifest_path(version).read_text(encoding="utf-8"))
        spec = SurfaceWeightSpec.from_dict(value)
        if spec.version != version:
            raise ValueError("surface-weight manifest version mismatch")
        if spec.base_fingerprint != self.base_fingerprint:
            raise ValueError("surface-weight base fingerprint mismatch")
        artifact = Path(spec.artifact)
        if (not (artifact.is_file() or artifact.is_dir())
                or sha256_artifact(artifact) != spec.sha256):
            raise ValueError("surface-weight artifact hash mismatch")
        return spec

    def activate(self, version: str) -> SurfaceWeightSpec:
        spec = self.load(version)
        _atomic_json(self.pointer_path, {
            "schema_version": SURFACE_SCHEMA_VERSION,
            "active_version": spec.version,
            "activated_at": utc_now(),
        })
        return spec

    def active(self) -> SurfaceWeightSpec | None:
        try:
            pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(pointer, dict) or not isinstance(pointer.get("active_version"), str):
            raise ValueError("invalid surface-weight active pointer")
        return self.load(pointer["active_version"])

    def rollback(self, version: str | None = None) -> SurfaceWeightSpec | None:
        if version is None:
            current = self.active()
            version = current.parent_version if current else None
        if not version:
            self.pointer_path.unlink(missing_ok=True)
            return None
        return self.activate(version)


class FastLowRankSurfaceAdapter:
    """Per-conversation low-rank residual head updated from explicit feedback.

    This is for a small policy/appraisal head, not the Qwen 14B LM head.  The
    caller supplies a learned feature vector and base-head logits.  Only the
    low-rank residual coefficients change; the base projection is immutable.
    """

    schema_version = "arkchat-v9-fast-head-1"

    def __init__(self, feature_dim: int, output_dim: int, *, rank: int = 4,
                 seed: int = 17, learning_rate: float = 0.05,
                 regularization: float = 0.1, max_delta_norm: float = 0.5,
                 base_fingerprint: str = ""):
        if (any(type(value) is not int for value in (feature_dim, output_dim, rank))
                or feature_dim < 1 or output_dim < 1 or rank < 1
                or rank > min(feature_dim, output_dim)):
            raise ValueError("invalid fast-adapter dimensions or rank")
        if (not all(math.isfinite(value) for value in (learning_rate, regularization, max_delta_norm))
                or not 0 < learning_rate <= 1 or regularization < 0 or max_delta_norm <= 0):
            raise ValueError("invalid fast-adapter update bounds")
        self.feature_dim = int(feature_dim)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.max_delta_norm = float(max_delta_norm)
        self.base_fingerprint = str(base_fingerprint)
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(self.feature_dim)
        self.projection = [
            [rng.gauss(0.0, scale) for _ in range(self.rank)]
            for _ in range(self.feature_dim)
        ]
        self.coefficients = [
            [0.0 for _ in range(self.output_dim)] for _ in range(self.rank)
        ]
        self.update_count = 0

    @property
    def delta_norm(self) -> float:
        total = 0.0
        for feature in range(self.feature_dim):
            for output in range(self.output_dim):
                value = sum(
                    self.projection[feature][rank] * self.coefficients[rank][output]
                    for rank in range(self.rank)
                )
                total += value * value
        return math.sqrt(total)

    @staticmethod
    def _vector(value: Any, name: str) -> list[float]:
        if not isinstance(value, (list, tuple)):
            try:
                value = list(value)
            except TypeError as exc:
                raise ValueError(f"{name} must be a one-dimensional sequence") from exc
        try:
            result = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain finite numbers") from exc
        if not all(math.isfinite(item) for item in result):
            raise ValueError(f"{name} must contain finite numbers")
        return result

    def predict(self, features: Any, base_output: Any) -> list[float]:
        x = self._vector(features, "features")
        y = self._vector(base_output, "base_output")
        if len(x) != self.feature_dim or len(y) != self.output_dim:
            raise ValueError("fast-adapter input/output dimension mismatch")
        latent = [
            sum(x[index] * self.projection[index][rank]
                for index in range(self.feature_dim))
            for rank in range(self.rank)
        ]
        result = [
            y[output] + sum(
                latent[rank] * self.coefficients[rank][output]
                for rank in range(self.rank)
            )
            for output in range(self.output_dim)
        ]
        if not all(math.isfinite(value) for value in result):
            raise ValueError("fast-adapter prediction overflow")
        return result

    def update(self, features: Any, base_output: Any, target: Any, *,
               explicit_feedback: bool = False) -> dict[str, float | int]:
        if not explicit_feedback:
            raise PermissionError("fast weights update only from explicit feedback")
        x = self._vector(features, "features")
        y = self._vector(base_output, "base_output")
        target = self._vector(target, "target")
        if (len(x) != self.feature_dim or len(y) != self.output_dim
                or len(target) != self.output_dim):
            raise ValueError("fast-adapter input/output dimension mismatch")
        latent = [
            sum(x[index] * self.projection[index][rank]
                for index in range(self.feature_dim))
            for rank in range(self.rank)
        ]
        before = self.predict(x, y)
        residual = [target[index] - before[index] for index in range(self.output_dim)]
        loss_before = sum(value * value for value in residual) / self.output_dim
        denominator = self.regularization + sum(value * value for value in latent)
        if not math.isfinite(denominator) or not math.isfinite(loss_before):
            raise ValueError("fast-adapter update overflow")
        denominator = max(denominator, 1e-8)
        previous = [list(row) for row in self.coefficients]
        coefficients = [list(row) for row in self.coefficients]
        for rank in range(self.rank):
            for output in range(self.output_dim):
                coefficients[rank][output] += (
                    self.learning_rate * latent[rank] * residual[output] / denominator
                )
        if not all(math.isfinite(value) for row in coefficients for value in row):
            raise ValueError("fast-adapter update overflow")
        self.coefficients = coefficients
        norm = self.delta_norm
        if not math.isfinite(norm):
            self.coefficients = previous
            raise ValueError("fast-adapter parameter norm overflow")
        if norm > self.max_delta_norm:
            scale = self.max_delta_norm / norm
            self.coefficients = [
                [value * scale for value in row] for row in self.coefficients
            ]
        try:
            after = self.predict(x, y)
            loss_after = sum((target[i] - after[i]) ** 2 for i in range(self.output_dim)) / self.output_dim
            if not math.isfinite(loss_after):
                raise ValueError("fast-adapter loss overflow")
        except (ValueError, OverflowError):
            self.coefficients = previous
            raise
        self.update_count += 1
        return {
            "update_count": self.update_count,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "delta_norm": self.delta_norm,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_dim": self.feature_dim,
            "output_dim": self.output_dim,
            "rank": self.rank,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "max_delta_norm": self.max_delta_norm,
            "base_fingerprint": self.base_fingerprint,
            "projection": [list(row) for row in self.projection],
            "coefficients": [list(row) for row in self.coefficients],
            "update_count": self.update_count,
            "updated_at": utc_now(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *,
                  expected_base_fingerprint: str | None = None) -> "FastLowRankSurfaceAdapter":
        if value.get("schema_version") != cls.schema_version:
            raise ValueError("unsupported fast-adapter schema")
        fingerprint = str(value.get("base_fingerprint", ""))
        if expected_base_fingerprint is not None and fingerprint != expected_base_fingerprint:
            raise ValueError("fast-adapter base fingerprint mismatch")
        adapter = cls(
            int(value["feature_dim"]), int(value["output_dim"]),
            rank=int(value["rank"]),
            learning_rate=float(value["learning_rate"]),
            regularization=float(value["regularization"]),
            max_delta_norm=float(value["max_delta_norm"]),
            base_fingerprint=fingerprint,
        )
        projection = [
            cls._vector(row, "projection") for row in value["projection"]
        ]
        coefficients = [
            cls._vector(row, "coefficients") for row in value["coefficients"]
        ]
        if (len(projection) != adapter.feature_dim
                or any(len(row) != adapter.rank for row in projection)
                or len(coefficients) != adapter.rank
                or any(len(row) != adapter.output_dim for row in coefficients)):
            raise ValueError("invalid fast-adapter parameter arrays")
        adapter.projection = projection
        adapter.coefficients = coefficients
        adapter.update_count = max(0, int(value.get("update_count", 0)))
        if not math.isfinite(adapter.delta_norm) or adapter.delta_norm > adapter.max_delta_norm + 1e-6:
            raise ValueError("fast-adapter delta exceeds its norm bound")
        return adapter


class FastSurfaceStateStore:
    """Persist hashed low-rank snapshots with an atomic, rollbackable pointer."""

    pointer_schema = "arkchat-v9-fast-surface-pointer-1"

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, conversation_id: str) -> Path:
        digest = hashlib.sha256(str(conversation_id).encode("utf-8")).hexdigest()
        return self.root / f"{digest}.fast-surface.json"

    def save(self, conversation_id: str | None,
             adapter: FastLowRankSurfaceAdapter) -> str | None:
        if not conversation_id:
            return None
        state = adapter.to_dict()
        FastLowRankSurfaceAdapter.from_dict(state)
        encoded = json.dumps(state, sort_keys=True, ensure_ascii=False, allow_nan=False)
        version = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        directory = self._version_dir(conversation_id)
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = directory / f"{version}.json"
        try:
            with snapshot.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        except FileExistsError:
            if snapshot.read_text(encoding="utf-8") != encoded:
                raise ValueError("fast-surface snapshot hash mismatch")
        _atomic_json(self._path(conversation_id), {
            "schema_version": self.pointer_schema,
            "version": version,
        })
        return version

    def _version_dir(self, conversation_id: str) -> Path:
        return self.root / "versions" / self._path(conversation_id).stem

    def _load_version(self, conversation_id: str, version: str,
                      expected_base_fingerprint: str | None = None) -> FastLowRankSurfaceAdapter:
        if len(version) != 64 or any(ch not in "0123456789abcdef" for ch in version):
            raise ValueError("invalid fast-surface snapshot version")
        encoded = (self._version_dir(conversation_id) / f"{version}.json").read_text(encoding="utf-8")
        if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != version:
            raise ValueError("fast-surface snapshot hash mismatch")
        return FastLowRankSurfaceAdapter.from_dict(
            json.loads(encoded), expected_base_fingerprint=expected_base_fingerprint
        )

    def load(self, conversation_id: str | None, *,
             expected_base_fingerprint: str | None = None) -> FastLowRankSurfaceAdapter | None:
        if not conversation_id:
            return None
        try:
            value = json.loads(self._path(conversation_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if value.get("schema_version") == self.pointer_schema:
            return self._load_version(conversation_id, value["version"], expected_base_fingerprint)
        # Read snapshots written by the first v9 prototype without migration loss.
        return FastLowRankSurfaceAdapter.from_dict(
            value, expected_base_fingerprint=expected_base_fingerprint
        )

    def rollback(self, conversation_id: str, version: str, *,
                 expected_base_fingerprint: str | None = None) -> FastLowRankSurfaceAdapter:
        adapter = self._load_version(conversation_id, version, expected_base_fingerprint)
        _atomic_json(self._path(conversation_id), {
            "schema_version": self.pointer_schema, "version": version,
        })
        return adapter

    def forget(self, conversation_id: str | None) -> bool:
        if not conversation_id:
            return False
        path = self._path(conversation_id)
        existed = path.exists()
        versions = self._version_dir(conversation_id)
        existed = existed or versions.exists()
        if versions.exists():
            # This directory is derived only from the hashed conversation key.
            shutil.rmtree(versions)
        path.unlink(missing_ok=True)
        return existed


class FastSurfaceWeightManager:
    """Apply a registered adapter to a PEFT-like model at a turn boundary."""

    def __init__(self, registry: SurfaceWeightRegistry):
        self.registry = registry
        self.loaded: dict[str, str] = {}
        self.current_model_adapter: str | None = None
        self.original_model_adapter: str | None = None
        self._bound_model: Any = None
        self._disabled = False
        self._lock = threading.RLock()

    def _bind_model(self, model: Any) -> None:
        if self._bound_model is not None and self._bound_model is not model:
            raise ValueError("surface manager is already bound to another model")
        if self._bound_model is None:
            original = getattr(model, "active_adapter", None)
            if original is not None and not isinstance(original, str):
                raise ValueError("surface switching requires a single active adapter")
            self.original_model_adapter = original
            status = getattr(model, "get_model_status", None)
            if callable(status) and status().enabled is False:
                self.original_model_adapter = None
            self.current_model_adapter = self.original_model_adapter
            self._bound_model = model

    @staticmethod
    def _freeze(model: Any) -> None:
        # PEFT set_adapter() can mark the chosen adapter trainable again.
        if callable(getattr(model, "parameters", None)):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        if callable(getattr(model, "eval", None)):
            model.eval()

    def _set_active(self, model: Any, name: str | None) -> None:
        tuner = getattr(model, "base_model", model)
        if name is None:
            disable = getattr(tuner, "disable_adapter_layers", None)
            if not callable(disable):
                disable = getattr(model, "disable_adapters", None)
            if not callable(disable):
                raise TypeError("model does not support persistent adapter disabling")
            disable()
            self._disabled = True
        else:
            model.set_adapter(name)
            enable = getattr(tuner, "enable_adapter_layers", None)
            if not callable(enable):
                enable = getattr(model, "enable_adapters", None)
            if callable(enable):
                enable()
            self._disabled = False
        self._freeze(model)

    def preload(self, model: Any, version: str, *, adapter_name: str | None = None) -> str:
        """Load a candidate without activating it; preload before serving traffic."""
        with self._lock:
            self._bind_model(model)
            spec = self.registry.load(version)
            if spec.kind != "lora":
                raise ValueError("PEFT surface switching supports registered LoRA artifacts only")
            name = adapter_name or spec.version
            if name in self.loaded and self.loaded[name] != spec.version:
                raise ValueError("adapter name is already bound to another version")
            if name not in self.loaded:
                if name in getattr(model, "peft_config", {}):
                    raise ValueError("adapter name conflicts with an existing model adapter")
                if not hasattr(model, "load_adapter"):
                    raise TypeError("model does not expose the PEFT adapter interface")
                model.load_adapter(spec.artifact, adapter_name=name, is_trainable=False)
                self._freeze(model)
                self.loaded[name] = spec.version
            return name

    def activate_on_model(self, model: Any, version: str, *, adapter_name: str | None = None) -> SurfaceWeightSpec:
        """Switch a preloaded adapter at a caller-enforced turn boundary."""
        with self._lock:
            self._bind_model(model)
            previous_model_adapter = self.current_model_adapter
            name = self.preload(model, version, adapter_name=adapter_name)
            try:
                self._set_active(model, name)
                spec = self.registry.activate(version)
                self.current_model_adapter = name
                return spec
            except Exception:
                # The registry pointer only changes after a successful atomic
                # write. Restore model selection if switching or writing fails.
                self._set_active(model, previous_model_adapter)
                raise

    def rollback_on_model(self, model: Any, version: str | None = None) -> SurfaceWeightSpec | None:
        """Switch both the active model and registry back to an earlier version."""
        with self._lock:
            self._bind_model(model)
            target = version
            if target is None:
                current = self.registry.active()
                target = current.parent_version if current else None
            if not target:
                previous = self.current_model_adapter
                try:
                    self._set_active(model, self.original_model_adapter)
                    self.registry.pointer_path.unlink(missing_ok=True)
                except Exception:
                    self._set_active(model, previous)
                    raise
                self.current_model_adapter = self.original_model_adapter
                return None
            return self.activate_on_model(model, target)


__all__ = [
    "FastLowRankSurfaceAdapter", "FastSurfaceStateStore",
    "FastSurfaceWeightManager", "GenerationTelemetry", "GenerationTimer",
    "INFRA_SCHEMA_VERSION", "SurfaceWeightRegistry", "SurfaceWeightSpec",
    "peak_memory_mib", "sha256_artifact", "sha256_file",
]
