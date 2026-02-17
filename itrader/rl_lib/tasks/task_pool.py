"""
TaskPool: Persistent task storage with JSONL backend and in-memory indexing.

Tasks are stored as JSONL with a lightweight in-memory index for fast sampling.
Each task has a stable UID based on a canonical hash of its TaskSpec dict.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import hashlib
from collections import defaultdict
import numpy as np


def _canonical_task_dict(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize task dict for stable hashing (sorted keys, consistent structure)."""
    # Deep copy and sort keys recursively
    def _sort_dict(d):
        if isinstance(d, dict):
            return {k: _sort_dict(v) for k, v in sorted(d.items())}
        elif isinstance(d, list):
            return [_sort_dict(item) for item in d]
        else:
            return d
    return _sort_dict(task_dict)


def _task_uid(task_dict: Dict[str, Any]) -> str:
    """Generate stable UID from canonical TaskSpec dict."""
    canon = _canonical_task_dict(task_dict)
    # JSON with sorted keys for determinism
    json_str = json.dumps(canon, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:16]


def task_uid(task_dict: Dict[str, Any]) -> str:
    """Public API to compute task UID from task dict."""
    return _task_uid(task_dict)


class TaskPool:
    """
    Persistent task pool with JSONL storage and in-memory indexing.

    APIs:
    - add(task_dict, meta=None): Add task to pool (append to JSONL)
    - sample(batch_size, strategy="weighted"): Sample tasks with various strategies
    - update_stats(task_uid, episode_return, success, extra={}): Update performance stats
    - prune(max_size, keep_elite_k): Prune pool keeping top performers
    """

    def __init__(self, jsonl_path: str | Path, read_only: bool = False):
        """
        Args:
            jsonl_path: Path to JSONL file (created if doesn't exist)
            read_only: If True, skip writes (for training-time sampling)
        """
        self.jsonl_path = Path(jsonl_path)
        self.read_only = read_only
        self._index: Dict[str, Dict[str, Any]] = {}  # uid -> {task, meta, stats}
        self._stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "returns": [],
            "successes": [],
            "count": 0,
            "mean_return": 0.0,
            "mean_success": 0.0,
        })
        self._load()

    def _load(self):
        """Load tasks from JSONL into memory index."""
        if not self.jsonl_path.exists():
            return
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    task_dict = entry.get("task", {})
                    meta = entry.get("meta", {})
                    uid = _task_uid(task_dict)
                    self._index[uid] = {"task": task_dict, "meta": meta}
                    # Initialize stats if not present
                    if uid not in self._stats:
                        self._stats[uid] = {
                            "returns": [],
                            "successes": [],
                            "count": 0,
                            "mean_return": 0.0,
                            "mean_success": 0.0,
                        }
                except Exception as e:
                    # Skip malformed lines
                    continue

    def add(self, task_dict: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> str:
        """
        Add task to pool. Returns task UID.

        Args:
            task_dict: TaskSpec as dict (from TaskSpec.to_dict())
            meta: Optional metadata (e.g., generation timestamp, source)
        """
        uid = _task_uid(task_dict)
        if uid in self._index:
            # Already exists, skip duplicate
            return uid

        self._index[uid] = {"task": task_dict, "meta": meta or {}}

        if not self.read_only:
            entry = {"task": task_dict, "meta": meta or {}}
            with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, separators=(',', ':')) + '\n')

        return uid

    def sample(self, batch_size: int = 1, strategy: str = "weighted") -> List[Dict[str, Any]]:
        """
        Sample tasks from pool.

        Args:
            batch_size: Number of tasks to sample
            strategy: "weighted" (mix of elite + recent + random), "uniform", "elite", "recent"

        Returns:
            List of task dicts
        """
        if not self._index:
            return []

        uids = list(self._index.keys())
        if len(uids) == 0:
            return []

        # For now, simple strategies (can be enhanced later)
        if strategy == "uniform":
            selected_uids = np.random.choice(uids, size=min(batch_size, len(uids)), replace=False).tolist()
        elif strategy == "elite":
            # Sort by mean_return, take top
            scored = [(uid, self._stats[uid]["mean_return"]) for uid in uids]
            scored.sort(key=lambda x: x[1], reverse=True)
            elite_k = min(batch_size, len(uids) // 4)  # top 25%
            selected_uids = [uid for uid, _ in scored[:elite_k]]
            if len(selected_uids) < batch_size:
                # Fill remainder with random
                remaining = [uid for uid, _ in scored[elite_k:]]
                if remaining:
                    n_extra = batch_size - len(selected_uids)
                    selected_uids.extend(np.random.choice(remaining, size=min(n_extra, len(remaining)), replace=False).tolist())
        elif strategy == "recent":
            # Use meta timestamp if available, else reverse order of load
            # For now, just reverse order
            selected_uids = uids[-batch_size:] if len(uids) >= batch_size else uids
        else:  # "weighted"
            # Mix: 40% elite, 30% recent, 30% random
            n_elite = int(batch_size * 0.4)
            n_recent = int(batch_size * 0.3)
            n_random = batch_size - n_elite - n_recent

            selected_uids = []
            # Elite
            if n_elite > 0 and len(uids) > 0:
                scored = [(uid, self._stats[uid]["mean_return"]) for uid in uids]
                scored.sort(key=lambda x: x[1], reverse=True)
                elite_k = min(n_elite, len(uids) // 4)
                selected_uids.extend([uid for uid, _ in scored[:elite_k]])

            # Recent
            if n_recent > 0:
                recent_pool = [uid for uid in uids if uid not in selected_uids]
                n_recent_actual = min(n_recent, len(recent_pool))
                selected_uids.extend(recent_pool[-n_recent_actual:])

            # Random
            if n_random > 0:
                random_pool = [uid for uid in uids if uid not in selected_uids]
                if random_pool:
                    n_random_actual = min(n_random, len(random_pool))
                    selected_uids.extend(np.random.choice(random_pool, size=n_random_actual, replace=False).tolist())

            # Pad if needed
            if len(selected_uids) < batch_size:
                remaining = [uid for uid in uids if uid not in selected_uids]
                if remaining:
                    n_extra = batch_size - len(selected_uids)
                    selected_uids.extend(np.random.choice(remaining, size=min(n_extra, len(remaining)), replace=False).tolist())

        return [self._index[uid]["task"] for uid in selected_uids]

    def update_stats(self, task_uid: str, episode_return: float, success: bool, extra: Optional[Dict[str, Any]] = None):
        """
        Update performance statistics for a task.

        Args:
            task_uid: Task UID (from add() return value or _task_uid())
            episode_return: Total episode reward/return
            success: Boolean success flag
            extra: Optional extra metrics to store
        """
        if task_uid not in self._stats:
            self._stats[task_uid] = {
                "returns": [],
                "successes": [],
                "count": 0,
                "mean_return": 0.0,
                "mean_success": 0.0,
            }

        stats = self._stats[task_uid]
        stats["returns"].append(float(episode_return))
        stats["successes"].append(1.0 if success else 0.0)
        stats["count"] += 1

        # Update running means (keep last 100 for memory efficiency)
        max_history = 100
        if len(stats["returns"]) > max_history:
            stats["returns"] = stats["returns"][-max_history:]
            stats["successes"] = stats["successes"][-max_history:]

        stats["mean_return"] = np.mean(stats["returns"])
        stats["mean_success"] = np.mean(stats["successes"])

    def prune(self, max_size: int, keep_elite_k: int = 100):
        """
        Prune pool to max_size, keeping top keep_elite_k performers.
        WARNING: This rewrites the JSONL file. Use with caution.

        Args:
            max_size: Maximum pool size after pruning
            keep_elite_k: Number of elite tasks to always keep
        """
        if self.read_only:
            raise RuntimeError("Cannot prune read-only pool")

        if len(self._index) <= max_size:
            return

        # Score all tasks
        scored = []
        for uid in self._index.keys():
            score = self._stats[uid]["mean_return"]
            scored.append((uid, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep top keep_elite_k + random sample of rest to fill max_size
        keep_uids = {uid for uid, _ in scored[:keep_elite_k]}
        remaining = [uid for uid, _ in scored[keep_elite_k:]]
        if remaining and len(keep_uids) < max_size:
            n_extra = max_size - len(keep_uids)
            keep_uids.update(np.random.choice(remaining, size=min(n_extra, len(remaining)), replace=False).tolist())

        # Rewrite JSONL
        self.jsonl_path.unlink(missing_ok=True)
        self._index = {uid: self._index[uid] for uid in keep_uids if uid in self._index}
        for uid in keep_uids:
            if uid in self._index:
                entry = {"task": self._index[uid]["task"], "meta": self._index[uid]["meta"]}
                with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, separators=(',', ':')) + '\n')

    def size(self) -> int:
        """Return current pool size."""
        return len(self._index)
