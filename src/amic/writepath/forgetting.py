"""ForgettingManager (L7, brief section 7): decay, TTL, compaction and per-learner erasure.

Runs as a scheduled job (`python -m amic.jobs forget`, part of `nightly`), never on the response
path. Per learner:
- M2 decay: a current fact's effective confidence is confidence * 0.5^(age / half-life), age
  counted from its last evidence (last_seen, else valid_from). Below `fact_min_confidence` the
  fact is CLOSED (valid_to = now, closed_by = "decay"), not deleted: history stays auditable.
- M2 TTL: closed facts whose valid_to is older than `closed_fact_ttl_days` are hard-deleted.
- M1 TTL: episodes older than `episode_ttl_days` are deleted (off by default).
- M1 compaction: near-duplicate turn records (cosine >= `compact_sim`) collapse to the newest;
  explicit /remember notes are never compacted.
- M3 decay: per-concept mastery drifts back to the prior (forgetting curve).
Any change bumps the learner's memory version (M4 sketch rebuilt, cached decisions dropped).
Erasure removes M1, M2, M3, M0/session state and, through the version, M4 and the cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from amic.contracts.config import ForgettingConfig

DAY = 86400.0


@dataclass
class ForgetCounts:
    facts_closed: int = 0
    facts_deleted: int = 0
    episodes_expired: int = 0
    episodes_compacted: int = 0
    profiles_decayed: int = 0
    tenants: int = 0
    per_tenant: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, tenant: str, **kv: int) -> None:
        row = self.per_tenant.setdefault(tenant, {})
        for k, v in kv.items():
            setattr(self, k, getattr(self, k) + v)
            row[k] = row.get(k, 0) + v


def duplicate_ids(
    ids: list[str], tss: list[float], embs: np.ndarray, sim: float, chunk: int = 2048
) -> list[str]:
    """Ids to drop so that no two kept records have cosine >= sim; the newest of a group wins."""
    if len(ids) < 2:
        return []
    order = np.argsort(-np.asarray(tss))  # newest first
    v = embs[order]
    kept: list[int] = []
    drop: list[str] = []
    for start in range(0, len(order), chunk):
        block = v[start : start + chunk]
        for j in range(len(block)):
            i = start + j
            if kept and float((v[kept] @ block[j]).max()) >= sim:
                drop.append(ids[order[i]])
            else:
                kept.append(i)
    return drop


class ForgettingManager:
    def __init__(
        self,
        cfg: ForgettingConfig,
        episodic: Any,
        semantic: Any = None,
        profiles: Any = None,
        sessions: Any = None,
    ) -> None:
        self.cfg = cfg
        self.episodic = episodic
        self.semantic = semantic
        self.profiles = profiles
        self.sessions = sessions

    # ------------------------------------------------------------------ per learner
    def run_tenant(
        self, tenant: str, now: datetime | None = None, dry_run: bool = False
    ) -> dict[str, int]:
        c = self.cfg
        now = now or datetime.now(UTC)
        t = now.timestamp()
        out = {
            "facts_closed": 0,
            "facts_deleted": 0,
            "episodes_expired": 0,
            "episodes_compacted": 0,
            "profiles_decayed": 0,
        }
        if self.semantic is not None:
            to_delete = []
            for fid, _, meta, _ in self.semantic.records(tenant):
                valid_to = float(meta.get("valid_to", 0.0))
                if valid_to == 0.0:  # current: decay
                    last = float(meta.get("last_seen") or meta.get("valid_from") or t)
                    conf = float(meta.get("confidence", 1.0))
                    eff = conf * 0.5 ** (max(0.0, t - last) / DAY / c.fact_half_life_days)
                    if eff < c.fact_min_confidence:
                        out["facts_closed"] += 1
                        if not dry_run:
                            self.semantic.patch(tenant, fid, valid_to=t, closed_by="decay")
                elif c.closed_fact_ttl_days is not None and t - valid_to > (
                    c.closed_fact_ttl_days * DAY
                ):
                    to_delete.append(fid)
            out["facts_deleted"] = len(to_delete)
            if to_delete and not dry_run:
                self.semantic.delete_ids(tenant, to_delete)
        recs = self.episodic.records(tenant)
        expired: list[str] = []
        if c.episode_ttl_days is not None:
            cutoff = t - c.episode_ttl_days * DAY
            expired = [rid for rid, _, m, _ in recs if float(m.get("ts", t)) < cutoff]
        gone = set(expired)
        live = [r for r in recs if r[0] not in gone and r[2].get("kind") != "remember"]
        dups = (
            duplicate_ids(
                [r[0] for r in live],
                [float(r[2].get("ts", 0.0)) for r in live],
                np.stack([r[3] for r in live]),
                c.compact_sim,
            )
            if len(live) > 1
            else []
        )
        out["episodes_expired"], out["episodes_compacted"] = len(expired), len(dups)
        if not dry_run and (expired or dups):
            self.episodic.delete_ids(tenant, expired + dups)
        prof = self.profiles.get(tenant) if self.profiles is not None else None
        if prof is not None and prof.decay(now, c.mastery_half_life_days):
            out["profiles_decayed"] = 1
            if not dry_run:
                self.profiles.put(prof)  # type: ignore[union-attr]
        if not dry_run and any(out.values()) and self.sessions is not None:
            self.sessions.bump_mem_version(tenant)
        return out

    def tenants(self) -> list[str]:
        ids: set[str] = set(self.episodic.tenants())
        if self.semantic is not None:
            ids |= set(self.semantic.tenants())
        if self.profiles is not None:
            ids |= set(self.profiles.tenants())
        return sorted(ids)

    def run(
        self,
        now: datetime | None = None,
        tenants: list[str] | None = None,
        dry_run: bool = False,
    ) -> ForgetCounts:
        counts = ForgetCounts()
        for tenant in tenants if tenants is not None else self.tenants():
            counts.add(tenant, **self.run_tenant(tenant, now, dry_run))
            counts.tenants += 1
        return counts

    # ------------------------------------------------------------------ erasure
    def erase(self, tenant: str) -> dict[str, int]:
        """Right to erasure: every tier of one learner, synchronously (plan R6)."""
        out = {
            "deleted_records": self.episodic.delete_tenant(tenant),
            "deleted_facts": self.semantic.delete_tenant(tenant) if self.semantic else 0,
            "deleted_profile": self.profiles.delete(tenant) if self.profiles else 0,
        }
        if self.sessions is not None:  # M0 + session meta; a new memory version drops M4/cache
            self.sessions.erase_tenant(tenant)
        return out
