"""Educator Console backend (L1, brief section 5): what the system remembers about a learner,
why the gate decided what it did, and the learner's data rights. The Streamlit page
(`streamlit run ui/educator.py`) is a thin view over this class, which is unit-tested.

Read: learners, the M3 profile (mastery, misconceptions, notes, and the exact block the tutor
sees), M2 facts with their validity intervals (current and closed), recent M1 episodes, and the
learner's recent gate decisions. Write (all bump the memory version, so the M4 sketch and
cached decisions follow): close a wrong fact, add a goal/preference note, correct a concept's
mastery, preview the forgetting job, erase the learner.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from amic.writepath.forgetting import ForgettingManager


def _iso(ts: float | None) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d %H:%M") if ts else ""


class EducatorConsole:
    def __init__(self, comps: Any) -> None:
        self.c = comps
        s = comps.settings
        self.forgetting = ForgettingManager(
            s.forgetting, comps.episodic, comps.semantic, comps.profiles, comps.sessions
        )

    # ------------------------------------------------------------------ read
    def learners(self) -> list[dict[str, Any]]:
        out = []
        for t in self.forgetting.tenants():
            prof = self.c.profiles.get(t) if self.c.profiles else None
            out.append(
                {
                    "tenant_id": t,
                    "episodes": self.c.episodic.count(t),
                    "facts": self.c.semantic.count(t) if self.c.semantic else 0,
                    "profile": prof is not None,
                    "updated": _iso(prof.updated) if prof else "",
                }
            )
        return out

    def profile(self, tenant: str) -> dict[str, Any] | None:
        prof = self.c.profiles.get(tenant) if self.c.profiles else None
        if prof is None:
            return None
        mastery = sorted(
            (
                {
                    "concept": c,
                    "mastery": round(st.m, 3),
                    "evidence": st.n,
                    "seen": st.seen,
                    "last": _iso(st.last),
                }
                for c, st in prof.mastery.items()
            ),
            key=lambda r: (r["evidence"] == 0, r["mastery"]),
        )
        return {
            "mastery": mastery,
            "misconceptions": dict(prof.misconceptions),
            "notes": {k: dict(v) for k, v in prof.notes.items()},
            "block": prof.render(self.c.settings.routes.profile_tokens),
            "updated": _iso(prof.updated),
        }

    def facts(self, tenant: str, include_closed: bool = True) -> list[dict[str, Any]]:
        if self.c.semantic is None:
            return []
        rows = []
        for fid, text, m in self.c.semantic.all(tenant):
            current = float(m.get("valid_to", 0.0)) == 0.0
            if current or include_closed:
                rows.append(
                    {
                        "id": fid,
                        "key": m.get("key", ""),
                        "kind": m.get("kind", ""),
                        "text": text,
                        "confidence": m.get("confidence"),
                        "valid_from": _iso(m.get("valid_from")),
                        "valid_to": "" if current else _iso(m.get("valid_to")),
                        "closed_by": m.get("closed_by", ""),
                        "current": current,
                    }
                )
        return sorted(rows, key=lambda r: (not r["current"], r["valid_from"]), reverse=False)

    def episodes(self, tenant: str, limit: int = 20) -> list[dict[str, Any]]:
        recs = sorted(self.c.episodic.records(tenant), key=lambda r: -float(r[2].get("ts", 0)))
        return [
            {"id": rid, "ts": _iso(m.get("ts")), "kind": m.get("kind", ""), "text": text}
            for rid, text, m, _ in recs[:limit]
        ]

    def decisions(self, tenant: str, limit: int = 50) -> list[dict[str, Any]]:
        tel = self.c.telemetry
        if not hasattr(tel, "fetch"):
            return []
        return [
            {
                "ts": t.ts.strftime("%Y-%m-%d %H:%M"),
                "arm": t.arm,
                "query": t.query,
                "route": t.route.value,
                "resolved_by": t.resolved_by.value,
                "p": None if t.p is None else round(t.p, 3),
                "escalated": t.escalated,
                "memory_tokens": t.memory_tokens,
                "latency_ms": round(t.latency.total, 1),
            }
            for t in tel.fetch(limit=limit, tenant_id=tenant)
        ]

    # ------------------------------------------------------------------ write
    def _bump(self, tenant: str) -> None:
        self.c.sessions.bump_mem_version(tenant)

    def close_fact(self, tenant: str, fact_id: str) -> None:
        """An educator marks a fact wrong/outdated: closed (kept for audit), never retrieved."""
        self.c.semantic.patch(
            tenant, fact_id, valid_to=datetime.now(UTC).timestamp(), closed_by="educator"
        )
        self._bump(tenant)

    def add_note(self, tenant: str, kind: str, text: str) -> None:
        from amic.writepath.extract import ExtractedFact

        prof = self.c.profiles.get_or_new(tenant)
        prof.observe_turn(
            "",
            datetime.now(UTC),
            [ExtractedFact(f"{kind}:educator", kind, text, 1.0)],
            max_notes=self.c.settings.profile.max_notes,
        )
        self.c.profiles.put(prof)
        self._bump(tenant)

    def set_mastery(self, tenant: str, concept: str, value: float) -> None:
        from amic.memory.stores.profile import ConceptState

        prof = self.c.profiles.get_or_new(tenant)
        st = prof.mastery.setdefault(concept, ConceptState())
        st.m, st.n, st.last = (
            float(min(1.0, max(0.0, value))),
            max(1, st.n),
            datetime.now(UTC).timestamp(),
        )
        self.c.profiles.put(prof)
        self._bump(tenant)

    def forgetting_preview(self, tenant: str) -> dict[str, int]:
        return self.forgetting.run_tenant(tenant, dry_run=True)

    def erase(self, tenant: str) -> dict[str, int]:
        return self.forgetting.erase(tenant)
