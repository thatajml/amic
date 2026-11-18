"""Tier 0: deterministic policy engine (~µs). Resolves the obvious cases; otherwise abstains.

Rule order (deviation from brief 6.2, flagged): explicit commands are checked FIRST, so a
`/remember` from a brand-new learner is still honoured instead of being swallowed by R1.

    R4  /remember, /forget                            -> WRITE
    R1  learner has no long-term memory               -> SKIP   (cold-start guarantee)
    R2  follow-up to today's conversation, no
        past-time reference                           -> SKIP   (M0 answers it)
    R3  explicit reference to past sessions           -> DEEP
    --  otherwise                                     -> ABSTAIN (None)
"""

from __future__ import annotations

from dataclasses import dataclass

from amic.contracts.models import Route
from amic.control.signals.bus import Signals


@dataclass(frozen=True)
class Tier0Result:
    route: Route | None  # None = abstain
    rule: str


class Tier0:
    def decide(self, sig: Signals) -> Tier0Result:
        f = sig.features
        if sig.is_command:
            return Tier0Result(Route.WRITE, "R4: explicit memory command")
        if f["s3_has_memory"] == 0.0:  # set from sketch existence, even when S3 is ablated
            return Tier0Result(Route.SKIP, "R1: no long-term memory for this learner")
        if f["s2_m0_turns"] > 0 and f["s1_followup"] > 0 and f["s1_explicit_ref"] == 0:
            return Tier0Result(Route.SKIP, "R2: follow-up answerable from the current session")
        if f["s1_explicit_ref"] > 0:
            return Tier0Result(Route.DEEP, "R3: explicit reference to past sessions")
        return Tier0Result(None, "abstain")
