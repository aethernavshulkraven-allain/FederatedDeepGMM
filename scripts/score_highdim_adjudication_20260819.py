#!/usr/bin/env python3
"""Mechanical scorer for the 500-round, 3-seed Psi-vs-MSE adjudication and
confirmation stages, implementing
PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md's decision tree (section 0)
exactly:

1. A candidate is eligible only if all 3 seeds have complete artifacts,
   finite required metrics, and diverged=False. No two-seed median, no
   imputation, no rerunning an identical deterministic seed.
2. 0 eligible candidates -> cell requires retuning.
3. 1 eligible candidate -> promoted directly, no ranking needed.
4. >=2 eligible candidates -> rank by M_c = median across seeds of
   S_{c,s} = mean(Psi_451:500); compare the top-ranked candidate against
   every other eligible candidate with the frozen pairwise practical-tie
   rule; form the tie set (top plus everything that ties with top); if the
   tie set has more than one member, promote whichever has the lowest
   median last-50-round mean validation MSE within that set.

Critic collapse is diagnostic-only and never enters this scorer's
decision -- by design, this module has no notion of critic output at all.

This module is imported by tests/test_adjudication_scorer_20260819.py,
which exercises it against the six required synthetic scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


@dataclass(frozen=True)
class SeedResult:
    seed: int
    psi_last50_mean: float | None
    mse_last50_mean: float | None
    diverged: bool
    artifacts_complete: bool
    finite: bool

    @property
    def ok(self) -> bool:
        return (
            self.artifacts_complete
            and self.finite
            and not self.diverged
            and self.psi_last50_mean is not None
            and self.mse_last50_mean is not None
        )


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    label: str
    seeds: tuple[SeedResult, ...]

    @property
    def eligible(self) -> bool:
        return len(self.seeds) == 3 and all(s.ok for s in self.seeds)

    @property
    def psi_scores(self) -> list[float]:
        return [s.psi_last50_mean for s in self.seeds]

    @property
    def mse_scores(self) -> list[float]:
        return [s.mse_last50_mean for s in self.seeds]

    @property
    def median_psi(self) -> float:
        return median(self.psi_scores)

    @property
    def median_mse(self) -> float:
        return median(self.mse_scores)

    @property
    def psi_range(self) -> float:
        return max(self.psi_scores) - min(self.psi_scores)


def practical_tie(a: Candidate, b: Candidate) -> bool:
    """Frozen pairwise Psi tie rule from PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md."""
    diffs = [sa - sb for sa, sb in zip(a.psi_scores, b.psi_scores)]
    signs = {(1 if d > 0 else (-1 if d < 0 else 0)) for d in diffs}
    signs_differ = len(signs) > 1
    gap_small = abs(a.median_psi - b.median_psi) <= max(a.psi_range, b.psi_range)
    return signs_differ or gap_small


@dataclass
class CellOutcome:
    outcome: str  # "promoted" | "tie_resolved_by_mse" | "retune_required"
    winner: Candidate | None
    eligible: list[Candidate]
    excluded: list[Candidate]
    tie_set: list[Candidate] = field(default_factory=list)
    detail: str = ""


def score_cell(candidates: list[Candidate]) -> CellOutcome:
    eligible = [c for c in candidates if c.eligible]
    excluded = [c for c in candidates if not c.eligible]

    if not eligible:
        return CellOutcome(
            outcome="retune_required", winner=None, eligible=eligible, excluded=excluded,
            detail="no eligible candidate: every candidate failed the three-seed stability rule",
        )

    if len(eligible) == 1:
        return CellOutcome(
            outcome="promoted", winner=eligible[0], eligible=eligible, excluded=excluded,
            tie_set=list(eligible), detail="only one eligible candidate, no ranking needed",
        )

    ranked = sorted(eligible, key=lambda c: -c.median_psi)
    top = ranked[0]
    tie_set = [top] + [c for c in ranked[1:] if practical_tie(top, c)]

    if len(tie_set) == 1:
        return CellOutcome(
            outcome="promoted", winner=top, eligible=eligible, excluded=excluded,
            tie_set=tie_set, detail="top Psi candidate, no practical tie with any other eligible candidate",
        )

    winner = min(tie_set, key=lambda c: c.median_mse)
    return CellOutcome(
        outcome="tie_resolved_by_mse", winner=winner, eligible=eligible, excluded=excluded,
        tie_set=tie_set,
        detail=f"practical tie among {[c.candidate_id for c in tie_set]}, resolved by lowest median MSE",
    )
