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
which exercises it (as a pure library, synthetic Candidate/SeedResult
objects, no disk I/O) against the six required synthetic scenarios.

It is ALSO an end-to-end CLI (see `main()` / `load_seed_result` /
`build_cell_candidates` below) that reads the real completed/reused runs
for a given adjudication manifest (results/highdim_psi_adjudication_20260819_v2/
for new runs, results/highdim_deterministic_finals_20260813/ for
reused-from-finals candidates) and writes final per-cell adjudication
outcomes -- this wiring is frozen now, before any real adjudication
results exist, per the same before-seeing-the-data discipline as every
other rule in this campaign.

Usage:
  python scripts/score_highdim_adjudication_20260819.py --cells x
  python scripts/score_highdim_adjudication_20260819.py --cells signal
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_manifest import METHOD_TO_OPTIMIZER  # noqa: E402
from validate_smoke_run import _load_json  # noqa: E402

CAMPAIGN_V2_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/psi_adjudication_20260819_v2"
FINALS_DIR = REPO_ROOT / "experiments/highdim_coauthor_protocol_v1/deterministic_finals_20260813"
NEW_RUN_RESULTS_ROOT = REPO_ROOT / "results/highdim_psi_adjudication_20260819_v2"
REQUIRED_ARTIFACTS = (
    "effective_config.json", "metrics.json", "mse_by_round.csv",
    "predictions.npz", "checkpoints/best_validation.pt", "checkpoints/final.pt",
)
LAST_N_ROUNDS = 50


@dataclass(frozen=True)
class SeedResult:
    seed: int
    psi_last50_mean: float | None
    mse_last50_mean: float | None
    diverged: bool
    artifacts_complete: bool
    finite: bool
    best_round_psi: float | None = None  # diagnostic only, never used for scoring
    run_dir: str = ""
    note: str = ""

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


# --------------------------------------------------------------------------
# Real-data wiring: reads actual completed/reused runs off disk and scores
# them. Everything above this line is a pure library with no disk I/O.
# --------------------------------------------------------------------------

def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_seed_result(run_dir: Path, seed: int) -> SeedResult:
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).exists()]
    if missing:
        return SeedResult(
            seed=seed, psi_last50_mean=None, mse_last50_mean=None,
            diverged=False, artifacts_complete=False, finite=False,
            run_dir=str(run_dir), note=f"missing artifacts: {missing}",
        )
    try:
        metrics = _load_json(run_dir / "metrics.json")
        diverged = bool(metrics.get("diverged", False))
        with (run_dir / "mse_by_round.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return SeedResult(
                seed=seed, psi_last50_mean=None, mse_last50_mean=None,
                diverged=diverged, artifacts_complete=True, finite=False,
                run_dir=str(run_dir), note="mse_by_round.csv has no rows",
            )
        tail = rows[-LAST_N_ROUNDS:]
        psi_vals = [float(r["gmm_eval"]) for r in tail]
        mse_vals = [float(r["val_mse"]) for r in tail]
        all_psi_vals = [float(r["gmm_eval"]) for r in rows]
        row_flags_ok = all(str(r.get("finite")) == "True" and str(r.get("diverged")) == "False" for r in tail)
        finite = (
            row_flags_ok
            and all(v == v and abs(v) != float("inf") for v in psi_vals + mse_vals)  # NaN/inf guard
        )
        return SeedResult(
            seed=seed,
            psi_last50_mean=(sum(psi_vals) / len(psi_vals)) if finite else None,
            mse_last50_mean=(sum(mse_vals) / len(mse_vals)) if finite else None,
            diverged=diverged, artifacts_complete=True, finite=finite,
            best_round_psi=max(all_psi_vals) if all_psi_vals else None,
            run_dir=str(run_dir),
            note="" if finite else "non-finite or diverged value in last-50-round window",
        )
    except Exception as exc:  # noqa: BLE001
        return SeedResult(
            seed=seed, psi_last50_mean=None, mse_last50_mean=None,
            diverged=False, artifacts_complete=True, finite=False,
            run_dir=str(run_dir), note=f"{type(exc).__name__}: {exc}",
        )


def resolve_new_run_dir(dataset: str, method: str, seed: int, lr: float, cm: float) -> Path:
    run_id = f"det_adjudicate_{dataset}_{method}_seed{seed}_alpha0p5_lr{token(lr)}_cm{token(cm)}"
    return NEW_RUN_RESULTS_ROOT / dataset / method / f"seed_{seed}" / run_id


def resolve_reused_run_dir(finals_index: dict, dataset: str, method: str, seed: int, lr: float, cm: float) -> Path | None:
    key = (dataset, method, round(lr, 6), round(cm, 6), seed, 0.5)
    row = finals_index.get(key)
    if row is None:
        return None
    d = Path(row["final_result_dir"])
    return d if d.is_absolute() else REPO_ROOT / d


def load_finals_index() -> dict:
    with (FINALS_DIR / "finals_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (r["dataset"], r["method"], round(float(r["learning_rate"]), 6),
         round(float(r["critic_multiplier"]), 6), int(r["seed"]), round(float(r["alpha"]), 6)): r
        for r in rows
    }


def build_cell_candidates(cell_plan: dict, finals_index: dict, seeds=(0, 1, 2)) -> list[Candidate]:
    dataset, method = cell_plan["dataset"], cell_plan["method"]
    candidates = []
    for cand in cell_plan["candidates"]:
        lr, cm = cand["lr"], cand["cm"]
        candidate_id = f"{dataset}/{method}/lr{lr:g}_cm{cm:g}"
        label = ",".join(cand["labels"])
        seed_results = []
        for seed in seeds:
            if cand["reused_from_finals"]:
                run_dir = resolve_reused_run_dir(finals_index, dataset, method, seed, lr, cm)
                if run_dir is None:
                    seed_results.append(SeedResult(
                        seed=seed, psi_last50_mean=None, mse_last50_mean=None,
                        diverged=False, artifacts_complete=False, finite=False,
                        note="reused_from_finals=True but no matching finals_manifest.csv row found",
                    ))
                    continue
            else:
                run_dir = resolve_new_run_dir(dataset, method, seed, lr, cm)
            seed_results.append(load_seed_result(run_dir, seed))
        candidates.append(Candidate(candidate_id=candidate_id, label=label, seeds=tuple(seed_results)))
    return candidates


def seed_result_to_dict(s: SeedResult) -> dict:
    return asdict(s)


def candidate_to_dict(c: Candidate) -> dict:
    return {
        "candidate_id": c.candidate_id, "label": c.label, "eligible": c.eligible,
        "median_psi": c.median_psi if c.eligible else None,
        "median_mse": c.median_mse if c.eligible else None,
        "seeds": [seed_result_to_dict(s) for s in c.seeds],
    }


def score_manifest(cells: str) -> dict:
    with (CAMPAIGN_V2_DIR / f"adjudication_{cells}_summary.json").open() as handle:
        summary = json.load(handle)
    finals_index = load_finals_index()

    cell_results = {}
    for cell_plan in summary["plan"]:
        dataset, method = cell_plan["dataset"], cell_plan["method"]
        candidates = build_cell_candidates(cell_plan, finals_index)
        outcome = score_cell(candidates)
        cell_results[f"{dataset}/{method}"] = {
            "dataset": dataset, "method": method,
            "outcome": outcome.outcome,
            "winner": candidate_to_dict(outcome.winner) if outcome.winner else None,
            "eligible_candidates": [c.candidate_id for c in outcome.eligible],
            "excluded_candidates": [c.candidate_id for c in outcome.excluded],
            "tie_set": [c.candidate_id for c in outcome.tie_set],
            "detail": outcome.detail,
            "all_candidates": [candidate_to_dict(c) for c in candidates],
        }
    return cell_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", choices=["x", "signal"], required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = score_manifest(args.cells)
    out_path = Path(args.out) if args.out else CAMPAIGN_V2_DIR / f"adjudication_{args.cells}_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for cell, r in sorted(results.items()):
        winner = r["winner"]["candidate_id"] if r["winner"] else None
        print(f"{cell:28s} {r['outcome']:22s} winner={winner}  {r['detail']}")
    print(f"\nWritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
