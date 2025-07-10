#!/usr/bin/env python3
"""
Estimate how often a production line can run given sickness absences
and a skills matrix.

Algorithm
---------
1. Read operations & employees (CSV) from a data directory.
2. Simulate daily availability for each employee with src.absence.
3. For every day:
   • Build a bipartite graph (employees ↔ operation slots).
   • Run a maximum bipartite matching.
   • If every slot is filled, the line runs.
4. Output the % of operational days and a few extra KPIs.

Usage
-----
python scripts/line_robustness.py --data_dir data/random --days 250 --seed 123
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np

from optimal_line.io import load_all
from optimal_line.absence import availability_calendar, generate_frailty
from optimal_line.models import Employee


# ---------------------------------------------------------------------
# 1.  Bipartite matching helper (classic DFS augmenting-path algorithm)
# ---------------------------------------------------------------------
def line_runs_today(
    available_emp_ids: List[int],
    employees: List[Employee],
    op_slots: List[tuple[str, int]],
    emp_skill_map: Dict[str, set[str]],
) -> bool:
    """
    Parameters
    ----------
    available_emp_ids : list[int]
        Indexes (into `employees`) of employees present today.
    op_slots : list[(op_id, slot_idx)]
        Each operation appears `operators_required` times.
    emp_skill_map : employee_id -> set(op_id)
    """

    # Build adjacency lists: emp_index -> [slot_index, ...]
    adj: List[List[int]] = [[] for _ in available_emp_ids]
    for ei, emp_idx in enumerate(available_emp_ids):
        emp_id = employees[emp_idx].employee_id
        skills = emp_skill_map[emp_id]
        for si, (op_id, _) in enumerate(op_slots):
            if op_id in skills:
                adj[ei].append(si)

    # ── maximum bipartite matching ------------------------------------
    match_r = [-1] * len(op_slots)

    def bpm(u: int, seen: List[bool]) -> bool:
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_r[v] == -1 or bpm(match_r[v], seen):
                match_r[v] = u
                return True
        return False

    match_count = 0
    for u in range(len(adj)):
        if bpm(u, [False] * len(op_slots)):
            match_count += 1
        else:
            # this employee couldn't augment matching – continue
            pass

    return match_count == len(op_slots)


# ---------------------------------------------------------------------
# 2.  Simulator
# ---------------------------------------------------------------------
def simulate_line(
    data_dir: Path,
    days: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    # Load data --------------------------------------------------------
    ops, emps = load_all(data_dir)

    # Pre-compute operation slots
    op_slots: List[tuple[str, int]] = []
    for op in ops.values():
        op_slots.extend([(op.operation_id, i) for i in range(op.operators_required)])
    total_slots = len(op_slots)

    # Quick feasibility check
    min_heads = sum(o.operators_required for o in ops.values())
    if len(emps) < min_heads:
        raise RuntimeError(
            f"Roster too small: {len(emps)} employees for {min_heads} required slots"
        )

    # Skill lookup
    emp_skill_map = {e.employee_id: set(e.skills) for e in emps}

    # Simulate availability calendars
    calendars = []
    for emp in emps:
        if emp.frailty == 1.0:
            emp.frailty = generate_frailty(rng)
        calendars.append(availability_calendar(emp, days, rng).calendar)

    calendars = np.vstack(calendars)  # shape (n_emp, days)

    # ----- Daily loop -------------------------------------------------
    run_days = 0
    for d in range(days):
        present_idxs = np.where(calendars[:, d])[0].tolist()

        # early exit: not even enough heads
        if len(present_idxs) < total_slots:
            continue

        if line_runs_today(present_idxs, emps, op_slots, emp_skill_map):
            run_days += 1

    pct = run_days / days * 100
    print(f"✅  Line operational {run_days}/{days} days  ({pct:.1f}%)")


# ---------------------------------------------------------------------
# 3.  CLI
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/random"),
        help="Folder containing operations.csv & employees.csv",
    )
    ap.add_argument("--days", type=int, default=250)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    simulate_line(args.data_dir, args.days, args.seed)


if __name__ == "__main__":
    main()