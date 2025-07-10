#!/usr/bin/env python3
"""
generate_random_data.py   –   coverage-aware roster generator
================================================================

Creates `operations.csv` and `employees.csv` under *data/<line_name>/*.

New feature
-----------
--profile {safe,normal,risky}

• safe   -> plenty of redundancy (many extra heads + multi-skill)  
• normal -> middle ground (default)  
• risky  -> tight staffing (minimal buffer + narrow skillsets)

You can still pass --buffer to override the automatic buffer that
corresponds to the chosen profile.

Examples
--------
# Very safe dataset
python scripts/generate_random_data.py --line_name demo_safe --n_ops 12 --profile safe --seed 7

# Tight roster, hard for the line to run
python scripts/generate_random_data.py --line_name demo_risky --n_ops 12 --profile risky --seed 7
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ------------------------------------------------------------------ Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


# ------------------------------------------------------------------ helpers -
def rand_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx}"


# ---------------------------------------------------------------- Operations -
def gen_operations(n_ops: int, rng: random.Random) -> pd.DataFrame:
    """
    Each operation needs 1-3 operators per shift.
    """
    rows = [
        dict(
            operation_id=rand_id("O", i),
            operators_required=rng.randint(1, 3),
            description=f"Op #{i}",
        )
        for i in range(1, n_ops + 1)
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------ Employees -----
def gen_employees(
    operations: pd.DataFrame,
    buffer: int,
    profile: str,
    rng: random.Random,
) -> pd.DataFrame:
    """
    Generates a staff roster **guaranteed** to cover every operation with at
    least `operators_required` qualified heads, plus `buffer` additional
    employees to create redundancy.
    """
    op_ids = operations.operation_id.tolist()
    op_req = dict(zip(op_ids, operations.operators_required))

    total_req = sum(op_req.values())
    n_emp = total_req + buffer

    # Skill parameters depend on chosen profile -----------------------------
    if profile == "safe":
        min_skill, max_skill = 2, min(6, len(op_ids))
    elif profile == "risky":
        min_skill, max_skill = 1, min(3, len(op_ids))
    else:  # normal
        min_skill, max_skill = 1, min(4, len(op_ids))

    # Track how many qualified workers each operation already has
    coverage: dict[str, int] = defaultdict(int)
    employees = []

    for i in range(1, n_emp + 1):
        # decide how many skills this employee will have
        k = rng.randint(min_skill, max_skill)

        # guarantee coverage for still-uncovered operations ---------------
        must_have = [op for op, req in op_req.items() if coverage[op] < req]

        skills: set[str] = set()
        if must_have:
            skills.add(rng.choice(must_have))

        # fill remaining skill slots
        while len(skills) < k:
            skills.add(rng.choice(op_ids))

        for op in skills:
            coverage[op] += 1

        employees.append(
            dict(
                employee_id=rand_id("E", i),
                skills=" ".join(sorted(skills)),
                age=rng.randint(25, 60),
                is_female=rng.choice([True, False]),
                stress_level=round(rng.uniform(0, 1), 2),
                chronic_condition=rng.choice([True, False]),
            )
        )

    # sanity check ----------------------------------------------------------
    uncovered = [op for op, c in coverage.items() if c < op_req[op]]
    if uncovered:
        raise RuntimeError(f"Coverage failed for: {uncovered}")

    return pd.DataFrame(employees)


# ------------------------------------------------------------------ main ----
def main(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    # 1. Operations ---------------------------------------------------------
    ops_df = gen_operations(args.n_ops, rng)

    # 2. Decide buffer if not overridden -----------------------------------
    if args.buffer is None:
        total_req = ops_df.operators_required.sum()
        if args.profile == "safe":
            args.buffer = max(3, int(round(total_req * 0.5)))
        elif args.profile == "risky":
            args.buffer = max(0, int(round(total_req * 0.05)))
        else:  # normal
            args.buffer = max(2, int(round(total_req * 0.25)))

    # 3. Employees ---------------------------------------------------------
    emps_df = gen_employees(ops_df, args.buffer, args.profile, rng)

    # 4. Write CSVs --------------------------------------------------------
    out_dir = DATA_ROOT / args.line_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ops_df.to_csv(out_dir / "operations.csv", index=False)
    emps_df.to_csv(out_dir / "employees.csv", index=False)

    print(
        f"✅  Generated {len(ops_df)} operations & {len(emps_df)} employees "
        f"(buffer={args.buffer}) in {out_dir.relative_to(ROOT)}"
    )


# ------------------------------------------------------------------ CLI -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--line_name", default="random")
    parser.add_argument("--n_ops", type=int, default=8, help="number of operations")
    parser.add_argument(
        "--profile",
        choices=["safe", "normal", "risky"],
        default="normal",
        help="staffing risk profile (chooses buffer & skill breadth)",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=None,
        help="override extra heads beyond minimum (negates --profile default)",
    )
    parser.add_argument("--seed", type=int, default=1)
    main(parser.parse_args())