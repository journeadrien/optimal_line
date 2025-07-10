from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Dict, List
from .models import Operation, Employee, validate_consistency




# ---------- Loaders ---------- #

def load_operations(path: Path | None = None) -> Dict[str, Operation]:
    path = path / "operations.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    ops: Dict[str, Operation] = {}
    for _, row in df.iterrows():
        ops[row.operation_id] = Operation(
            operation_id=row.operation_id,
            operators_required=int(row.operators_required),
            description=row.description or None,
        )
    return ops


def _parse_bool(x: str | float) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x) or x == "":
        return False
    return str(x).strip().lower() in {"1", "true", "yes"}


def load_employees(path: Path | None = None) -> List[Employee]:
    path = path / "employees.csv"
    df = pd.read_csv(path, dtype=str).fillna("")
    emps: List[Employee] = []
    for _, row in df.iterrows():
        emps.append(
            Employee(
                employee_id=row.employee_id,
                skills=[s for s in row.skills.replace(";", " ").split() if s],
                age=float(row.age) if row.age else 40.0,
                is_female=_parse_bool(row.is_female),
                stress_level=float(row.stress_level) if row.stress_level else 0.3,
                chronic_condition=_parse_bool(row.chronic_condition),
            )
        )
    return emps


# ---------- Convenience loader with validation ---------- #

def load_all(
    operation_line_path: Path
) -> tuple[dict[str, Operation], list[Employee]]:
    ops = load_operations(operation_line_path)
    emps = load_employees(operation_line_path)
    validate_consistency(ops, emps)
    return ops, emps
    