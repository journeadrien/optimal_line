from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Iterable
import numpy as np
import math

# ---------- Dataclasses ---------- #

@dataclass
class Operation:
    operation_id: str
    operators_required: int
    description: str | None = None

    # basic validation in __post_init__
    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("operation_id empty")
        if self.operators_required < 1:
            raise ValueError(f"{self.operation_id}: operators_required must be >=1")


# ---------------------------------------------------
# ONLY the Employee dataclass changed; the Operation
# one and the functions we wrote earlier stay as-is.
# ---------------------------------------------------

@dataclass
class Employee:
    employee_id: str
    skills: List[str]
    age: float = 40.0
    is_female: bool = False
    stress_level: float = 0.3
    chronic_condition: bool = False
    frailty: float = 1.0            # Γ(1/σ², σ²)  drawn later

    # ----- epidemiological helpers ---------------------------------
    def covariate_vector(self) -> np.ndarray:
        age_centered = (self.age - 40.0) / 10.0
        return np.array([
            age_centered,
            float(self.is_female),
            self.stress_level,
            float(self.chronic_condition)
        ])

    def __post_init__(self) -> None:
        if not self.employee_id:
            raise ValueError("employee_id empty")
        if not self.skills:
            raise ValueError(f"{self.employee_id}: needs at least one skill")
        if not (0.0 <= self.stress_level <= 1.0):
            raise ValueError(f"{self.employee_id}: stress_level must be 0-1")


# ---------- Cross-file validation ---------- #

def validate_consistency(
    operations: Dict[str, Operation],
    employees: Iterable[Employee],
) -> None:
    unknown_ops: set[str] = set()
    for emp in employees:
        for op in emp.skills:
            if op not in operations:
                unknown_ops.add(op)
    if unknown_ops:
        raise ValueError(
            f"Employees reference unknown operations: {', '.join(sorted(unknown_ops))}"
        )