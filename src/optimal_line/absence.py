"""
Sickness-absence simulator (headless).

Key features
------------
* All parameters are defined in the CONFIG dict below – no extra file.
* Uses numpy.random.Generator for full reproducibility.
* AbsenceType enum prevents stray strings.
* availability_calendar(emp, days, rng)  → Boolean vector ready for the
  production-line robustness engine.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np
from .models import Employee   # same Employee dataclass we edited earlier


# =============================================================================
# 0. PARAMETER BLOCK  ❱❱  edit here to calibrate the model
# =============================================================================
CONFIG: Dict = {
    "baseline_hazard": 0.008,          # daily baseline illness onset rate
    "frailty_variance": 0.30,          # Γ variance for ω_i

    # β coefficients for the proportional-hazard covariate vector
    #   (age centred / female / stress / chronic)
    "covariate_coefficients": [0.02, 0.15, 0.30, 0.50],

    # Absence-type catalogue
    "absence_types": {
        "minor_illness": {
            "probability": 0.272,
            "hazard_modifier": 1.2,
            "duration": {"mean": 2.5, "std": 1.5, "max": 7}
        },
        "musculoskeletal": {
            "probability": 0.197,
            "hazard_modifier": 0.8,
            "duration": {"mean": 8, "std": 6, "max": 30},
            "age_multiplier_per_year": 0.02          # ↑ probability with age
        },
        "mental_health": {
            "probability": 0.124,
            "hazard_modifier": 0.6,
            "duration": {"mean": 14, "std": 10, "max": 60},
            "stress_multiplier": 1.0                 # ↑ probability with stress
        },
        "gastrointestinal": {
            "probability": 0.15,
            "hazard_modifier": 1.1,
            "duration": {"mean": 3, "std": 2, "max": 10}
        },
        "injury": {
            "probability": 0.08,
            "hazard_modifier": 0.4,
            "duration": {"mean": 12, "std": 8, "max": 45},
            "age_reduction_per_year": 0.01           # ↓ probability with age
        },
        "headache_migraine": {
            "probability": 0.06,
            "hazard_modifier": 0.9,
            "duration": {"mean": 1.5, "std": 0.8, "max": 4}
        },
        "other": {
            "probability": 0.117,
            "hazard_modifier": 1.0,
            "duration": {"mean": 5, "std": 4, "max": 20}
        }
    }
}
# -----------------------------------------------------------------------------


# =============================================================================
# 1.  Helpful globals derived from CONFIG
# =============================================================================
_BETA = np.asarray(CONFIG["covariate_coefficients"], dtype=float)
_BASELINE_HAZARD = float(CONFIG["baseline_hazard"])
_FRAILTY_VAR = float(CONFIG["frailty_variance"])
_TYPE_CFG: Dict[str, dict] = CONFIG["absence_types"]


class AbsenceType(str, Enum):
    MINOR_ILLNESS = "minor_illness"
    MUSCULOSKELETAL = "musculoskeletal"
    MENTAL_HEALTH = "mental_health"
    GASTROINTESTINAL = "gastrointestinal"
    INJURY = "injury"
    HEADACHE_MIGRAINE = "headache_migraine"
    OTHER = "other"


_ALL_TYPES = [AbsenceType(t) for t in _TYPE_CFG.keys()]


# =============================================================================
# 2.  Dataclasses for simulation results
# =============================================================================
@dataclass(slots=True)
class SicknessEvent:
    start_day: int
    duration: int
    absence_type: AbsenceType

    @property
    def end_day(self) -> int:
        return self.start_day + self.duration


@dataclass(slots=True)
class Availability:
    employee_id: str
    calendar: np.ndarray   # shape = (days,), True = present


# =============================================================================
# 3.  Low-level probability helpers
# =============================================================================
def generate_frailty(rng: np.random.Generator) -> float:
    """Draw Γ(shape=1/σ², scale=σ²) frailty."""
    shape = 1.0 / _FRAILTY_VAR
    scale = _FRAILTY_VAR
    return rng.gamma(shape=shape, scale=scale)


def _unnormalised_type_probs(emp: Employee) -> List[float]:
    """Return unnormalised probabilities for each absence type."""
    probs = []
    for at in _ALL_TYPES:
        cfg = _TYPE_CFG[at.value]
        p = cfg["probability"]

        # optional probability modifiers
        if at is AbsenceType.MUSCULOSKELETAL:
            p *= 1 + cfg.get("age_multiplier_per_year", 0) * (emp.age - 40)
        elif at is AbsenceType.MENTAL_HEALTH:
            p *= 1 + cfg.get("stress_multiplier", 0) * emp.stress_level
        elif at is AbsenceType.INJURY:
            red = 1 - cfg.get("age_reduction_per_year", 0) * (emp.age - 40)
            p *= max(0.50, red)

        probs.append(p)
    s = sum(probs)
    return [p / s for p in probs]


def _draw_absence_type(emp: Employee, rng: np.random.Generator) -> AbsenceType:
    p = _unnormalised_type_probs(emp)
    idx = rng.choice(len(_ALL_TYPES), p=p)
    return _ALL_TYPES[idx]


def _hazard(emp: Employee, atype: AbsenceType) -> float:
    """Daily onset hazard for given absence type."""
    mod = _TYPE_CFG[atype.value]["hazard_modifier"]
    cov_effect = np.exp(np.dot(emp.covariate_vector(), _BETA))
    return _BASELINE_HAZARD * mod * cov_effect * emp.frailty


def _duration(emp: Employee, atype: AbsenceType, rng: np.random.Generator) -> int:
    cfg = _TYPE_CFG[atype.value]["duration"]
    mean, std, dmax = cfg["mean"], cfg["std"], cfg["max"]
    mult = emp.frailty * (1.5 if emp.chronic_condition else 1.0)

    d = rng.normal(mean * mult, std)
    d = max(1, min(d, dmax * mult))
    return int(round(d))


# =============================================================================
# 4.  Core simulation
# =============================================================================
EPS = 1e-9            # minimal positive hazard

def simulate_employee(
    emp: Employee,
    days: int,
    rng: np.random.Generator,
) -> List[SicknessEvent]:
    """
    Simulate sickness events for one employee over `days` days.
    Returns a list of SicknessEvent objects.
    """
    t = 0.0                     # current time (float, allows fractions of days)
    days_since_last = 30.0      # start with full recovery
    events: List[SicknessEvent] = []

    while t < days:
        atype = _draw_absence_type(emp, rng)

        # Daily onset hazard capped between EPS and 0.25
        haz = _hazard(emp, atype) * min(1.0, days_since_last / 14.0)
        haz = max(min(haz, 0.25), EPS)

        # Time until next onset (continuous)
        dt = rng.exponential(1.0 / haz)

        # If the next onset happens beyond the simulation horizon, stop here
        if t + dt >= days:
            break

        # Advance time and days-since-last-absence up to onset
        t += dt
        days_since_last += dt

        # Absence occurs
        start_day = int(t)
        dur = _duration(emp, atype, rng)

        events.append(
            SicknessEvent(
                start_day=start_day,
                duration=dur,
                absence_type=atype,
            )
        )

        # Jump past the spell (whole-day granularity)
        t = start_day + dur + 1
        days_since_last = 0.0        # just had an absence

    return events

# =============================================================================
# 5.  Public adapter: availability_calendar
# =============================================================================
def availability_calendar(
    emp: Employee,
    days: int,
    rng: np.random.Generator,
) -> Availability:
    """
    Simulate absences and return a Boolean vector of length `days`
    (True = employee available / at work).
    """
    # Make sure the employee has a frailty value
    if emp.frailty == 1.0:          # crude check: default value means "unset"
        emp.frailty = generate_frailty(rng)

    events = simulate_employee(emp, days, rng)
    cal = np.ones(days, dtype=bool)
    for ev in events:
        cal[ev.start_day : ev.end_day] = False
    return Availability(employee_id=emp.employee_id, calendar=cal)