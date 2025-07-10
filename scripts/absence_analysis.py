#!/usr/bin/env python3
"""
Reproduce the “duration analysis” plots, but driven by our
src/absence.py engine (no legacy functions required).

• Loads employees from CSV via src/io.load_all
• Simulates daily sickness events with src.absence.simulate_employee
• Builds the same KPIs (short / medium / long spells, etc.)
• Generates the 6-panel matplotlib figure

Run:
    python scripts/absence_analysis.py --days 250 --seed 123 --pop 1000
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # optional but improves default palettes

from optimal_line.io import load_all
from optimal_line.models import Employee
from optimal_line.absence import (
    simulate_employee,
    generate_frailty,
    AbsenceType,
    SicknessEvent,      # reuse dataclass, handy for typing
)

DATA_DIR = Path("").resolve() / "data"

sns.set_style("whitegrid")  # prettier plots

# ---------------------------------------------------------------------------
# 1. Helper functions
# ---------------------------------------------------------------------------
def duration_category(d: int) -> str:
    if d <= 3:
        return "SHORT"
    if d <= 14:
        return "MEDIUM"
    return "LONG"


def simulate_population(
    employees: List[Employee],
    days: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, List[SicknessEvent]]:
    """
    Return (per-employee KPI dataframe, full list of events)
    """
    all_events: List[SicknessEvent] = []
    rows: List[Dict] = []

    for emp in employees:
        # ensure each employee has a frailty draw
        if emp.frailty == 1.0:
            emp.frailty = generate_frailty(rng)

        events = simulate_employee(emp, days, rng)
        all_events.extend(events)

        # spell metrics
        cats = [duration_category(ev.duration) for ev in events]
        short = cats.count("SHORT")
        med = cats.count("MEDIUM")
        long = cats.count("LONG")
        total_days = sum(ev.duration for ev in events)

        rows.append(
            dict(
                employee_id=emp.employee_id,
                frailty=emp.frailty,
                age=emp.age,
                is_female=emp.is_female,
                chronic_condition=emp.chronic_condition,
                total_spells=len(events),
                short_spells=short,
                medium_spells=med,
                long_spells=long,
                total_days_absent=total_days,
                absence_rate=total_days / days * 100.0,
                avg_duration=total_days / len(events) if events else 0.0,
                events=events,  # keep raw list for later
            )
        )

    return pd.DataFrame(rows), all_events


def duration_stats_by_type(events: List[SicknessEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "duration": [ev.duration for ev in events],
            "type": [ev.absence_type.value for ev in events],
        }
    )
    return (
        df.groupby("type")["duration"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            min="min",
            max="max",
            p25=lambda s: s.quantile(0.25),
            p50=lambda s: s.quantile(0.50),
            p75=lambda s: s.quantile(0.75),
        )
        .round(2)
    )


# ---------------------------------------------------------------------------
# 2. Plotting
# ---------------------------------------------------------------------------
def plot_duration_analysis(results: pd.DataFrame, stats_df: pd.DataFrame,
    outfile: str = "benchmark_results.png") -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # --- 1. Histogram by SHORT / MEDIUM / LONG ----------------------------
    ax = axes[0, 0]
    durations = []
    cats = []
    for evs in results["events"]:
        for ev in evs:
            durations.append(ev.duration)
            cats.append(duration_category(ev.duration))

    if durations:
        tmp = pd.DataFrame({"dur": durations, "cat": cats})
        for cat, color in zip(["SHORT", "MEDIUM", "LONG"], ["#77dd77", "#ffb347", "#ff6961"]):
            sns.histplot(
                tmp[tmp["cat"] == cat]["dur"],
                ax=ax,
                bins=20,
                kde=False,
                color=color,
                label=cat,
                alpha=0.6,
            )
        ax.set_title("Duration distribution by category")
        ax.set_xlabel("Duration (days)")
        ax.set_ylabel("Frequency")
        ax.legend()

    # --- 2. Mean duration per absence type --------------------------------
    ax = axes[0, 1]
    if not stats_df.empty:
        stats_df["mean"].plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title("Mean duration by absence type")
        ax.set_xlabel("")
        ax.set_ylabel("Mean days")
        ax.tick_params(axis="x", rotation=45)

    # --- 3. Pie chart of spell categories ---------------------------------
    ax = axes[0, 2]
    spell_totals = results[["short_spells", "medium_spells", "long_spells"]].sum()
    spell_totals.index = ["Short", "Medium", "Long"]
    spell_totals.plot.pie(
        ax=ax,
        autopct="%1.1f%%",
        colors=["#77dd77", "#ffb347", "#ff6961"],
        ylabel="",
    )
    ax.set_title("Spell duration mix")

    # --- 4. Frailty vs average duration -----------------------------------
    ax = axes[1, 0]
    mask = results["total_spells"] > 0
    ax.scatter(
        results.loc[mask, "frailty"],
        results.loc[mask, "avg_duration"],
        alpha=0.5,
        c="purple",
    )
    ax.set_xlabel("Frailty (ω)")
    ax.set_ylabel("Avg duration (days)")
    ax.set_title("Frailty vs average spell duration")

    # --- 5. Avg duration by age group -------------------------------------
    ax = axes[1, 1]
    age_groups = pd.cut(results["age"], bins=[20, 30, 40, 50, 65],
                        labels=["20-30", "31-40", "41-50", "51-65"])
    results.groupby(age_groups)["avg_duration"].mean().plot.bar(
        ax=ax, color="coral"
    )
    ax.set_xlabel("Age group")
    ax.set_ylabel("Avg duration (days)")
    ax.set_title("Avg spell duration by age group")

    # --- 6. Frequency of absence types ------------------------------------
    ax = axes[1, 2]
    type_counts: Dict[str, int] = {}
    for evs in results["events"]:
        for ev in evs:
            t = ev.absence_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
    pd.Series(type_counts).sort_values(ascending=False).plot.bar(
        ax=ax, color="darkgreen"
    )
    ax.set_xlabel("Absence type")
    ax.set_ylabel("Occurrences")
    ax.set_title("Frequency of absence types")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.show()


# ---------------------------------------------------------------------------
# 3. Main CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=250, help="Simulation horizon")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--pop",
        type=int,
        default=None,
        help="Limit population size (default = all employees in CSV)",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="images/random.png",
        help="Output filename for the generated graph",
    )
    args = parser.parse_args()

    # Load employees from CSV files prepared earlier
    _, employees = load_all(DATA_DIR / "random")

    # Optionally subsample population (useful for speed)
    if args.pop:
        employees = employees[: args.pop]

    rng = np.random.default_rng(args.seed)

    results_df, events = simulate_population(employees, args.days, rng)
    stats_df = duration_stats_by_type(events)

    # High-level summary
    print(stats_df)
    print("\nMean absence rate:", results_df["absence_rate"].mean().round(2), "%")

    plot_duration_analysis(results_df, stats_df, args.outfile)


if __name__ == "__main__":
    main()