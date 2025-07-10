#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from optimal_line.io import load_all
from optimal_line.absence import availability_calendar

DAYS = 250

def main():
    rng = np.random.default_rng(seed=123)
    ops, emps = load_all("random")         # loads operations & employees CSVs
    calendars = [availability_calendar(e, DAYS, rng) for e in emps]

    # Simple KPI: % utilisation per employee
    for cal in calendars:
        utilisation = cal.calendar.mean() * 100
        print(f"{cal.employee_id:5s}  present {utilisation:5.1f}% of days")

if __name__ == "__main__":
    main()