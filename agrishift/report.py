"""Markdown export of a rotation plan (pure function, unit-tested)."""
from __future__ import annotations

import datetime as dt

from .engine import FAIL_THRESHOLD, Plan, PlannerResult
from .models import Soil


def plan_to_markdown(result: PlannerResult, plan: Plan, lat: float, lon: float, soil: Soil,
                     weights: dict, irrigation_cap_mm: float, today: dt.date | None = None) -> str:
    today = today or dt.date.today()
    cs, b = result.climate, result.baseline
    lines = [
        f"# AgriShift rotation plan - {' / '.join(plan.labels)}",
        f"*Generated {today.isoformat()} for lat {lat:.4f}, lon {lon:.4f}. "
        f"Climate record: NASA POWER (MERRA-2) {cs['first_year']}-{cs['last_year']}.*", "",
        "## Plan (repeats every %d years)" % round(plan.cycle_years),
    ]
    for i, (t, c) in enumerate(zip(plan.timeline, plan.cover), start=1):
        lines.append(f"{i}. {t}" + (f"  \n   Cover crop afterwards: {c}" if c else ""))
    lines += ["", "## Compared with continuing " + b.crops[0], "",
              "| Metric | This plan | Baseline |", "|---|---|---|",
              f"| Relative gross margin (index/yr) | {plan.metrics['profit_index']:.2f} | {b.metrics['profit_index']:.2f} |",
              f"| Irrigation for full supply (mm/yr) | {plan.metrics['irrigation_mm']:.0f} | {b.metrics['irrigation_mm']:.0f} |",
              f"| Soil-health index (0-100) | {plan.metrics['soil_index']:.0f} | {b.metrics['soil_index']:.0f} |",
              f"| Failed seasons (< {FAIL_THRESHOLD:.0%} of potential) | {plan.metrics['failure_pct']:.0f}% | {b.metrics['failure_pct']:.0f}% |",
              "", "## Why"]
    lines += [f"- {w}" for w in plan.why]
    lines += ["", "## Inputs used",
              f"- Soil (0-30 cm): pH {soil.ph:.1f}, organic matter {soil.om_pct:.1f}%, clay {soil.clay_pct:.0f}%, "
              f"sand {soil.sand_pct:.0f}% ({soil.texture}). Source: {soil.source}.",
              f"- Irrigation capacity: {irrigation_cap_mm:.0f} mm per season.",
              "- Priority weights (0-10): " + ", ".join(f"{k} {v:g}" for k, v in weights.items()) + ".",
              "", "## Read this before acting",
              "This is a decision-support *exploration* based on a transparent heuristic model, not a yield forecast. "
              "Indices are relative. Climate comes from a ~50 km model grid; soil from a 250 m model. "
              "Crop parameters are generic reference values and profit uses an index you set. "
              "Check results against local agronomic advice and your own field history."]
    return "\n".join(lines) + "\n"
