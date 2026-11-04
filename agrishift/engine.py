"""Rotation planning engine.

Pipeline (vectorised; a full run takes about a second):

1. ``prepare``            monthly NASA POWER record -> ET0 (Hargreaves) and effective rain
2. ``evaluate_window``    one crop x one sowing month x every historical year:
                          thermal-time phenology, monthly soil-water bucket (FAO-33 response),
                          temperature / heat / frost / waterlogging / soil-pH factors -> yield index 0-1
3. ``evaluate_all``       best sowing month per crop, irrigation need, and the best off-season cover
                          crop that fits in the real gap between any two consecutive crops
4. ``chain_offsets`` +    every N-crop sequence is laid out on a real calendar (sow -> harvest -> gap ->
   ``rotation_metrics``   next sowing) and "replayed" through every historical start year
5. ``run_planner``        min-max normalise criteria, weight them by farmer priorities, return the best
                          plans plus the baseline "keep growing the same crop"

All numbers are *relative* indices from a transparent heuristic model - not yield forecasts.
See README "Method & limitations".
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import agro
from .models import Climate, DataError, Soil

# ---- model constants (documented in README) -------------------------------------------
FAIL_THRESHOLD = 0.40       # a crop-season below 40 % of potential counts as a failed season
SUITABLE_MIN = 0.35         # a crop is a rotation candidate only if its mean index >= this
COVER_MIN = 0.50            # a cover crop is only used if its rain-fed mean index >= this
LEGUME_CARRYOVER = 0.07     # yield bonus for a non-legume that follows a legume (N credit)
SAME_FAMILY_PENALTY = 0.88  # yield multiplier when the same botanical family repeats (disease/pests)
OVERLAP_TOLERANCE = 1       # months the next crop may be sown before the previous is fully harvested
FROST_DAMAGE_TENDER = 0.5   # yield multiplier when a tender crop (t_kill >= 0) meets frost
FROST_DAMAGE_HARDY = 0.1    # ... when a normally hardy crop meets lethal cold (winterkill)
MAX_POOL = 14               # max crops combined into rotations (keeps the search fast)
COVER_COST = 0.3            # margin-index points charged per established cover crop
SOIL_WATER_FRACTION = 0.5   # share of root-zone wetness x capacity assumed available at sowing
MEAN_CANOPY = 0.836         # mean of agro.kc_shape over a season (canopy proxy used as growth reference)

CRITERIA = ("profit", "water", "soil", "risk")


# ---- climate preparation ----------------------------------------------------------------
def prepare(clim: Climate) -> dict:
    """Monthly arrays (n_years, 12) used by the crop model."""
    c = clim.filled()
    ra = np.array([agro.extraterrestrial_radiation(c.lat, d) for d in agro.MID_DOY])
    et0 = agro.hargreaves_et0(c.tmean, c.tmax, c.tmin, ra[None, :]) * agro.DAYS_IN_MONTH[None, :]
    return {"years": np.asarray(c.years, dtype=int), "tmean": c.tmean, "tmax": c.tmax,
            "tmin": c.tmin, "precip": c.precip_mm, "peff": agro.effective_rain(c.precip_mm),
            "root_wet": c.root_wet, "et0": et0, "ra": ra}


def climate_summary(cd: dict) -> dict:
    """Headline numbers, long-term monthly means and indicative linear trends."""
    years = cd["years"].astype(float)
    ann_t = cd["tmean"].mean(axis=1)
    ann_p = cd["precip"].sum(axis=1)
    ann_et0 = cd["et0"].sum(axis=1)

    def slope_per_decade(y):
        return float(np.polyfit(years, y, 1)[0] * 10.0) if len(years) >= 8 else float("nan")

    return {
        "first_year": int(cd["years"][0]), "last_year": int(cd["years"][-1]),
        "t_mean": float(ann_t.mean()), "precip": float(ann_p.mean()), "et0": float(ann_et0.mean()),
        "aridity": float(ann_p.mean() / max(ann_et0.mean(), 1e-9)),
        "t_trend_per_decade": slope_per_decade(ann_t),
        "p_trend_pct_per_decade": slope_per_decade(ann_p) / max(float(ann_p.mean()), 1e-9) * 100.0,
        "monthly": pd.DataFrame({"Month": agro.MONTH_NAMES,
                                 "Temperature (°C)": cd["tmean"].mean(axis=0),
                                 "Rain (mm)": cd["precip"].mean(axis=0),
                                 "ET0 (mm)": cd["et0"].mean(axis=0)}),
        "annual": pd.DataFrame({"Year": cd["years"], "Temperature (°C)": ann_t,
                                "Rain (mm)": ann_p, "ET0 (mm)": ann_et0}),
    }


# ---- crop x window model -----------------------------------------------------------------
def evaluate_window(crop, sow_m: int, cd: dict, soil: Soil, irrigation_cap_mm: float) -> dict:
    """Simulate one crop sown in month ``sow_m`` in every sowing year 0..Y-2.

    * Phenology: thermal time (growing-degree-days above ``t_base``, capped at ``t_opt_hi``) decides how
      long the cycle lasts - cool spells lengthen it, heat shortens it.
    * Water: monthly soil-water bucket (supply = store + effective rain + irrigation up to the cap;
      demand = Kc x ET0); FAO-33 response: relative yield = 1 - Ky (1 - ETa/ETc).
    * Growth (radiation-use-efficiency logic): growth = sum over the cycle of
      radiation x canopy cover x temperature efficiency, where temperature efficiency combines the
      temperature trapezoid, heat stress (monthly mean of daily max) and waterlogging (crops that are
      not flood-tolerant). It is compared with the growth of a full ``season_days`` cycle in the
      sunniest part of the year, so cool, sunny, long seasons score high and hot, compressed or
      dark-season cycles score low.
    * Cold survival: a month whose mean daily minimum is below the crop's ``t_kill`` damages tender
      crops (x0.5) or winterkills hardy ones (x0.1); soil pH scales the result.
    * A crop that does not reach maturity within 12 months yields 0.
    """
    ye = cd["tmean"].shape[0] - 1            # last year dropped so seasons may run into the "next" year
    taw = soil.pawc_mm_per_m * float(crop.root_depth_m)
    store = SOIL_WATER_FRACTION * cd["root_wet"][:ye, sow_m] * taw
    irr_left = np.full(ye, float(irrigation_cap_mm))
    gdd_req = float(crop.gdd_req)
    t_opt_hi = float(crop.t_opt_hi)
    tmax_tol = t_opt_hi + 8.0
    gdd, days, months = np.zeros(ye), np.zeros(ye), np.zeros(ye)
    etc_tot, eta_tot, irr_tot, fsum, growth = (np.zeros(ye) for _ in range(5))
    frost = np.zeros(ye, dtype=bool)

    for k in range(12):
        mi = sow_m + k
        m, off = mi % 12, mi // 12
        sl = slice(off, off + ye)
        t = cd["tmean"][sl, m]
        dgdd = np.maximum(np.minimum(t, t_opt_hi) - float(crop.t_base), 0.0) * agro.DAYS_IN_MONTH[m]
        remaining = np.maximum(gdd_req - gdd, 0.0)
        # share of this month during which the crop is still growing
        frac = np.where(remaining <= 0.0, 0.0,
                        np.where(dgdd > 0.0, np.minimum(1.0, remaining / np.maximum(dgdd, 1e-9)), 1.0))
        if not frac.any():
            break
        progress = np.clip((gdd + 0.5 * frac * dgdd) / gdd_req, 0.0, 1.0)
        etc = frac * (float(crop.kc_mid) * agro.kc_shape(progress) * cd["et0"][sl, m]
                      + float(crop.paddy_extra_mm_month))
        avail = store + frac * cd["peff"][sl, m]
        eta_rain = np.minimum(etc, avail)
        irr = np.minimum(etc - eta_rain, irr_left)
        irr_left = irr_left - irr
        store = np.clip(avail - eta_rain, 0.0, taw)
        etc_tot += etc
        eta_tot += eta_rain + irr
        irr_tot += irr
        f = (agro.trapezoid(t, crop.t_lo, crop.t_opt_lo, crop.t_opt_hi, crop.t_hi)
             * agro.heat_factor(cd["tmax"][sl, m], tmax_tol))
        if not crop.waterlog_tolerant:
            f = f * agro.excess_rain_factor(cd["precip"][sl, m])
        fsum += frac * f
        growth += frac * agro.DAYS_IN_MONTH[m] * cd["ra"][m] * agro.kc_shape(progress) * f
        months += frac
        days += frac * agro.DAYS_IN_MONTH[m]
        gdd = gdd + frac * dgdd
        frost |= (frac > 0.0) & (cd["tmin"][sl, m] < float(crop.t_kill))

    complete = gdd >= gdd_req - 1e-6
    tfac = np.where(months > 0, fsum / np.maximum(months, 1e-9), 0.0)
    growth_ref = max(float(cd["ra"].max()), 1e-6) * float(crop.season_days) * MEAN_CANOPY
    gfac = np.clip(growth / growth_ref, 0.0, 1.0)
    rel_et = np.where(etc_tot > 0, eta_tot / np.maximum(etc_tot, 1e-9), 1.0)
    fw = np.clip(1.0 - float(crop.ky) * (1.0 - rel_et), 0.0, 1.0)
    fs = max(0.1, float(agro.trapezoid(soil.ph, crop.ph_lo, crop.ph_opt_lo, crop.ph_opt_hi, crop.ph_hi)))
    frost_mult = FROST_DAMAGE_TENDER if float(crop.t_kill) >= 0 else FROST_DAMAGE_HARDY
    yld = gfac * fw * fs * np.where(frost, frost_mult, 1.0) * complete
    return {"yield_idx": yld, "irr": irr_tot, "etc": etc_tot, "eta": eta_tot, "tfac": tfac, "gfac": gfac,
            "months": months, "days": days, "complete": complete, "fs": fs}


@dataclass
class CropTable:
    df: pd.DataFrame             # per-crop summary for display
    sow: np.ndarray              # (C,) best sowing month 0..11
    nmon: np.ndarray             # (C,) typical cycle length in months at that sowing month
    Y: np.ndarray                # (C, Ye) yield index at best window, under the irrigation cap
    IRR: np.ndarray              # (C, Ye) irrigation needed for full water supply (mm per season)
    mean_yield: np.ndarray       # (C,)
    pair_cover: np.ndarray       # (C, C) best cover crop for the gap after crop g and before crop h (-1 none)
    pair_cover_sow: np.ndarray   # (C, C) its sowing month


def evaluate_all(cd: dict, soil: Soil, crops: pd.DataFrame, irrigation_cap_mm: float) -> CropTable:
    n_crops = len(crops)
    ye = cd["tmean"].shape[0] - 1
    sow = np.zeros(n_crops, dtype=int)
    nmon = np.ones(n_crops, dtype=int)
    cycle_days = np.zeros(n_crops)
    Y, IRR = np.zeros((n_crops, ye)), np.zeros((n_crops, ye))
    cov_mean = np.zeros((n_crops, 12))
    cov_len = np.ones((n_crops, 12), dtype=int)
    is_cover = crops.role.isin(["cover", "both"]).to_numpy()
    is_main = crops.role.isin(["cash", "both"]).to_numpy()
    nfix = crops.n_fixer.to_numpy()

    for g, crop in enumerate(crops.itertuples()):
        means, irr_means, lens = np.zeros(12), np.zeros(12), np.ones(12, dtype=int)
        for s in range(12):
            r = evaluate_window(crop, s, cd, soil, irrigation_cap_mm)
            means[s], irr_means[s] = r["yield_idx"].mean(), r["irr"].mean()
            lens[s] = max(1, int(np.ceil(np.median(r["months"]))))
            if is_cover[g]:                                  # cover crops are always rain-fed
                rc = evaluate_window(crop, s, cd, soil, 0.0)
                cov_mean[g, s] = rc["yield_idx"].mean()
                cov_len[g, s] = max(1, int(np.ceil(np.median(rc["months"]))))
        rounded = np.round(means, 2)
        ties = np.where(rounded == rounded.max())[0]
        sow[g] = int(ties[np.argmin(irr_means[ties])])       # among equally good windows: least irrigation
        nmon[g] = lens[sow[g]]
        best = evaluate_window(crop, sow[g], cd, soil, irrigation_cap_mm)
        Y[g], cycle_days[g] = best["yield_idx"], float(np.median(best["days"]))
        IRR[g] = evaluate_window(crop, sow[g], cd, soil, 1e9)["irr"]

    # best cover crop for the real gap between consecutive main crops (harvest of g -> sowing of h)
    pair_cover = np.full((n_crops, n_crops), -1, dtype=int)
    pair_sow = np.zeros((n_crops, n_crops), dtype=int)
    cover_ids = np.where(is_cover)[0]
    for g in np.where(is_main)[0]:
        end = int(sow[g] + nmon[g])
        for h in np.where(is_main)[0]:
            earliest = end - OVERLAP_TOLERANCE
            gap = earliest + int((sow[h] - earliest) % 12) - end
            if gap < 1:
                continue
            best_score = -1.0
            for kc in cover_ids:
                if kc == g or kc == h:
                    continue
                for a in range(gap):
                    s = (end + a) % 12
                    if a + cov_len[kc, s] <= gap and cov_mean[kc, s] >= COVER_MIN:
                        score = cov_mean[kc, s] + (0.05 if nfix[kc] else 0.0)
                        if score > best_score:
                            best_score, pair_cover[g, h], pair_sow[g, h] = score, kc, s

    mean_y = Y.mean(axis=1)
    viable_any = Y.max(axis=1) > 0
    df = pd.DataFrame({
        "crop": crops.crop, "type": crops.type, "family": crops.family, "role": crops.role,
        "n_fixer": nfix,
        "best_sow": [agro.MONTH_NAMES[s] if v else "–" for s, v in zip(sow, viable_any)],
        "cycle_days": np.round(cycle_days),
        "mean_yield_idx": mean_y, "irrigation_need_mm": IRR.mean(axis=1),
        "season_failure_pct": (Y < FAIL_THRESHOLD).mean(axis=1) * 100.0,
        "viable": mean_y >= SUITABLE_MIN,
    })
    return CropTable(df=df, sow=sow, nmon=nmon, Y=Y, IRR=IRR, mean_yield=mean_y,
                     pair_cover=pair_cover, pair_cover_sow=pair_sow)


# ---- rotation search ------------------------------------------------------------------------
def select_pool(tab: CropTable, crops: pd.DataFrame, allowed: np.ndarray, max_pool: int = MAX_POOL) -> list[int]:
    """Viable main crops ranked by suitability x margin, max 3 per botanical family, some legumes kept."""
    main = crops.role.isin(["cash", "both"]).to_numpy()
    fam, nfix = crops.family.to_numpy(), crops.n_fixer.to_numpy()
    rank = tab.mean_yield * (crops.margin_index.to_numpy(float) + 2.0)
    cand = [g for g in np.argsort(-rank, kind="stable")
            if main[g] and allowed[g] and tab.mean_yield[g] >= SUITABLE_MIN]
    per_family: dict[str, int] = {}
    ranked: list[int] = []
    for g in cand:
        if per_family.get(fam[g], 0) < 3:
            ranked.append(int(g))
            per_family[fam[g]] = per_family.get(fam[g], 0) + 1
    pool = ranked[:max_pool]
    legumes = [g for g in ranked if nfix[g]]
    need = min(2, len(legumes))
    for g in legumes:
        if sum(bool(nfix[x]) for x in pool) >= need:
            break
        if g not in pool:
            drop = next((x for x in reversed(pool) if not nfix[x]), None)
            if drop is not None:
                pool.remove(drop)
                pool.append(g)
    return pool


def enumerate_rotations(pool: list[int], n_crops: int, current_g: int) -> np.ndarray:
    """All N-crop sequences from ``pool`` (global crop indices), shape (M, N).

    The first crop must differ from the current one and no crop may follow itself.
    """
    if len(pool) < 2:
        return np.zeros((0, n_crops), dtype=np.int32)
    idx = np.array(list(itertools.product(range(len(pool)), repeat=n_crops)), dtype=np.int32)
    G = np.array(pool, dtype=np.int32)[idx]
    ok = G[:, 0] != current_g
    for i in range(n_crops - 1):
        ok &= G[:, i] != G[:, i + 1]
    return G[ok]


def chain_offsets(G: np.ndarray, tab: CropTable) -> np.ndarray:
    """Absolute sowing month of every slot, counted from January of the first sowing year, shape (M, N).

    Each next crop is sown in its own best month, but not before the previous crop is harvested
    (minus ``OVERLAP_TOLERANCE``). Slot year offset = ``starts // 12``; gaps become fallow or cover.
    """
    n_rot, n = G.shape
    starts = np.empty((n_rot, n), dtype=int)
    starts[:, 0] = tab.sow[G[:, 0]]
    for i in range(n - 1):
        earliest = starts[:, i] + tab.nmon[G[:, i]] - OVERLAP_TOLERANCE
        starts[:, i + 1] = earliest + ((tab.sow[G[:, i + 1]] - earliest) % 12)
    return starts


def cycle_years(G: np.ndarray, tab: CropTable, starts: np.ndarray) -> np.ndarray:
    """Length in years of one full, repeating rotation cycle (a whole number of years).

    It runs from the first sowing to the next cycle's first sowing, so fallow gaps and winter
    crops that block the field are charged, and double cropping is rewarded.
    """
    last_end = starts[:, -1] + tab.nmon[G[:, -1]] - OVERLAP_TOLERANCE
    next_start = last_end + ((tab.sow[G[:, 0]] - last_end) % 12)
    return np.maximum((next_start - starts[:, 0]) / 12.0, 1.0)


@dataclass
class Context:
    tab: CropTable
    fam: np.ndarray
    nfix: np.ndarray
    residue: np.ndarray
    margin: np.ndarray
    pair_nfix: np.ndarray        # (C, C) the cover crop of this gap fixes nitrogen
    pair_res: np.ndarray         # (C, C) residue score of the cover crop of this gap
    use_cover: bool
    current_g: int


def build_context(tab: CropTable, crops: pd.DataFrame, use_cover: bool, current_g: int) -> Context:
    """Lookup arrays shared by all rotation calculations."""
    has = tab.pair_cover >= 0
    safe = np.where(has, tab.pair_cover, 0)
    nfix = crops.n_fixer.to_numpy()
    residue = crops.residue_om.to_numpy(float)
    return Context(tab=tab, fam=pd.factorize(crops.family)[0], nfix=nfix, residue=residue,
                   margin=crops.margin_index.to_numpy(float), pair_nfix=nfix[safe] & has,
                   pair_res=np.where(has, residue[safe], 0.0), use_cover=use_cover, current_g=current_g)


def rotation_metrics(G: np.ndarray, ctx: Context, n_start: int | None = None) -> dict:
    """Replay every rotation through every historical start year and summarise it.

    Income and irrigation are *per calendar year of the repeating cycle*, so fallow gaps are not free.
    """
    n_rot, n = G.shape
    tab = ctx.tab
    starts = chain_offsets(G, tab)
    offs = starts // 12
    ye = tab.Y.shape[1]
    if n_start is None:
        n_start = ye - int(offs.max())
    if n_start < 8:
        raise DataError("The climate record is too short for this rotation length.")
    cyc = cycle_years(G, tab, starts)

    slot_y = np.empty((n_rot, n, n_start))
    slot_irr = np.empty((n_rot, n, n_start))
    prev = np.full(n_rot, ctx.current_g, dtype=int)
    for i in range(n):
        g = G[:, i]
        cols = offs[:, i][:, None] + np.arange(n_start)[None, :]
        mult = (np.where(ctx.fam[g] == ctx.fam[prev], SAME_FAMILY_PENALTY, 1.0)
                * np.where(ctx.nfix[prev] & ~ctx.nfix[g], 1.0 + LEGUME_CARRYOVER, 1.0))
        slot_y[:, i, :] = np.minimum(1.0, tab.Y[g[:, None], cols] * mult[:, None])
        slot_irr[:, i, :] = tab.IRR[g[:, None], cols]
        prev = g

    # cover crops in the real gap *after* slot i (none after the last crop of the plan)
    cover = np.zeros((n_rot, n), dtype=bool)
    c_nfix = np.zeros((n_rot, n), dtype=bool)
    c_res = np.zeros((n_rot, n))
    if ctx.use_cover:
        for i in range(n - 1):
            gi, gj = G[:, i], G[:, i + 1]
            cover[:, i] = tab.pair_cover[gi, gj] >= 0
            c_nfix[:, i] = ctx.pair_nfix[gi, gj]
            c_res[:, i] = ctx.pair_res[gi, gj]
    income = slot_y * ctx.margin[G][:, :, None] - cover[:, :, None] * COVER_COST

    # soil-health index (0-100): nitrogen input, residue return, family diversity, off-season cover
    n_input = ctx.nfix[G] | (cover & c_nfix)
    res_year = np.minimum(1.0, ctx.residue[G] + 0.5 * cover * c_res)
    s_sorted = np.sort(ctx.fam[G], axis=1)
    distinct = 1 + (np.diff(s_sorted, axis=1) != 0).sum(axis=1)
    div_score = (distinct - 1) / max(n - 1, 1)
    cover_frac = cover[:, : n - 1].mean(axis=1) if n > 1 else np.zeros(n_rot)
    w = (0.30, 0.25, 0.25, 0.20) if ctx.use_cover else (0.375, 0.3125, 0.3125, 0.0)
    soil_idx = 100.0 * (w[0] * np.minimum(1.0, n_input.mean(axis=1) / 0.34)
                        + w[1] * res_year.mean(axis=1) + w[2] * div_score + w[3] * cover_frac)
    return {"income": income.sum(axis=1).mean(axis=1) / cyc,
            "irr": slot_irr.sum(axis=1).mean(axis=1) / cyc,
            "irr_peak": slot_irr.mean(axis=2).max(axis=1), "soil": soil_idx,
            "fail": (slot_y < FAIL_THRESHOLD).mean(axis=(1, 2)), "yield": slot_y.mean(axis=(1, 2)),
            "slot_y": slot_y, "starts": starts, "cycle_years": cyc, "cover": cover,
            "n_start": n_start}


def _norm(x: np.ndarray, floor: float) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / max(hi - lo, floor)


def composite_score(m: dict, weights: dict) -> np.ndarray:
    """Weighted sum of min-max normalised criteria (1 = best of the candidates)."""
    w = np.array([max(float(weights.get(k, 0.0)), 0.0) for k in CRITERIA])
    if w.sum() <= 0:
        w = np.ones(4)
    s = np.vstack([_norm(m["income"], 1.0), 1.0 - _norm(m["irr"], 100.0),
                   _norm(m["soil"], 15.0), 1.0 - _norm(m["fail"], 0.05)])
    return (w[:, None] * s).sum(axis=0) / w.sum()


# ---- results ---------------------------------------------------------------------------------
@dataclass
class Plan:
    labels: list[str]
    crops: list[str]
    timeline: list[str]                   # "Wheat (winter): sow Oct (yr 1) -> harvest Jul (yr 2)"
    cover: list[str]                      # cover crop in the gap after each crop ("" = none)
    cycle_years: float                    # the plan repeats every this many years
    metrics: dict
    yield_by_start: np.ndarray            # (S,) mean yield index of the rotation per start year
    slot_yield: np.ndarray                # (N, S)
    why: list[str] = field(default_factory=list)


@dataclass
class PlannerResult:
    baseline: Plan
    plans: list[Plan]
    crop_table: pd.DataFrame
    start_years: np.ndarray
    climate: dict
    n_candidates: int
    pool: list[str]
    warnings: list[str]


def _metrics_row(m: dict, i: int, score: float) -> dict:
    return {"profit_index": float(m["income"][i]), "irrigation_mm": float(m["irr"][i]),
            "irrigation_peak_mm": float(m["irr_peak"][i]),
            "soil_index": float(m["soil"][i]), "failure_pct": float(m["fail"][i] * 100.0),
            "mean_yield_idx": float(m["yield"][i]), "score": float(score)}


def _make_plan(labels, row, ctx, crops, single, metrics) -> Plan:
    tab, n = ctx.tab, len(row)
    starts = single["starts"][0]
    timeline, cover = [], []
    for i, g in enumerate(row):
        sow_abs, end_abs = int(starts[i]), int(starts[i] + tab.nmon[g] - 1)
        timeline.append(f"{crops.crop[g]}: sow {agro.MONTH_NAMES[sow_abs % 12]} (yr {sow_abs // 12 + 1}) "
                        f"→ harvest {agro.MONTH_NAMES[end_abs % 12]} (yr {end_abs // 12 + 1})")
        if ctx.use_cover and i < n - 1 and tab.pair_cover[g, row[i + 1]] >= 0:
            k = tab.pair_cover[g, row[i + 1]]
            cover.append(f"{crops.crop[k].replace(' (cover)', '')} "
                         f"(sow {agro.MONTH_NAMES[tab.pair_cover_sow[g, row[i + 1]]]})")
        else:
            cover.append("")
    return Plan(labels=list(labels), crops=[crops.crop[g] for g in row], timeline=timeline, cover=cover,
                cycle_years=float(single["cycle_years"][0]), metrics=metrics,
                yield_by_start=single["slot_y"][0].mean(axis=0), slot_yield=single["slot_y"][0])


def explain(plan: Plan, base: Plan, crops: pd.DataFrame, irrigation_cap: float) -> list[str]:
    """Plain-language 'why' lines that cite the numbers behind the recommendation."""
    b = base.crops[0]
    out = []
    saved = base.metrics["irrigation_mm"] - plan.metrics["irrigation_mm"]
    if saved > 25:
        out.append(f"💧 Needs **{saved:.0f} mm/yr less irrigation** than continuing {b} "
                   f"(≈ {saved * 10:,.0f} m³ per hectare per year).")
    elif saved < -25:
        out.append(f"💧 Needs **{-saved:.0f} mm/yr more irrigation** than continuing {b}.")
    else:
        out.append(f"💧 Irrigation need is similar to continuing {b} (within 25 mm/yr).")
    if irrigation_cap < plan.metrics["irrigation_peak_mm"] - 5:
        out.append(f"⚠️ Your thirstiest crop season would need about {plan.metrics['irrigation_peak_mm']:.0f} mm of "
                   f"irrigation for full water supply, more than the {irrigation_cap:.0f} mm per season you stated - "
                   "the yield index already includes that water stress.")
    legumes = sorted({c for c in plan.crops if crops.loc[crops.crop == c, "n_fixer"].iloc[0]})
    fams = len({crops.loc[crops.crop == c, "family"].iloc[0] for c in plan.crops})
    bits = [f"{fams} botanical famil{'y' if fams == 1 else 'ies'} across {len(plan.crops)} crops"]
    if legumes:
        bits.append("nitrogen-fixing " + ", ".join(legumes))
    covers = sorted({c.split(" (")[0] for c in plan.cover if c})
    if covers:
        bits.append("off-season cover: " + ", ".join(covers))
    out.append(f"🌱 Soil-health index **{plan.metrics['soil_index']:.0f}/100** "
               f"(baseline {base.metrics['soil_index']:.0f}): " + "; ".join(bits) + ".")
    out.append(f"🛡️ In **{plan.metrics['failure_pct']:.0f}%** of replayed crop-seasons the yield index fell below "
               f"{FAIL_THRESHOLD:.0%} of potential, versus **{base.metrics['failure_pct']:.0f}%** for continuing {b}.")
    out.append(f"💰 Relative gross-margin index {plan.metrics['profit_index']:.1f}/yr vs "
               f"{base.metrics['profit_index']:.1f}/yr (from the margin values you can edit - not market prices).")
    return out


def run_planner(clim: Climate, soil: Soil, crops: pd.DataFrame, current_crop: str,
                irrigation_cap_mm: float = 0.0, rotation_years: int = 3,
                weights: dict | None = None, use_cover: bool = True,
                allowed_crops: list[str] | None = None,
                margin_overrides: dict | None = None) -> PlannerResult:
    """Full planning run (``rotation_years`` = number of crops in the plan). Raises DataError for user-visible problems."""
    weights = weights or {k: 5.0 for k in CRITERIA}
    crops = crops.reset_index(drop=True).copy()
    for name, val in (margin_overrides or {}).items():
        crops.loc[crops.crop == name, "margin_index"] = float(val)
    if current_crop not in set(crops.loc[crops.role.isin(["cash", "both"]), "crop"]):
        raise DataError(f"'{current_crop}' is not a crop that can be rotated.")
    if not 2 <= rotation_years <= 4:
        raise ValueError("rotation_years must be between 2 and 4")

    cd = prepare(clim)
    tab = evaluate_all(cd, soil, crops, irrigation_cap_mm)
    current_g = int(crops.index[crops.crop == current_crop][0])
    allowed = (crops.crop.isin(allowed_crops).to_numpy() if allowed_crops is not None
               else np.ones(len(crops), dtype=bool))
    ctx = build_context(tab, crops, use_cover, current_g)

    pool = select_pool(tab, crops, allowed)
    G = enumerate_rotations(pool, rotation_years, current_g)
    base_row = np.full((1, rotation_years), current_g, dtype=np.int32)
    all_g = np.vstack([base_row, G]) if len(G) else base_row
    n_start = tab.Y.shape[1] - int(chain_offsets(all_g, tab).max() // 12)
    m = rotation_metrics(all_g, ctx, n_start=n_start)
    score = composite_score(m, weights)
    start_years = cd["years"][:n_start]

    base_single = rotation_metrics(base_row, ctx, n_start=n_start)
    base_plan = _make_plan([f"Baseline: keep growing {current_crop}"], base_row[0], ctx, crops, base_single,
                           _metrics_row(m, 0, score[0]))

    plans: list[Plan] = []
    warnings: list[str] = []
    if len(G):
        sc, inc, irr, soil_i = score[1:], m["income"][1:], m["irr"][1:], m["soil"][1:]
        fail, yl = m["fail"][1:], m["yield"][1:]
        picks = [("⭐ Best for your priorities", int(np.argmax(sc))),
                 ("💧 Best for water saving", int(np.lexsort((-sc, np.round(irr / 5.0)))[0])),
                 ("🌱 Best for soil health", int(np.lexsort((-sc, -np.round(soil_i)))[0])),
                 ("💰 Best for profit", int(np.lexsort((-sc, -np.round(inc, 2)))[0])),
                 ("🛡️ Most stable", int(np.lexsort((-sc, -yl, fail))[0]))]
        seen: dict[tuple, Plan] = {}
        for label, i in picks:
            key = tuple(int(x) for x in G[i])
            if key in seen:
                seen[key].labels.append(label)
                continue
            single = rotation_metrics(G[i:i + 1], ctx, n_start=n_start)
            p = _make_plan([label], G[i], ctx, crops, single, _metrics_row(m, i + 1, sc[i]))
            seen[key] = p
            plans.append(p)
        for p in plans:
            p.why = explain(p, base_plan, crops, irrigation_cap_mm)
    else:
        warnings.append("Not enough viable crops here to build a rotation (need at least two). "
                        "Check the NASA climate tab - water supply or temperature may be limiting.")

    cur_y = float(tab.mean_yield[current_g])
    if cur_y < SUITABLE_MIN:
        warnings.append(f"{current_crop} has a low climate-suitability index ({cur_y:.2f}) at this location "
                        "- the baseline is expected to struggle.")
    clim_sum = climate_summary(cd)
    if clim_sum["aridity"] < 0.2 and irrigation_cap_mm <= 0:
        warnings.append("This site is arid (rain < 20 % of reference evapotranspiration): rain-fed cropping is "
                        "marginal. Enter an irrigation capacity to see irrigated options.")
    return PlannerResult(baseline=base_plan, plans=plans, crop_table=tab.df, start_years=start_years,
                         climate=clim_sum, n_candidates=int(len(G)), pool=[crops.crop[g] for g in pool],
                         warnings=warnings)
