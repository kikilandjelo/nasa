"""AgriShift v2 - crop-rotation explorer built on NASA data.

NASA Space Apps Challenge 2026 - "Field Shift: Adapting Farms with NASA Data".
All science lives in the ``agrishift`` package; this file only handles the user interface.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from agrishift import crops as crop_db
from agrishift import engine, geo, mapview, sources
from agrishift.lineage import lineage_frame
from agrishift.models import DEFAULT_SOIL, DataError, Soil
from agrishift.report import plan_to_markdown

try:  # optional: browser GPS button
    from streamlit_geolocation import streamlit_geolocation
except ImportError:  # pragma: no cover
    streamlit_geolocation = None

st.set_page_config(page_title="AgriShift - rotation planner on NASA data", page_icon="🌾", layout="wide")

DEFAULT_LAT, DEFAULT_LON = 40.1438, 47.5769
PRESETS = {
    "⚖️ Balanced": {"water": 5, "soil": 5, "profit": 5, "risk": 5},
    "💧 Save water": {"water": 9, "soil": 4, "profit": 3, "risk": 5},
    "🌱 Restore soil": {"water": 4, "soil": 9, "profit": 3, "risk": 5},
    "💰 Maximise profit": {"water": 2, "soil": 3, "profit": 9, "risk": 4},
}
MONTH_LABELS = [f"{i + 1:02d}-{m}" for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])]


# ----------------------------------------------------------------------------- cached data access
@st.cache_data(show_spinner=False)
def get_crops() -> pd.DataFrame:
    return crop_db.load_crops()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_climate(lat: float, lon: float, start: int, end: int):
    return sources.fetch_power_monthly(lat, lon, start, end)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_soil(lat: float, lon: float):
    return sources.fetch_soilgrids(lat, lon)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_ndvi(lat: float, lon: float, year: int):
    return sources.fetch_modis_ndvi(lat, lon, year)


# ----------------------------------------------------------------------------- state and callbacks
def init_state(crops_df: pd.DataFrame) -> None:
    main = crop_db.main_crop_names(crops_df)
    defaults = {
        "lat": DEFAULT_LAT, "lon": DEFAULT_LON, "last_geo": None, "url_msg": "", "data": None, "ndvi": None,
        "preset": list(PRESETS)[0], "w_water": 5, "w_soil": 5, "w_profit": 5, "w_risk": 5,
        "current_crop": "Wheat (winter)" if "Wheat (winter)" in main else main[0],
        "irrigation_mm": 0, "n_crops": 3, "use_cover": True,
        "types": list(crop_db.DEFAULT_TYPES), "maps_url": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_preset() -> None:
    for key, value in PRESETS.get(st.session_state.get("preset"), {}).items():
        st.session_state[f"w_{key}"] = value


def apply_maps_url() -> None:
    text = (st.session_state.get("maps_url") or "").strip()
    if not text:
        st.session_state["url_msg"] = ""
        return
    coords = geo.extract_lat_lon_from_url(text)
    if coords:
        st.session_state["lat"], st.session_state["lon"] = coords
        st.session_state["url_msg"] = f"✅ Location set to {coords[0]:.4f}, {coords[1]:.4f}"
    else:
        st.session_state["url_msg"] = ("⚠️ Could not find coordinates in that text. Paste a Google Maps link "
                                       "or 'latitude, longitude'.")


# ----------------------------------------------------------------------------- sidebar
def sidebar(crops_df: pd.DataFrame) -> bool:
    main_crops = crop_db.main_crop_names(crops_df)
    types = sorted(crops_df.loc[crops_df.role.isin(["cash", "both"]), "type"].unique())
    with st.sidebar:
        st.title("🌾 AgriShift")
        st.caption("Explore crop rotations that fit your changing climate, using NASA data.")

        st.subheader("1 · Your field")
        if streamlit_geolocation is not None:
            loc = streamlit_geolocation()
            if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
                cur = (float(loc["latitude"]), float(loc["longitude"]))
                if st.session_state["last_geo"] != cur:
                    st.session_state["lat"], st.session_state["lon"] = cur
                    st.session_state["last_geo"] = cur
        st.text_input("Google Maps link or 'lat, lon'", key="maps_url", on_change=apply_maps_url,
                      placeholder="https://maps.app.goo.gl/…  or  40.1438, 47.5769")
        if st.session_state["url_msg"]:
            st.caption(st.session_state["url_msg"])
        st.number_input("Latitude", -90.0, 90.0, step=0.01, format="%.4f", key="lat")
        st.number_input("Longitude", -180.0, 180.0, step=0.01, format="%.4f", key="lon")

        st.subheader("2 · Your farm")
        st.selectbox("Crop you grow now", main_crops, key="current_crop")
        st.number_input("Irrigation available (mm per season, 0 = rain-fed)", 0, 1500, step=25, key="irrigation_mm",
                        help="Water you can add per crop season. 100 mm = 1,000 m³ per hectare.")
        st.multiselect("Crop types I am willing to grow", types, key="types",
                       help="Vegetables are off by default; add them if you grow them.")

        st.subheader("3 · Your priorities")
        st.radio("Main priority", list(PRESETS), key="preset", on_change=apply_preset)
        with st.expander("Fine-tune (0 = ignore, 10 = vital)"):
            st.slider("💧 Water saving", 0, 10, key="w_water")
            st.slider("🌱 Soil health", 0, 10, key="w_soil")
            st.slider("💰 Profit", 0, 10, key="w_profit")
            st.slider("🛡️ Stability (fewer failed seasons)", 0, 10, key="w_risk")
        st.slider("Crops in the plan", 2, 4, key="n_crops")
        st.checkbox("Use off-season cover crops where they fit", key="use_cover")
        return st.button("🚀 Fetch NASA data & plan rotations", type="primary")


# ----------------------------------------------------------------------------- data fetching
def fetch_data() -> None:
    lat, lon = round(float(st.session_state["lat"]), 3), round(float(st.session_state["lon"]), 3)
    start, end = sources.default_year_range()
    with st.spinner(f"Requesting {end - start + 1} years of NASA POWER (MERRA-2) monthly data…"):
        try:
            clim = load_climate(lat, lon, start, end)
        except DataError as exc:
            st.error(f"NASA POWER request failed: {exc}")
            st.stop()
    soil_raw, soil_err = None, ""
    with st.spinner("Requesting soil properties (ISRIC SoilGrids - not a NASA product)…"):
        try:
            soil_raw = load_soil(lat, lon)
        except DataError as exc:
            soil_err = str(exc)
    soil0 = soil_raw or DEFAULT_SOIL
    # keep fetched values inside the ranges of the input widgets (SoilGrids can return extremes, e.g. peat)
    initial = (round(min(max(soil0.ph, 3.5), 9.5), 1), round(min(max(soil0.om_pct, 0.0), 20.0), 1),
               int(min(max(round(soil0.clay_pct), 0), 80)), int(min(max(round(soil0.sand_pct), 0), 95)))
    st.session_state.update({
        "data": {"clim": clim, "soil_raw": soil_raw, "soil_err": soil_err, "lat": lat, "lon": lon,
                 "initial": initial},
        "soil_ph": initial[0], "soil_om": initial[1], "soil_clay": initial[2], "soil_sand": initial[3],
        "ndvi": None,
    })


# ----------------------------------------------------------------------------- results
def soil_inputs(data: dict) -> Soil:
    with st.expander("🧪 Soil - auto-filled from SoilGrids; replace with your own soil test if you have one",
                     expanded=bool(data["soil_err"])):
        if data["soil_err"]:
            st.warning(f"Soil data unavailable ({data['soil_err']}). Typical loam is assumed - please enter "
                       "your own values.")
        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("pH", 3.5, 9.5, step=0.1, format="%.1f", key="soil_ph")
        c2.number_input("Organic matter %", 0.0, 20.0, step=0.1, format="%.1f", key="soil_om")
        c3.number_input("Clay %", 0, 80, step=1, key="soil_clay")
        c4.number_input("Sand %", 0, 95, step=1, key="soil_sand")
    current = (round(float(st.session_state["soil_ph"]), 1), round(float(st.session_state["soil_om"]), 1),
               round(float(st.session_state["soil_clay"])), round(float(st.session_state["soil_sand"])))
    if data["soil_raw"] is not None and current == data["initial"]:
        source = data["soil_raw"].source
    elif data["soil_raw"] is None and current == data["initial"]:
        source = DEFAULT_SOIL.source
    else:
        source = "Entered by the farmer"
    return Soil(ph=current[0], om_pct=current[1], clay_pct=min(current[2], 100 - current[3]),
                sand_pct=current[3], source=source)


def margin_inputs(crops_df: pd.DataFrame) -> dict:
    with st.expander("💰 Local profitability - edit these 0-10 margin values to match your market"):
        st.caption("Defaults are illustrative, NOT market prices. The profit criterion is only as good as these "
                   "numbers; set them from your own costs and selling prices.")
        base = crops_df.loc[crops_df.role.isin(["cash", "both"]), ["crop", "margin_index"]].reset_index(drop=True)
        edited = st.data_editor(
            base, key="margin_editor", hide_index=True, disabled=["crop"], num_rows="fixed",
            column_config={"margin_index": st.column_config.NumberColumn("Margin index (0-10)", min_value=0.0,
                                                                         max_value=10.0, step=0.5)})
    return {c: float(v) for c, v in zip(edited["crop"], edited["margin_index"]) if pd.notna(v)}


def render_plan(plan: engine.Plan, base: engine.Plan, idx: int, ctx: dict) -> None:
    with st.expander(f"{' · '.join(plan.labels)}  —  {' → '.join(plan.crops)}", expanded=(idx == 0)):
        st.markdown(f"**Calendar** (this plan repeats every {round(plan.cycle_years)} year(s))")
        for timeline, cover in zip(plan.timeline, plan.cover):
            st.markdown(f"- {timeline}" + (f"  \n  ↳ cover crop afterwards: **{cover}**" if cover else ""))
        m, b = plan.metrics, base.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Profit index / yr", f"{m['profit_index']:.1f}", f"{m['profit_index'] - b['profit_index']:+.1f}")
        c2.metric("Irrigation mm / yr", f"{m['irrigation_mm']:.0f}", f"{m['irrigation_mm'] - b['irrigation_mm']:+.0f}",
                  delta_color="inverse")
        c3.metric("Soil health / 100", f"{m['soil_index']:.0f}", f"{m['soil_index'] - b['soil_index']:+.0f}")
        c4.metric("Failed seasons", f"{m['failure_pct']:.0f}%", f"{m['failure_pct'] - b['failure_pct']:+.0f} pp",
                  delta_color="inverse")
        st.caption("Deltas compare with the baseline (keep growing your current crop).")
        for line in plan.why:
            st.markdown(line)
        st.download_button("⬇️ Download this plan (Markdown)", key=f"dl_{idx}", mime="text/markdown",
                           file_name=f"agrishift_plan_{idx + 1}.md",
                           data=plan_to_markdown(ctx["result"], plan, ctx["lat"], ctx["lon"], ctx["soil"],
                                                 ctx["weights"], ctx["irrigation"]))


def tab_plans(result: engine.PlannerResult, ctx: dict) -> None:
    for w in result.warnings:
        st.warning(w)
    if not result.plans:
        return
    st.caption(f"{result.n_candidates:,} rotations of {len(result.baseline.crops)} crops were replayed through "
               f"{len(result.start_years)} historical start years ({result.start_years[0]}-{result.start_years[-1]}) "
               "of NASA climate. Candidate crops: " + ", ".join(result.pool) + ".")
    for i, plan in enumerate(result.plans):
        render_plan(plan, result.baseline, i, ctx)
    rows = [{"Plan": "Baseline", "Crops": " → ".join(result.baseline.crops), "Cycle (yr)": result.baseline.cycle_years,
             **_row(result.baseline)}]
    rows += [{"Plan": " · ".join(p.labels), "Crops": " → ".join(p.crops), "Cycle (yr)": p.cycle_years, **_row(p)}
             for p in result.plans]
    st.subheader("Side by side")
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def _row(p: engine.Plan) -> dict:
    m = p.metrics
    return {"Profit idx/yr": round(m["profit_index"], 2), "Irrigation mm/yr": round(m["irrigation_mm"]),
            "Soil health": round(m["soil_index"]), "Failed seasons %": round(m["failure_pct"]),
            "Score": round(m["score"], 2)}


def tab_time_machine(result: engine.PlannerResult) -> None:
    st.markdown("**Rotation time machine** - the plan is replayed through every real year of the NASA record. "
                "A season counts as *failed* when the modelled yield index drops below "
                f"{engine.FAIL_THRESHOLD:.0%} of potential.")
    if not result.plans:
        st.info("No plan to replay.")
        return
    names = [f"{' · '.join(p.labels)}: {' → '.join(p.crops)}" for p in result.plans]
    if st.session_state.get("tm_pick", 0) not in range(len(names)):
        st.session_state["tm_pick"] = 0                      # plans changed since the last selection
    pick = st.selectbox("Plan to replay", list(range(len(names))), format_func=lambda i: names[i], key="tm_pick")
    plan, base = result.plans[pick], result.baseline
    df = pd.DataFrame({"Selected plan": plan.yield_by_start, f"Baseline ({base.crops[0]})": base.yield_by_start},
                      index=pd.Index(result.start_years, name="Start year"))
    st.line_chart(df)
    st.caption("Mean modelled yield index of all crops in the plan, by the year the rotation would have started.")
    order = plan.yield_by_start.argsort()[:5]
    worst = []
    for j in order:
        failed = [c for c, y in zip(plan.crops, plan.slot_yield[:, j]) if y < engine.FAIL_THRESHOLD]
        worst.append({"Start year": int(result.start_years[j]), "Plan yield index": round(float(plan.yield_by_start[j]), 2),
                      "Baseline yield index": round(float(base.yield_by_start[j]), 2),
                      "Crops that failed": ", ".join(failed) or "none"})
    st.markdown("**Five hardest start years for this plan**")
    st.dataframe(pd.DataFrame(worst), hide_index=True)


def tab_nasa(result: engine.PlannerResult, data: dict) -> None:
    cs = result.climate
    c1, c2, c3, c4 = st.columns(4)
    trend_t, trend_p = cs["t_trend_per_decade"], cs["p_trend_pct_per_decade"]
    c1.metric("Mean temperature", f"{cs['t_mean']:.1f} °C",
              f"{trend_t:+.2f} °C/decade" if trend_t == trend_t else None, delta_color="inverse")
    c2.metric("Annual rainfall", f"{cs['precip']:.0f} mm", f"{trend_p:+.1f} %/decade" if trend_p == trend_p else None)
    c3.metric("Reference ET₀", f"{cs['et0']:.0f} mm")
    c4.metric("Aridity (rain / ET₀)", f"{cs['aridity']:.2f}")
    st.caption(f"{cs['first_year']}-{cs['last_year']} monthly record from NASA POWER (MERRA-2). Trends are simple "
               "linear fits and are indicative only (not significance-tested).")

    monthly = cs["monthly"].copy()
    monthly["Month"] = MONTH_LABELS
    left, right = st.columns(2)
    with left:
        st.markdown("**Typical year: rain vs. crop water demand (mm/month)**")
        st.bar_chart(monthly.set_index("Month")[["Rain (mm)", "ET0 (mm)"]])
    with right:
        st.markdown("**Typical year: temperature (°C)**")
        st.line_chart(monthly.set_index("Month")[["Temperature (°C)"]])
    ann = cs["annual"].set_index("Year")
    left, right = st.columns(2)
    with left:
        st.markdown("**Annual mean temperature**")
        st.line_chart(ann[["Temperature (°C)"]])
    with right:
        st.markdown("**Annual rainfall (mm)**")
        st.line_chart(ann[["Rain (mm)"]])

    st.subheader("🛰️ NASA imagery of your field")
    components.html(mapview.build_map_html(data["lat"], data["lon"]), height=520)

    st.subheader("🌿 Satellite reality check: MODIS vegetation greenness (NDVI)")
    st.caption("Real MODIS Terra 250 m observations at your pixel (ORNL DAAC). Shown for context - it is not used in "
               "scoring. Loads on demand because the service can take 10-30 s.")
    year = dt.date.today().year - 1
    if st.button(f"Load NDVI for {year}", key="load_ndvi"):
        with st.spinner("Requesting MODIS NDVI from ORNL DAAC…"):
            try:
                st.session_state["ndvi"] = {"rows": load_ndvi(data["lat"], data["lon"], year), "year": year,
                                            "error": ""}
            except DataError as exc:
                st.session_state["ndvi"] = {"rows": [], "year": year, "error": str(exc)}
    nd = st.session_state.get("ndvi")
    if nd and nd["error"]:
        st.warning(f"MODIS NDVI unavailable: {nd['error']}")
    elif nd and nd["rows"]:
        frame = pd.DataFrame(nd["rows"], columns=["Date", "NDVI"]).set_index("Date")
        st.line_chart(frame)
        st.caption(f"MOD13Q1 NDVI {nd['year']} at the field pixel: higher values = greener canopy. Compare the green "
                   "season with the sowing and harvest months in your plan.")

    with st.expander("How every crop scored at this location"):
        table = result.crop_table.copy()
        table = table[table.role != "cover"].sort_values("mean_yield_idx", ascending=False)
        table = table.rename(columns={"crop": "Crop", "type": "Type", "best_sow": "Best sowing month",
                                      "cycle_days": "Cycle (days)", "mean_yield_idx": "Yield index (0-1)",
                                      "irrigation_need_mm": "Irrigation for full supply (mm/season)",
                                      "season_failure_pct": "Failed seasons %", "viable": "Viable"})
        table["Yield index (0-1)"] = table["Yield index (0-1)"].round(2)
        table["Irrigation for full supply (mm/season)"] = table["Irrigation for full supply (mm/season)"].round()
        table["Failed seasons %"] = table["Failed seasons %"].round()
        st.dataframe(table[["Crop", "Type", "Best sowing month", "Cycle (days)", "Yield index (0-1)",
                            "Irrigation for full supply (mm/season)", "Failed seasons %", "Viable"]],
                     hide_index=True)


def tab_method() -> None:
    st.subheader("Where every number comes from")
    st.dataframe(lineage_frame(), hide_index=True)
    st.markdown(
        """
**What this tool is.** A transparent decision-support *exploration*, not a yield forecast. There is no black-box
model: every number follows from the equations below and the editable crop table.

**Model in five steps**
1. **Climate** - monthly NASA POWER (MERRA-2) temperature, rain and root-zone wetness for the last 30 full years.
2. **Water** - ET₀ by Hargreaves-Samani (FAO-56 radiation); monthly soil-water bucket with USDA-SCS effective rain;
   yield response to water stress from FAO-33: `1 − Ky·(1 − ETa/ETc)`.
3. **Growth** - radiation × canopy cover × temperature efficiency over a cycle whose length comes from
   thermal time (growing-degree-days); heat, waterlogging, lethal cold and soil pH are penalised.
4. **Rotation** - sowing → harvest → gap → next sowing laid out on a real calendar; legumes give the next crop a
   nitrogen bonus; repeating a botanical family is penalised; cover crops are only used if they fit in a real gap
   and grow rain-fed.
5. **Replay** - every plan is replayed through every historical start year ("time machine"); criteria are
   normalised and weighted by *your* priorities.

**Limitations - please read**
- Climate is a ~50 km model grid (MERRA-2), soil is a 250 m model: neither sees your field's micro-variation.
- Crop parameters are generic reference values (`agrishift/data/crops.csv`); calibrate them locally.
- Indices are relative (0-1). Profit uses margin values *you* enter, not market prices.
- Radiation uses extraterrestrial radiation (no cloud effect); salinity, pests, hail and management are not modelled.
- Perennials and orchards are not part of annual rotations and are not modelled.
- Cover crops are assumed rain-fed and their water use is not charged to the following crop.
- Soil-health is a heuristic index (N-fixation, residue, diversity, cover), not a prediction of measured carbon.
"""
    )


# ----------------------------------------------------------------------------- main
def main() -> None:
    crops_df = get_crops()
    init_state(crops_df)
    run = sidebar(crops_df)
    if run:
        fetch_data()

    st.title("🌾 AgriShift")
    st.markdown("**Adapt your farm to a changing climate: crop rotations explored with NASA data, your soil and your "
                "priorities.**")
    data = st.session_state["data"]
    if data is None:
        st.info("👈 Pick your field in the sidebar, set what matters to you, and press **Fetch NASA data & plan "
                "rotations**.")
        st.markdown("Climate comes from **NASA POWER** (MERRA-2), imagery from **NASA GIBS**, vegetation from "
                    "**MODIS**; soil comes from ISRIC SoilGrids (not NASA).")
        tab_method()
        st.stop()

    moved = (round(float(st.session_state["lat"]), 3), round(float(st.session_state["lon"]), 3)) != (data["lat"], data["lon"])
    if moved:
        st.warning("The sidebar location changed after the last fetch - press the button to refresh the NASA data. "
                   "Results below still show the previous location "
                   f"({data['lat']:.3f}, {data['lon']:.3f}).")

    soil = soil_inputs(data)
    margins = margin_inputs(crops_df)
    weights = {"profit": st.session_state["w_profit"], "water": st.session_state["w_water"],
               "soil": st.session_state["w_soil"], "risk": st.session_state["w_risk"]}
    allowed = list(crops_df.loc[crops_df.type.isin(st.session_state["types"])
                                & crops_df.role.isin(["cash", "both"]), "crop"])
    try:
        result = engine.run_planner(
            data["clim"], soil, crops_df, st.session_state["current_crop"],
            irrigation_cap_mm=float(st.session_state["irrigation_mm"]), rotation_years=int(st.session_state["n_crops"]),
            weights=weights, use_cover=bool(st.session_state["use_cover"]), allowed_crops=allowed,
            margin_overrides=margins)
    except DataError as exc:
        st.error(str(exc))
        st.stop()

    ctx = {"result": result, "lat": data["lat"], "lon": data["lon"], "soil": soil, "weights": weights,
           "irrigation": float(st.session_state["irrigation_mm"])}
    t1, t2, t3, t4 = st.tabs(["🔄 Rotation plans", "⏳ Time machine", "🛰️ NASA evidence", "📋 Method & limits"])
    with t1:
        tab_plans(result, ctx)
    with t2:
        tab_time_machine(result)
    with t3:
        tab_nasa(result, data)
    with t4:
        tab_method()
    st.caption("Data: NASA POWER & MERRA-2 (GMAO), NASA GIBS, MODIS MOD13Q1 (ORNL DAAC), GPM IMERG, ISRIC SoilGrids "
               "2.0 (CC BY 4.0). Built for the NASA Space Apps Challenge 2026.")


main()
