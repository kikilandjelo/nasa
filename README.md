# 🌾 AgriShift - crop rotations explored with NASA data

**NASA Space Apps Challenge 2026 · Challenge: _Field Shift - Adapting Farms with NASA Data_**

> Seasons are shifting: rain arrives late, heat arrives early. AgriShift replays candidate crop rotations through
> **30 years of real NASA climate records**, combines them with your soil, your water and what *you* care about
> (water, soil health, profit, stability), and shows which sequences of crops would have survived, and *why*.

<!-- TODO before submission: demo link, 2-3 screenshots in docs/, 2-minute video link -->

## What it does

1. **Pick a field** (GPS, Google Maps link, or `lat, lon`) and tell AgriShift what you grow now, how much water you can add, and your priorities.
2. **NASA data** - 30 years of monthly NASA POWER (MERRA-2) temperature, rain and root-zone wetness for that point.
   Topsoil (pH, organic matter, clay, sand) comes from ISRIC SoilGrids (**not NASA**) and is editable with your own soil test.
3. **A transparent crop model** simulates 36 crops: thermal-time phenology, a monthly soil-water balance (FAO-56/FAO-33),
   radiation x canopy x temperature growth, plus heat, lethal-cold, waterlogging and soil-pH stress.
4. **Rotation search** - every 2-4 crop sequence is laid out on a real calendar (sow -> harvest -> gap -> next sowing).
   Legumes give the next crop a nitrogen bonus, repeating a botanical family is penalised, and cover crops are used
   only where they genuinely fit in the gap.
5. **Time machine** - each plan is replayed through every historical start year to count *failed seasons*.
6. **Your priorities** (water, soil, profit, stability) weight the ranking; profit uses an index **you** set, not invented prices.
7. **Evidence** - NASA GIBS imagery (MODIS NDVI / land-surface temperature, GPM IMERG rain), an on-demand MODIS NDVI curve at the
   field pixel, a full data-lineage table, and a downloadable plan.

No API keys, accounts or secrets are needed.

## Data lineage - where every number comes from

| Variable | Dataset / product | Resolution | Used for | Status |
|---|---|---|---|---|
| Air temperature (mean / max / min, 2 m) | NASA POWER monthly API: T2M, T2M_MAX, T2M_MIN (MERRA-2 reanalysis, NASA GMAO) | ~0.5° x 0.625° | Growth efficiency, heat and cold stress, thermal time (cycle length) | Live NASA data (modelled reanalysis) |
| Precipitation | NASA POWER: PRECTOTCORR (MERRA-2 precipitation with NASA POWER bias correction) | ~0.5° x 0.625° | Effective rain in the soil-water balance | Live NASA data (modelled reanalysis) |
| Root-zone soil wetness | NASA POWER: GWETROOT (MERRA-2 land model - this is NOT SMAP) | ~0.5° x 0.625° | Soil water available at sowing | Live NASA data (modelled) |
| Reference evapotranspiration (ET0) | Hargreaves-Samani equation on POWER temperatures + latitude (FAO-56 radiation) | point | Crop water demand (ET0 x Kc) | Derived by equation |
| Vegetation greenness (NDVI) | MODIS Terra MOD13Q1 via ORNL DAAC point service | 250 m, 16-day | Satellite reality check of the growing season (not used in scoring) | Live NASA data, on demand |
| Map imagery | NASA GIBS: MODIS NDVI, MODIS land surface temperature, GPM IMERG precipitation, MODIS true colour | 250 m - 10 km | Visual context only | Live NASA imagery |
| Soil pH, organic carbon, clay, sand | ISRIC SoilGrids 2.0, 0-30 cm weighted mean | 250 m (model) | Crop pH suitability, water-holding capacity, soil-health baseline | Live, NOT NASA - editable with your soil test |
| Crop characteristics | agrishift/data/crops.csv - approximate FAO-style reference values (EcoCrop, FAO-33, FAO-56) | - | Temperature, cold, water and pH limits; nitrogen fixation; residue | Assumption - verify with local agronomists |
| Profitability | Relative margin index you can edit in the app | - | Profit criterion | User assumption - not market data |

What is deliberately **not** claimed: SMAP, ECOSTRESS, Landsat/HLS and GRACE are not used (see Roadmap). Soil wetness is a MERRA-2 model
output, not a SMAP measurement.

## Method in brief

- **Water**: `ET0` by Hargreaves-Samani (FAO-56 radiation) on POWER temperatures; USDA-SCS effective rain; monthly soil bucket;
  relative yield `1 - Ky * (1 - ETa/ETc)` (FAO-33).
- **Growth**: sum over the cycle of `radiation x canopy cover x temperature efficiency`, compared with a full-length cycle in the
  sunniest part of the year. Cycle length comes from growing-degree-days, so warm years shorten and cool years lengthen seasons.
- **Rotation effects**: legume carry-over +7 % to the next non-legume; same-family repeat x0.88; soil-health index from nitrogen input,
  residue return, family diversity and off-season cover; metrics are per calendar year of the repeating cycle.
- **Risk**: a crop-season is *failed* when its modelled yield index is below 40 % of potential; plans are scored on the share of failed seasons across all replayed years.
- Every constant lives at the top of `agrishift/engine.py`; every crop parameter is in `agrishift/data/crops.csv`.

## Honest limitations

- Climate is a ~50 km model grid (MERRA-2) and soil a 250 m model: neither sees field-level variation.
- Crop parameters are generic FAO-style reference values, **not site-calibrated**. Indices are relative (0-1), not tonnes per hectare.
- Radiation uses extraterrestrial radiation (no cloud effect). Salinity, pests, hail and management are not modelled.
- Perennials and orchards are outside annual rotations. Cover-crop water use is not charged to the next crop.
- The profit criterion is only as good as the margin values you enter.

## Validation status

- **Done**: unit tests covering the FAO-56 worked example, parsers against fixtures that follow the documented API formats, agronomic
  invariants (irrigation never lowers yield, rotation effects exact in a noise-free climate, calendar-chain rules, southern-hemisphere
  mirror), and fuzz tests on random climates; plausibility checks on temperate, semi-arid, monsoon and hyper-arid synthetic climates.
- **Not yet done**: comparison with measured yields. Planned: correlate the modelled inter-annual yield index with regional
  yield statistics for a pilot region and calibrate crop parameters with local agronomists.

## Run locally

```bash
git clone https://github.com/kikilandjelo/nasa.git
cd nasa
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Python 3.9+. Deploys unchanged to Streamlit Community Cloud (main file: `app.py`).

## Tests

```bash
python -m unittest discover -s tests -v      # or: pytest
```

## Project structure

```
app.py                  Streamlit interface only
agrishift/agro.py       FAO-56 / FAO-33 equations (pure functions)
agrishift/crops.py      loads and validates data/crops.csv
agrishift/engine.py     crop model, calendar chaining, rotation search, scoring, explanations
agrishift/sources.py    NASA POWER, SoilGrids, MODIS NDVI clients (parsing separated from network)
agrishift/geo.py        Google Maps link / coordinate parsing
agrishift/mapview.py    Leaflet + NASA GIBS map builder
agrishift/lineage.py    data-lineage table
agrishift/report.py     Markdown plan export
tests/                  unit, property and fuzz tests
```

## Roadmap

1. NASA CERES surface radiation via POWER (`ALLSKY_SFC_SW_DWN`) instead of extraterrestrial radiation (cloud effect).
2. SMAP L4 root-zone soil moisture and HLS / Landsat field-scale NDVI (NASA Earthdata) for local validation.
3. Replay rotations through **future** climate (NASA NEX-GDDP-CMIP6) in addition to the past 30 years.
4. Calibrate and validate against regional yield statistics; salinity module.
5. Azerbaijani / Russian / English interface and a lightweight SMS or messaging summary for low-connectivity farms.

## Team

<!-- TODO: team name, members, roles, contact -->

## AI tool disclosure

<!-- TEMPLATE - edit so it is TRUE for your project. The Space Apps FAQ asks teams to describe how AI tools were used.
Example: "Claude (Anthropic) was used for code review, architecture suggestions and drafting parts of the code and tests.
The team selected the datasets, reviewed the model assumptions and ran all tests." -->

## Project history / pre-existing work

<!-- TODO: state what existed before the hackathon (see CHANGELOG.md for the 1.x -> 2.0 differences) and how it was
disclosed to your Local Lead, as required by your event's rules. -->

## Credits and data licences

- NASA POWER Project (NASA Langley Research Center) and MERRA-2 (NASA GMAO; Gelaro et al., 2017, *J. Climate* 30, 5419-5454).
- MODIS MOD13Q1 (Didan, 2021, doi:10.5067/MODIS/MOD13Q1.061) via the ORNL DAAC MODIS Web Service; NASA GIBS / Worldview; GPM IMERG.
- ISRIC SoilGrids 2.0 - Poggio et al. (2021), *SOIL* 7, 217-240 - CC BY 4.0.
- Allen et al. (1998) FAO-56; Doorenbos & Kassam (1979) FAO-33; Hargreaves & Samani (1985); USDA-SCS effective rainfall; FAO EcoCrop.
- Map: Leaflet; (c) OpenStreetMap contributors (c) CARTO.
