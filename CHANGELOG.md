# Changelog

## 2.0.0 - rewrite around a transparent rotation model

**Removed (v1 prototype)**
- Random Forest trained on synthetic data whose labels came from the same hand-written formula as the features
  (it only re-learned that formula). Replaced by an explicit, documented agronomic model.
- "NDVI" computed from soil wetness and temperature, and UI labels naming GPM / SMAP / ECOSTRESS / Landsat that did not
  match the data actually fetched (NASA POWER / MERRA-2). Labels now state the real dataset; NDVI is real MODIS data on demand.
- Static rotation text ("Drought-tolerant Legume ➔ Deep-root Grain") that ignored location. Replaced by a rotation engine.
- Synthetic "regional scan" heat-map (sine/cosine noise). Replaced by real NASA GIBS imagery layers.
- Unused `earthaccess` call, `scikit-learn` and `pydeck` dependencies; hard-coded coordinate shortcut; hard-coded
  ocean boxes that mislabelled Hawaii, north-east Brazil, Senegal and below-sea-level farmland (e.g. Caspian lowlands) as water.

**Added**
- 30-year monthly NASA POWER record; ISRIC SoilGrids soil with farmer override; MODIS NDVI reality check.
- Thermal-time crop model with FAO-56/33 water balance, heat/cold/waterlogging/pH stress; 36-crop editable table.
- Calendar-aware rotation search, off-season cover crops, "time machine" replay over historical years,
  priority weights, editable profit index, Markdown export, data-lineage table, unit tests.

**Fixed**
- Results no longer vanish when a widget changes; location widgets no longer get overwritten by a pasted link on every rerun;
  invalid links no longer silently reset the location; network failures are shown instead of swallowed.
