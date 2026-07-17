# Wildfire Perimeter Growth Prediction — Project Notes

## What this project is
`Wildfire_Prediction_RS_ML.ipynb` is an end-to-end geospatial ML pipeline that predicts
2021 Dixie Fire spread. It downloads NIFC fire perimeters, NASA HLS Landsat NDVI, USGS
3DEP terrain data, and OpenStreetMap roads, builds a 30-60m analysis grid, labels cells
burned/unburned from consecutive perimeter snapshots, engineers raster features via
zonal statistics, and trains a spatially cross-validated Random Forest. Outputs are 4
PNGs + one Folium HTML map in `./outputs/`.

`Wildfire_Prediction_RS_ML.ipynb.bak` is the pre-fix backup (kept for reference/rollback).

## Environment
- Conda env: `wildfire-p03` (`/opt/anaconda3/envs/wildfire-p03`), Python 3.11
- Key pinned-old APIs: GeoPandas 0.14.x (**no `.union_all()`** — use `.unary_union`,
  a property not a method), rasterio 1.3.x, shapely 2.0.x
- Data lives in `./data/landsat_data/` and `./data/dem_data/`; outputs in `./outputs/`
- **No NASA Earthdata credentials are cached in an automated/headless environment**
  (no `~/.netrc`). `earthaccess.login(strategy='interactive')` in cell 16 requires a
  live terminal and will fail fast (`StdinNotImplementedError`) when run headlessly via
  `nbconvert --execute`. As long as `./data/landsat_data/prefire_ndvi.tif` and the DEM
  files are already cached, the rest of the pipeline runs fine without auth. A fresh
  NDVI mosaic download requires the user to run cell 16 interactively themselves first.

## Known-slow cells / possible perf bug (not yet root-caused)
- `create_analysis_grid` / `create_burned_labels` (grid + labeling): notebook's own
  estimate is ~45-60 min at full scale (~1-2M grid cells). No spatial index is used for
  the `.within()` predicate.
- `extract_raster_features` / `predict_full_grid`: zonal_stats over ~1M and ~2M cells
  respectively — also ~45-60 min per the notebook's own estimate.
- **A headless `nbconvert --execute` test run (2026-07-15) ran for 8.5+ hours at
  ~100% CPU with zero new output after the terrain-features plot, and was killed as
  presumed-stuck** (PIDs 11280/11303, `kill -TERM`). That's well beyond the notebook's
  own combined estimate (~4h for all remaining steps), so this is likely a genuine
  performance issue — not just "slow" — in whichever cell runs right after
  `03_terrain_features.png` is saved (i.e. `create_analysis_grid` or
  `create_burned_labels`), most plausibly the non-indexed `.within()` calls over
  ~1-2M-row GeoDataFrames. **Not yet root-caused or fixed** — if you hit a
  similarly long hang in the same spot, that's the first place to look (e.g. profile
  whether `grid.geometry.within(newly_burned_zone)` is the bottleneck, and consider a
  spatial-index-backed approach, e.g. `sjoin` or building an `STRtree`, if so).
- When testing changes, always run `nbconvert` with `cwd` set to the **project
  directory** (not a scratch copy elsewhere) so relative `./data` / `./outputs` paths
  resolve to the existing caches — otherwise it silently redownloads/recomputes
  everything from scratch.

## Fix session — 2026-07-14/15

The notebook was previously erroring out around cell 50 (`model_df` empty →
cascading `NameError`s downstream). Root cause and fixes below, by cell (indices as of
the fixed notebook, after 5 scratch cells were deleted):

**Already correct before this session (verified, no change needed):**
- No `.union_all()` anywhere — `.unary_union` used consistently (perimeter, roads,
  burned-area checks).
- `download_nifc_perimeter` already keeps only the single largest record by acreage
  (`_keep_largest`), preventing multi-fire/whole-US API responses from blowing up the
  grid.
- `create_analysis_grid` already has a 300km bounds sanity-clip + a
  `MAX_CELLS = 5_000_000` auto-scale guard.
- `compute_slope_aspect` already fills nodata before `np.gradient()` and restores it
  after, avoiding NaN propagation across whole rows/columns.

**Fixed this session:**
- **Cell 4**: added `import re` and `from rioxarray.merge import merge_arrays`.
- **Cell 10** (perimeter download): added sanity assert on perimeter area
  (2,000-8,000 km²) so a wrong API response fails fast.
- **Cell 12** (snapshots): added `assert len(snapshots) >= 2`.
- **Cell 18** (`search_landsat_scenes`): added a `count` parameter (default 20) so it
  can be reused for a wider mosaic search; added a non-empty assert.
- **Cell 20** (`download_and_compute_ndvi`) — **the core fix**: a single HLS granule
  only covers one MGRS tile (~20% of the study grid). Rewrote to search ALL HLS
  granules intersecting `BBOX_WGS84` (`count=50`), keep one scene per unique MGRS tile,
  compute NDVI per tile, and mosaic with `rioxarray.merge_arrays`. NDVI nodata is now
  written as `-9999` **at the source** (tag NaN before reprojecting so GDAL's warp
  handles tile edges correctly, then swap to `-9999` after — this was also the fix for
  the `OverflowError` from HLS's raw `3.402823466e+38` fill value). Falls back to the
  original single-scene approach if the mosaic path throws. Cache check now reports
  valid-pixel coverage % and warns if a cached file looks like a stale single-tile
  version.
- **Cell 22** (`plot_ndvi`): masks `-9999` to NaN before `imshow` (previously nodata
  pixels rendered as a solid clipped color, not blank).
- **Cells 26 / 28** (DEM / slope-aspect): added existence asserts after each call.
- **Cells 33 / 35** (grid / labels): added non-empty asserts.
- **Cell 41**: added explicit per-column null-count print before `.describe()`.
- **Cells 44-49 → consolidated into 2 cells**: deleted 5 leftover ad-hoc debugging
  cells from earlier troubleshooting (raster-bounds dumps, a manual NDVI re-save, a
  duplicate zonal-stats re-run). Replaced with:
  1. One diagnostic cell: per-raster valid-pixel % + % of `labeled_grid` centroids
     inside each raster's bbox, plus an assert that all 4 rasters share the `-9999`
     nodata sentinel.
  2. A rewritten "prepare model_df" cell that **median-imputes** missing feature
     values per column instead of `dropna()` (which previously zeroed out `model_df`
     entirely whenever NDVI coverage was incomplete — this was the actual crash).
     Falls back to a raster's own global median if a column has zero grid overlap;
     raises a clear error if the raster itself has zero valid pixels. Stores
     `FEATURE_MEDIANS` for reuse at prediction time. Ends with non-empty/no-NaN
     asserts.
- **Cell 52** (`predict_full_grid`): was filling missing features with a flat `0`
  (wildly out-of-distribution for elevation ~1691m mean). Now reuses the training-time
  `FEATURE_MEDIANS`.
- **Cell 58** (results summary): `mean_auc` was referenced but never defined anywhere
  (`run_spatial_cv` returns a DataFrame, not that scalar) — added
  `mean_auc = cv_results['AUC-ROC'].mean()`.

## Verification status
- Static: all 60 cells pass `ast.parse`; zero `union_all(` occurrences; all 4 rasters
  (NDVI/DEM/slope/aspect) confirmed to use a consistent `-9999` nodata sentinel;
  all-NaN `FEATURE_COLS` case now explicitly handled (raster-median fallback or hard
  error) instead of silently producing NaN.
- Live, confirmed working: NIFC perimeter download + synthetic snapshot generation
  (cells 0-15) against the real NIFC API; USGS 3DEP DEM download + slope/aspect
  computation from scratch (cells 24-30); grid creation started successfully.
- **NOT confirmed end-to-end.** The headless verification attempt did not finish (see
  "Known-slow cells / possible perf bug" above) and was killed after 8.5+ hours with
  no result. Everything from `create_analysis_grid`/`create_burned_labels` onward
  (grid → labels → zonal-stats features → median-impute → spatial CV → AUC →
  full-grid prediction → Folium map → summary) is still **unverified against a live
  run**. Static checks (syntax, no `union_all`, consistent nodata, no all-NaN feature
  column) all pass, and the logic was reasoned through carefully, but that is not the
  same as having watched it complete.

## To fully verify / finish
1. **Run the notebook yourself, interactively, cell by cell, in your IDE** — this is
   now the recommended path rather than another headless background run. Reasons:
   you can authenticate at cell 16 with real NASA Earthdata credentials (enabling the
   real NDVI mosaic fix, which no headless run here can ever exercise), you'll see
   live progress printouts instead of a silent black box, and if a cell runs
   unexpectedly long you can inspect *which* one and interrupt it, rather than
   discovering 8+ hours later that something was stuck.
2. Watch specifically for whether `create_analysis_grid` / `create_burned_labels`
   complete in a reasonable time (tens of minutes, not hours) — if either hangs, see
   the perf-bug note above.
3. Confirm the full run has no cell errors, `model_df` non-empty, spatial CV reports
   an AUC, and all 5 files exist in `./outputs/` (4 PNGs + the Folium HTML map).
4. To get REAL full-coverage NDVI (not the current single-tile cache), delete
   `./data/landsat_data/prefire_ndvi.tif` first so cell 20's new mosaic path actually
   runs instead of hitting the cache.
