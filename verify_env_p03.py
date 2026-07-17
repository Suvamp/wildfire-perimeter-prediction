"""
verify_env_p03.py
─────────────────
Run this script immediately after activating the wildfire-p03 environment
to confirm every library installed correctly and at the expected version.

Usage:
    conda activate wildfire-p03
    python verify_env_p03.py
"""

import sys
import importlib

# ── (package_name_to_import, pip/conda name, min_version) ─────────────────
REQUIRED = [
    # Core scientific
    ("numpy",           "numpy",        "1.24"),
    ("pandas",          "pandas",       "2.0"),
    ("scipy",           "scipy",        "1.11"),
    ("matplotlib",      "matplotlib",   "3.7"),
    ("sklearn",         "scikit-learn", "1.3"),

    # Geospatial core
    ("rasterio",        "rasterio",     "1.3.9"),
    ("geopandas",       "geopandas",    "0.14"),
    ("shapely",         "shapely",      "2.0"),
    ("pyproj",          "pyproj",       "3.6"),
    ("fiona",           "fiona",        "1.9"),

    # Raster extended
    ("rioxarray",       "rioxarray",    "0.15"),
    ("xarray",          "xarray",       "2023.6"),
    ("rasterstats",     "rasterstats",  "0.19"),
    ("contextily",      "contextily",   "1.4"),

    # Network / OSM
    ("osmnx",           "osmnx",        "1.9"),
    ("networkx",        "networkx",     "3.1"),

    # Spatial stats
    ("libpysal",        "libpysal",     "4.9"),
    ("esda",            "esda",         "2.5"),

    # Web mapping & elevation
    ("folium",          "folium",       "0.15"),
    ("py3dep",          "py3dep",       "0.16"),

    # Utilities
    ("requests",        "requests",     "2.31"),
    ("tqdm",            "tqdm",         "4.66"),
    ("earthaccess",     "earthaccess",  "0.9"),
]

# ── Version comparison helper ──────────────────────────────────────────────
def version_ok(installed: str, minimum: str) -> bool:
    """Returns True if installed version >= minimum."""
    try:
        from packaging.version import Version
        return Version(installed) >= Version(minimum)
    except ImportError:
        # Fallback: simple string split comparison
        iv = tuple(int(x) for x in installed.split(".")[:3] if x.isdigit())
        mv = tuple(int(x) for x in minimum.split(".")[:3] if x.isdigit())
        return iv >= mv

# ── Run checks ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Environment Verification — GIS Portfolio Project 03")
print(f"  Python {sys.version.split()[0]}  |  {sys.executable}")
print(f"{'='*60}\n")

all_ok = True
col_w = 14

for import_name, pkg_name, min_ver in REQUIRED:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "unknown")
        ok = version_ok(ver, min_ver) if ver != "unknown" else True
        status = "✅" if ok else "⚠️ "
        if not ok:
            all_ok = False
        print(f"  {status}  {import_name:<20s}  {ver:<12s}  (min {min_ver})")
    except ImportError:
        print(f"  ❌  {import_name:<20s}  NOT INSTALLED  (min {min_ver})")
        all_ok = False

# ── GDAL (accessed via rasterio) ──────────────────────────────────────────
print()
try:
    import rasterio
    gdal_ver = rasterio.__gdal_version__
    print(f"  ✅  GDAL (via rasterio)      {gdal_ver}")
except Exception:
    print("  ⚠️   GDAL version could not be determined")

# ── CRS sanity check ──────────────────────────────────────────────────────
print()
print("  Running CRS sanity check (EPSG:5070)...")
try:
    import pyproj
    crs = pyproj.CRS("EPSG:5070")
    print(f"  ✅  EPSG:5070 resolved: {crs.name}")
except Exception as e:
    print(f"  ❌  CRS check failed: {e}")
    all_ok = False

# ── Rasterio read/write smoke test ────────────────────────────────────────
print()
print("  Running rasterio smoke test...")
try:
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.io import MemoryFile

    data = np.random.rand(1, 4, 4).astype("float32")
    transform = from_bounds(0, 0, 1, 1, 4, 4)
    with MemoryFile() as memfile:
        with memfile.open(driver="GTiff", height=4, width=4,
                          count=1, dtype="float32",
                          crs="EPSG:4326", transform=transform) as ds:
            ds.write(data)
        with memfile.open() as ds:
            result = ds.read(1)
    assert result.shape == (4, 4), "Shape mismatch"
    print("  ✅  rasterio in-memory read/write: OK")
except Exception as e:
    print(f"  ❌  rasterio smoke test failed: {e}")
    all_ok = False

# ── GeoPandas smoke test ──────────────────────────────────────────────────
print()
print("  Running geopandas smoke test...")
try:
    import geopandas as gpd
    from shapely.geometry import Point
    gdf = gpd.GeoDataFrame(
        {"val": [1, 2]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326"
    )
    gdf_proj = gdf.to_crs("EPSG:5070")
    print(f"  ✅  GeoDataFrame reprojected to EPSG:5070: OK ({len(gdf_proj)} rows)")
except Exception as e:
    print(f"  ❌  geopandas smoke test failed: {e}")
    all_ok = False

# ── Spatial autocorrelation smoke test ───────────────────────────────────
print()
print("  Running esda/libpysal smoke test (Moran's I)...")
try:
    import numpy as np
    from libpysal.weights import lat2W
    from esda.moran import Moran
    w = lat2W(5, 5)
    y = np.random.rand(25)
    mi = Moran(y, w)
    print(f"  ✅  Moran's I computed: I={mi.I:.4f} (random data, any value OK)")
except Exception as e:
    print(f"  ❌  esda smoke test failed: {e}")
    all_ok = False

# ── Final summary ─────────────────────────────────────────────────────────
print()
print("=" * 60)
if all_ok:
    print("  🎉  All checks passed! Environment is ready for Project 03.")
    print()
    print("  Next step: open wildfire_risk_analysis.ipynb and run Step 0.")
else:
    print("  ⚠️  Some checks failed. Fix issues above before proceeding.")
    print()
    print("  Common fixes:")
    print("    conda install -c conda-forge <package_name>")
    print("    pip install earthaccess  (pip-only packages)")
print("=" * 60)
print()
