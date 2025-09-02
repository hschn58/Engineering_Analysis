import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import rasterio
from pyproj import Transformer
from io import BytesIO
from PIL import Image

# User knob
GRID_SIZE = 400  # 'MAX' | int (max side) | (rows, cols)

# --- DEM download (no key in code; use env var) ---
api_key = os.environ.get("OPENTOPO_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENTOPO_API_KEY in your environment (export OPENTOPO_API_KEY=...)")

url = (
    "https://portal.opentopography.org/API/globaldem"
    f"?demtype=SRTMGL1&south=47.6&north=48.2&west=-114.5&east=-113.8"
    f"&outputFormat=GTiff&API_Key={api_key}"
)
if not os.path.exists("flathead_dem.tif"):
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open("flathead_dem.tif", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# --- Load DEM & project to UTM ---
with rasterio.open("flathead_dem.tif") as src:
    dem = src.read(1).astype(float)
    h, w = dem.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    lon, lat = rasterio.transform.xy(src.transform, rows, cols)

transformer = Transformer.from_crs("epsg:4326", "epsg:26911", always_xy=True)
x_flat, y_flat = transformer.transform(np.asarray(lon).ravel(), np.asarray(lat).ravel())
x = x_flat.reshape(h, w)
y = y_flat.reshape(h, w)

# --- Optional: crop to square footprint ---
dx, dy = x.max() - x.min(), y.max() - y.min()
side = min(dx, dy)
if dx > dy:
    cx = (x.max() + x.min()) / 2
    xmin, xmax = cx - side / 2, cx + side / 2
    cmin = np.argmin(np.abs(x[0, :] - xmin))
    cmax = np.argmin(np.abs(x[0, :] - xmax)) + 1
    x, y, dem = x[:, cmin:cmax], y[:, cmin:cmax], dem[:, cmin:cmax]
elif dy > dx:
    cy = (y.max() + y.min()) / 2
    ymin, ymax = cy - side / 2, cy + side / 2
    rmin = np.argmin(np.abs(y[:, 0] - ymax))
    rmax = np.argmin(np.abs(y[:, 0] - ymin)) + 1
    x, y, dem = x[rmin:rmax, :], y[rmin:rmax, :], dem[rmin:rmax, :]


# --- After you've produced x, y, and dem (and before resampling) ---

import numpy as np
from rasterio.fill import fillnodata
try:
    from scipy.ndimage import median_filter   # optional but nice to have
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def deband_vertical(z, win=9):
    """
    Remove column-wise banding by subtracting a smoothed column baseline.
    Uses nanmedian per-column, then a moving-average to estimate baseline.
    """
    z = z.astype(np.float32, copy=False)
    col_med = np.nanmedian(z, axis=0)                         # shape (W,)
    k = np.ones(win, dtype=np.float32) / float(win)
    # pad edges to avoid shrinking at borders
    pad = win // 2
    padded = np.pad(col_med, (pad, pad), mode='edge')
    smooth = np.convolve(padded, k, mode='valid')             # shape (W,)
    bias = (col_med - smooth)[None, :]                        # shape (1, W) for broadcast
    return z - bias

def fetch_naip_texture_single(x, y, shrink_bbox=1.0):
    """Fetch NAIP imagery in one request to avoid north–south seam artifacts."""
    ny, nx = x.shape
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    xmin += shrink_bbox
    ymin += shrink_bbox
    xmax -= shrink_bbox
    ymax -= shrink_bbox
    tile = _export_arcgis_png_rgba(NAIP_URL, xmin, ymin, xmax, ymax, nx, ny)
    return tile



def _resample_grid(x, y, z, grid_size):
    """Return (x2,y2,z2) resampled to grid_size."""
    ny, nx = x.shape
    if grid_size == "MAX":
        return x, y, z
    if isinstance(grid_size, int):
        if grid_size >= max(ny, nx):
            return x, y, z
        ty = max(2, int(np.ceil(ny * grid_size / max(ny, nx))))
        tx = max(2, int(np.ceil(nx * grid_size / max(ny, nx))))
    elif isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
        ty = max(2, min(int(grid_size[0]), ny))
        tx = max(2, min(int(grid_size[1]), nx))
    else:
        raise ValueError("GRID_SIZE must be 'MAX', an int, or (rows, cols).")

    def idx(n_old, n_new):
        if n_new >= n_old:
            return np.arange(n_old)
        step = (n_old - 1) / (n_new - 1)
        return np.array([int(round(i * step)) for i in range(n_new)])

    ri = idx(ny, ty)
    ci = idx(nx, tx)
    return x[np.ix_(ri, ci)], y[np.ix_(ri, ci)], z[np.ix_(ri, ci)]


# 1) Mask/fill nodata/voids
nodata_val = None
with rasterio.open("flathead_dem.tif") as _src:
    nodata_val = _src.nodata

dem = dem.astype(np.float32, copy=False)
if nodata_val is not None:
    dem[dem == nodata_val] = np.nan
# SRTM voids sometimes appear as extreme negatives
dem[dem < -10000] = np.nan

# Fill small NaN gaps using rasterio's in-painter
mask = ~np.isnan(dem)
dem_filled = fillnodata(dem, mask=mask, max_search_distance=10, smoothing_iterations=0)

# 2) Column bias removal (north–south banding fix)
dem_deband = deband_vertical(dem_filled, win=9)  # small window keeps real relief

# (Optional) Feather seams with a tiny horizontal median — avoids over-smoothing
if _HAVE_SCIPY:
    dem_deband = median_filter(dem_deband, size=(1, 3))  # 3 px across columns only

# Continue with resampling
x, y, z = _resample_grid(x, y, dem_deband, GRID_SIZE)

# --- NAIP imagery for texture ---
NAIP_URL = (
    "https://gisservicemt.gov/arcgis/rest/services/"
    "MSDI_Framework/NAIP_2023/ImageServer/exportImage"
)


def _export_arcgis_png_rgba(url, xmin, ymin, xmax, ymax, width, height,
                            sr="26911", band_ids="0,1,2"):
    mosaic_rule = '{"mosaicMethod":"esriMosaicBlend","where":"1=1"}'
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": sr,
        "imageSR": sr,
        "size": f"{width},{height}",
        "format": "png32",
        "transparent": "true",
        "f": "image",
        "bandIds": band_ids,
        "interpolation": "RSP_Bilinear",
        "mosaicRule": mosaic_rule,  # << add this
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    im = Image.open(BytesIO(r.content)).convert("RGBA")
    return np.asarray(im, dtype=np.float32) / 255.0


def fetch_naip_texture_for_grid(x, y, tile_max=2048, shrink_bbox=1.0):
    ny, nx = x.shape
    out = np.zeros((ny, nx, 4), dtype=np.float32)
    flip_ud = (y[0, 0] < y[-1, 0])  # if y increases downward
    flip_lr = (x[0, 0] > x[0, -1])  # if x decreases to the right
    for r0 in range(0, ny, tile_max):
        r1 = min(r0 + tile_max, ny)
        for c0 in range(0, nx, tile_max):
            c1 = min(c0 + tile_max, nx)
            xs = [x[r0, c0], x[r0, c1 - 1], x[r1 - 1, c0], x[r1 - 1, c1 - 1]]
            ys = [y[r0, c0], y[r0, c1 - 1], y[r1 - 1, c0], y[r1 - 1, c1 - 1]]
            xmin, xmax = float(min(xs)), float(max(xs))
            ymin, ymax = float(min(ys)), float(max(ys))
            xmin += shrink_bbox
            ymin += shrink_bbox
            xmax -= shrink_bbox
            ymax -= shrink_bbox
            w, h = int(c1 - c0), int(r1 - r0)
            tile = _export_arcgis_png_rgba(NAIP_URL, xmin, ymin, xmax, ymax, w, h)
            if flip_ud:
                tile = np.flipud(tile)
            if flip_lr:
                tile = np.fliplr(tile)
            out[r0:r1, c0:c1, :] = tile
    return out



import numpy as np
from scipy.ndimage import gaussian_filter

def estimate_water_mask(rgba):
    """Cheap RGB water detector: strong blue, low red/green."""
    rgb = rgba[..., :3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (b > 0.25) & (b > g + 0.02) & (b > r + 0.05)
    # feather edges slightly so corrections fade into shorelines
    return gaussian_filter(mask.astype(float), sigma=1.0) > 0.3

def deband_texture_columns(rgba, mask, win=17, strength=1.0):
    """
    Remove column-wise radiometric striping from imagery (RGB only),
    using per-column means computed over water pixels.
    """
    out = rgba.copy()
    ny, nx, _ = out.shape
    k = np.ones(win, dtype=np.float32) / float(win)
    pad = win // 2

    # avoid all-NaN warnings
    safe_mean = lambda v: np.nan if np.asarray(v).size == 0 else np.nanmean(v)


    for ch in range(3):  # RGB
        # per-column mean over water
        col_mean = np.empty(nx, dtype=np.float32)
        for j in range(nx):
            vals = out[:, j, ch]
            m = mask[:, j]
            v = vals[m]
            col_mean[j] = safe_mean(v if v.size else np.nan)

        # fill NaNs (thin columns without water) by edge-hold
        if np.isnan(col_mean).any():
            # linear fill then edge pad
            idx = np.arange(nx)
            good = ~np.isnan(col_mean)
            col_mean = np.interp(idx, idx[good], col_mean[good])

        smooth = np.convolve(np.pad(col_mean, (pad, pad), mode='edge'), k, mode='valid')
        bias = (col_mean - smooth)[None, :]  # shape (1, nx)

        # subtract bias only on water; feather via a soft mask
        # build a soft column field by repeating bias down rows
        bias_img = np.repeat(bias, ny, axis=0)
        out[..., ch] -= strength * bias_img * mask

    out[..., :3] = np.clip(out[..., :3], 0.0, 1.0)
    return out


rgba = fetch_naip_texture_single(x, y)  # your single-tile fetch

water_mask = estimate_water_mask(rgba)
rgba = deband_texture_columns(rgba, water_mask, win=17, strength=1.0)

rgba_faces = rgba[:-1, :-1, :]

# --- Plot ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(
    x, y, z,
    facecolors=rgba[:-1, :-1, :],
    rstride=1, cstride=1,
    edgecolor="none", linewidth=0, antialiased=False, shade=False
)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_zlabel("Elevation (m)")
ax.set_title("Flathead Range — Topography (no vertical exaggeration)")

# <<< key fix: make axis units isotropic (no vertical scaling) >>>
ax.set_box_aspect((
    float(x.max() - x.min()),
    float(y.max() - y.min()),
    float(z.max() - z.min())
))

plt.show()