# Topography

Downloads SRTM elevation data, fetches NAIP aerial imagery, and renders a textured 3D surface plot of the Flathead Range in Montana.

## Pipeline

1. **DEM download** -- Fetches SRTM GL1 (30 m) elevation data from the OpenTopography API for the Flathead Range area (47.6-48.2 N, 113.8-114.5 W)
2. **Reprojection** -- Transforms from WGS84 (lat/lon) to UTM Zone 11N (EPSG:26911) for metric coordinates
3. **Preprocessing** -- Fills nodata voids, removes vertical column banding artifacts, optional median filtering
4. **Resampling** -- Downsamples to a configurable grid size (default 400) for manageable rendering
5. **NAIP texture** -- Fetches 2023 aerial imagery from Montana's ArcGIS image service, with automatic tiling for large grids and water-mask-based column destriping
6. **Rendering** -- 3D surface plot with satellite texture overlay and isotropic axis scaling (no vertical exaggeration)

## Usage

```bash
export OPENTOPO_API_KEY=your_key_here
python Topography.py
```

An API key from [OpenTopography](https://opentopography.org/) is required (free with registration).

## Dependencies

numpy, scipy, matplotlib, requests, rasterio, pyproj, Pillow
