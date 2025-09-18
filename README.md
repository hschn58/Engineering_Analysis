# Engineering_Analysis

A curated set of small, focused numerical methods and applied-physics mini-projects—each living in its own folder with a notebook and/or Python script. The emphasis is on clarity of methods (discretizations, transforms, estimators) and reproducible figures.

> Folders in this repo include:
> `Fourier_Series_Gibbs_Analysis/`, `Diffusion_Constant_Evaluation/`, `Finite_Differences/`, `Finite_Element_Method/`, `Interpolation_Methods/`, `Topography/`, `X-Ray_Diffraction/`, `rPPG/`.  
> Most content is Jupyter notebooks with some Python scripts. (GitHub reports this repo as ~95% Jupyter Notebook, ~5% Python.)  
>
> _Last verified from the repo tree._


## What’s inside (one-liners)

- **Fourier_Series_Gibbs_Analysis/** — Explore Fourier partial sums and quantify Gibbs overshoot near discontinuities (step/square-like test functions). Typical outputs: partial-sum plots, error vs. N.
- **Diffusion_Constant_Evaluation/** — Simple estimators for diffusion/transport parameters from synthetic or provided data; sanity checks against analytic solutions where available.
- **Finite_Differences/** — Canonical FD stencils (1D/2D) for Poisson/heat/wave toy problems; boundary conditions, stability/CFL notes, and convergence demos.
- **Finite_Element_Method/** — Minimal FEM examples (triangular meshes, assembly of K, handling of Dirichlet BC) for Laplace on a disk.
- **Interpolation_Methods/** — Interpolation vs. approximation: piecewise linear/cubic, polynomial pitfalls, and residual visualization.
- **Topography/** — DEM download + resampling + basic rendering. Handy for turning real terrain data into meshes and shaded relief (used with OpenTopography APIs).
- **X-Ray_Diffraction/** — Bragg’s law mini-utilities, peak finding, and simple structure-factor illustrations for teaching/demo purposes.
- **rPPG/** — Remote photoplethysmography signal extraction from face video.  
  This is the most in-depth subproject in the repo: it evolved from basic FFT/wavelet/band-pass pipelines into a **full stack pipeline** that integrates  
  - ROI detection with edge-model AI  
  - Forward–backward Lucas–Kanade optical flow for ROI stabilization  
  - Standard-deviation motion metrics for robust segment weighting/rejection  
  - Camera consistency checks (aperture, exposure, white balance)  
  - Multiple spectral estimators (FFT, wavelet, Butterworth) on the stabilized signals  

  Outputs include synchronized BPM time series, confidence measures, and debug visualizations. See [`rPPG/Full_Stack`](./rPPG/Full_Stack) for the complete integrated workflow.


> Each folder is intentionally self-contained: read the notebook top cell (or `README.md` inside the folder, if present) for exact inputs/outputs.


## Quickstart

### 1) Environment
```bash
# a. create a clean environment (conda or venv)
conda create -n eng-analysis python=3.10 -y
conda activate eng-analysis
# or: python -m venv .venv && source .venv/bin/activate

# b. install common essentials (safe defaults for most folders)
pip install numpy scipy matplotlib jupyter pandas
```

## License 

This project is licensed under the [MIT License](./LICENSE).

