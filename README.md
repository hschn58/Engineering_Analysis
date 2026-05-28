# Engineering_Analysis

A curated set of small, focused numerical methods and applied-physics mini-projects—each living in its own folder with a notebook and/or Python script. The emphasis is on clarity of methods (discretizations, transforms, estimators) and reproducible figures.

> Folders in this repo include:
> `Fourier_Series_Gibbs_Analysis/`, `Diffusion_Constant_Evaluation/`, `Finite_Differences/`, `Finite_Element_Method/`, `Interpolation_Methods/`, `Topography/`, `X-Ray_Diffraction/`.  
> Each folder contains a Python script (or MATLAB, for XRD) and a README with details.


## What’s inside (one-liners)

- **Fourier_Series_Gibbs_Analysis/** — Interactive animation of Fourier partial sums on a smoothed sawtooth, with a slider to control discontinuity sharpness and observe the Gibbs phenomenon.
- **Diffusion_Constant_Evaluation/** — Estimates the diffusion constant of carbon in 1018 steel from pack-carburization Knoop hardness data using the error-function solution.
- **Finite_Differences/** — Two FD solvers: neutron flux in a cylindrical water tank (diffusion equation), and axial displacement of a bar with nonuniform elastic modulus.
- **Finite_Element_Method/** — Solves Laplace's equation on a unit disk with linear triangular elements, Delaunay mesh generation, sparse assembly, and Dirichlet BCs.
- **Interpolation_Methods/** — Curve fitting library: Gaussian, polynomial, exponential, and power-law fits via linearized least squares, with R² reporting.
- **Topography/** — Downloads SRTM elevation data, reprojects to UTM, fetches NAIP aerial imagery, and renders a textured 3D surface plot.
- **X-Ray_Diffraction/** — MATLAB analysis of a tungsten XRD pattern: peak finding, Bragg’s-law d-spacing, lattice parameter extraction, and error quantification.


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

