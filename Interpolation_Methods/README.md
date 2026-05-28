# Interpolation Methods

A library of curve fitting functions that use linearized least squares to fit common models to data.

## Available Fits

| Function | Model | Linearization |
|---|---|---|
| `poly_fit(data, degree)` | a0 + a1*x + ... + an*x^n | Direct (already linear) |
| `exp_fit(data)` | K * exp(lambda * x) | ln(y) = ln(K) + lambda * x |
| `power_fit(data)` | K * x^alpha | ln(y) = ln(K) + alpha * ln(x) |
| `gauss_fit(data)` | A * exp(-((x - mu) / sigma)^2) | ln(y) = -(x^2)/c^2 + 2xb/c^2 + const |

Each function:
- Solves the linearized least-squares problem via `numpy.linalg.lstsq`
- Reports R^2
- Optionally plots the data points and fitted curve (`plotit=True`)
- Returns the fitted coefficients

## Files

- `Curve_Fits.py` -- Full library with all four fit types plus `fit_plot`
- `fit_data.py` -- Standalone version with polynomial, exponential, and power-law fits

## Usage

```python
import numpy as np
from Curve_Fits import gauss_fit, poly_fit, exp_fit, power_fit

data = np.column_stack([x_values, y_values])
coeffs = poly_fit(data, degree=3, plotit=True)
```

## Dependencies

numpy, matplotlib
