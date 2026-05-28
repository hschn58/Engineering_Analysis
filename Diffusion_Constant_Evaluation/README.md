# Diffusion Constant Evaluation

Estimates the diffusion constant of carbon in 1018 steel from experimental Knoop hardness profiles obtained via pack carburization in the UW-Madison materials science lab.

## Method

Carbon diffusion into steel follows the complementary error function solution:

```
(H(x,t) - H0) / (Hs - H0) = erfc(x / (2 * sqrt(D * t)))
```

where H0 is the bulk hardness, Hs is the surface hardness, x is depth, t is carburization time, and D is the diffusion constant.

The script:
1. Reads hardness-vs-depth data for 11 carburization durations (500-3000 s) from `data_hardness.xlsx`
2. Plots all curves on a log-log scale
3. Inverts the error function at multiple depth points to estimate D for each dataset
4. Computes a weighted average across all measurements

![Hardness vs depth for all carburization times](Hardness%20Data.png)

## Result

Estimated D = 8.43 x 10^-10 m^2/s (theoretical prediction: 3.03 x 10^-11 m^2/s).

## Usage

```bash
python All_hardness.py
```

Requires `data_hardness.xlsx` in the same directory (not included in the repo).

## Dependencies

numpy, scipy, matplotlib, pandas, openpyxl
