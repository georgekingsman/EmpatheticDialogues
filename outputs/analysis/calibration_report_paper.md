# Calibration Report (Paper-Grade)

## Data Split
- Train: 60 samples (60%)
- Dev: 20 samples (20%)
- Test: 20 samples (20%)

## Isotonic Calibration (Route 1 — Primary Result)

### Test-Set Metrics

| Dimension | Raw MAE | Cal MAE | MAE Reduction | Raw ρ | Cal ρ |
|-----------|:---:|:---:|:---:|:---:|:---:|
| emotion | 0.547 | 0.205 | 62.6% | 0.607 | 0.581 |
| validation | 0.544 | 0.249 | 54.2% | 0.321 | 0.325 |
| helpfulness | 0.506 | 0.219 | 56.6% | 0.759 | 0.759 |
| safety | 0.425 | 0.285 | 32.9% | 0.789 | 0.785 |

### Bootstrap 95% CI (1000 iterations)

| Dimension | MAE reduction % [95% CI] | Cal MAE [95% CI] | Cal ρ [95% CI] |
|-----------|:---:|:---:|:---:|
| emotion | 61.9944 [51.0349, 73.2272] | 0.2079 [0.145, 0.2814] | 0.5656 [0.1629, 0.8454] |
| validation | 53.7669 [34.9674, 67.0708] | 0.2493 [0.1769, 0.3307] | 0.3183 [-0.1803, 0.7122] |
| helpfulness | 55.9601 [39.7251, 70.1867] | 0.2212 [0.1458, 0.3038] | nan [nan, nan] |
| safety | 32.8914 [5.3105, 53.3654] | 0.2828 [0.1808, 0.3871] | 0.7617 [0.4869, 0.9251] |

## Ordinal Calibration (Route 2 — Comparison)

| Dimension | Raw MAE | Cal MAE | MAE Reduction | Raw ρ | Cal ρ |
|-----------|:---:|:---:|:---:|:---:|:---:|
| emotion | 0.547 | 0.547 | 0.0% | 0.607 | 0.607 |
| validation | 0.544 | 0.544 | 0.0% | 0.321 | 0.321 |
| helpfulness | 0.506 | 0.506 | 0.0% | 0.759 | 0.759 |
| safety | 0.425 | 0.425 | 0.0% | 0.789 | 0.789 |