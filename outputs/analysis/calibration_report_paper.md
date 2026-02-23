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

### Bootstrap 95% CI (100 iterations)

| Dimension | MAE reduction % [95% CI] | Cal MAE [95% CI] | Cal ρ [95% CI] |
|-----------|:---:|:---:|:---:|
| emotion | 62.8273 [51.6583, 73.3787] | 0.2021 [0.1368, 0.2694] | 0.5541 [0.1743, 0.8444] |
| validation | 54.1111 [37.4247, 66.823] | 0.2485 [0.1673, 0.3438] | 0.3387 [-0.238, 0.722] |
| helpfulness | 56.773 [43.3439, 71.2092] | 0.2198 [0.1461, 0.2923] | 0.7558 [0.6032, 0.8934] |
| safety | 31.3025 [10.489, 54.0763] | 0.2849 [0.1926, 0.3986] | 0.7695 [0.5247, 0.9252] |

## Ordinal Calibration (Route 2 — Comparison)

| Dimension | Raw MAE | Cal MAE | MAE Reduction | Raw ρ | Cal ρ |
|-----------|:---:|:---:|:---:|:---:|:---:|
| emotion | 0.547 | 0.547 | 0.0% | 0.607 | 0.607 |
| validation | 0.544 | 0.544 | 0.0% | 0.321 | 0.321 |
| helpfulness | 0.506 | 0.506 | 0.0% | 0.759 | 0.759 |
| safety | 0.425 | 0.425 | 0.0% | 0.789 | 0.789 |