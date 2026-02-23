# Ablation B: Temperature / Repeats Sensitivity

## Effect of Number of Repeats (k)

### Stability (mean σ across repeats)

| Dim | k=1 | k=2 | k=3 |
|-----|:---:|:---:|:---:|
| emotion | — | 0.0158 | 0.0196 |
| validation | — | 0.0133 | 0.0189 |
| helpfulness | — | 0.0167 | 0.0228 |
| safety | — | 0.0692 | 0.0897 |

### Human Alignment (MAE / Spearman)

| Dim | k=1 MAE | k=1 ρ | k=2 MAE | k=2 ρ | k=3 MAE | k=3 ρ |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| emotion | 0.4717 | 0.6583 | 0.475 | 0.6612 | 0.4783 | 0.6508 |
| validation | 0.465 | 0.5078 | 0.4667 | 0.508 | 0.4694 | 0.4937 |
| helpfulness | 0.4467 | 0.5526 | 0.4483 | 0.5614 | 0.4478 | 0.5614 |
| safety | 0.395 | 0.8547 | 0.39 | 0.8709 | 0.3839 | 0.875 |

### Calibrated MAE (isotonic, 80/20 split)

| Dim | k=1 raw→cal | k=2 raw→cal | k=3 raw→cal |
|-----|:---:|:---:|:---:|
| emotion | 0.4333→0.2011 | 0.4167→0.2057 | 0.4222→0.204 |
| validation | 0.4167→0.2018 | 0.4167→0.2 | 0.4167→0.215 |
| helpfulness | 0.5167→0.2087 | 0.5167→0.2071 | 0.5222→0.2038 |
| safety | 0.2917→0.3107 | 0.3333→0.2691 | 0.3306→0.2541 |

## Key Findings

1. **How many repeats are needed?** Compare k=1 vs k=3 stability and alignment.
2. **Is k=2 a good cost-performance trade-off?**
3. **Does calibration compensate for fewer repeats?**