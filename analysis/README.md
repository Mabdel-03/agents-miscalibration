# Analysis

`fitting.py` provides the scaling-law fits used by the notebooks:

- `power_law_fit(x, y)` — `y = a·x^b` via log-log least squares (e.g. accuracy or ECE vs
  parameter count).
- `kim_regression(df, target)` — standardized OLS in the spirit of Kim et al. Eq. 1,
  regressing a system metric on capacity (+ capacity²), context-share rank, prompt
  complexity, and topology dummies, with 5-fold CV R².

## Workflow

```bash
python scripts/aggregate_results.py --run-id <run_id> --out analysis/<run_id>.parquet
```

Then in a notebook:

```python
import pandas as pd, json
from analysis.fitting import power_law_fit, kim_regression
df = pd.read_parquet("analysis/<run_id>.parquet")
# calibration_json / efficiency_json hold the nested per-cell metrics.
```

## Planned notebooks

1. **01_load_validate** — load the tidy table, confirm non-degenerate per-agent and
   system ECE, render a reliability diagram for one cell, sanity-check accuracies against
   published single-model numbers.
2. **02_scaling_curves** — performance / Ec / Ae / O% / message density / redundancy / ECE
   vs each of the three axes, faceted by topology.
3. **03_calibration_diagrams** — reliability diagrams across the grid; the **headline plot:
   (system ECE − mean per-agent ECE) vs each axis**, the core contribution.
