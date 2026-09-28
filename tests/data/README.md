# Test Data Fixtures

## hrs_edid_validation.csv

**Source:** Dobkin, C., Finkelstein, A., Kluender, R., & Notowidigdo, M. J. (2018).
"The Economic Consequences of Hospital Admissions." *American Economic Review*, 108(2), 308-352.
Replication kit: https://www.openicpsr.org/openicpsr/project/116186/version/V1/view

**License / redistribution:** This fixture is a derived four-column subset (de-identified HRS
public-use person id, wave, out-of-pocket spending, and first-hospitalization wave) of the
**publicly available** Dobkin et al. (2018) replication package deposited in the AEA's openICPSR
repository (project 116186, distributed by ICPSR for replication of the published article). Only
the derived subset is committed — the full source `HRS_long.dta` is **not** redistributed here
(`.gitignore`d as `replication_data/`; regenerate from the openICPSR deposit via the snippet
below). It is included solely as a regression-test fixture to replicate the paper's Table 6.
Consult the openICPSR deposit page for the deposit's exact Terms of Use.

**Sample selection:** Follows Sun & Abraham (2021), as used by Chen, Sant'Anna & Xie (2025)
Section 6:

1. Read `HRS_long.dta` from the Dobkin et al. replication kit
2. Keep waves 7-11, retain only individuals present in all 5 waves
3. Filter to ever-hospitalized individuals with `first_hosp >= 8`
4. Filter to ages 50-59 at hospitalization (`age_hosp`)
5. Drop wave 11 (no valid comparison group)
6. Recode `first_hosp == 11` as never-treated (`inf`)

**Expected counts:**

| Column | Values |
|--------|--------|
| Total individuals | 656 |
| Waves | 7, 8, 9, 10 |
| Rows | 2,624 |
| G=8 | 252 |
| G=9 | 176 |
| G=10 | 163 |
| G=inf | 65 |

**Columns:** `unit` (hhidpn), `time` (wave), `outcome` (oop_spend, 2005 dollars), `first_treat` (first_hosp)

**Note on sample size (656 vs 652):** Chen, Sant'Anna & Xie (2025) Table 6 reports **652**
individuals; this fixture yields **656**. The four-individual difference reflects a minor
sample-selection nuance (e.g. exact age-window or first-hospitalization tie handling) not fully
pinned down by the paper text. It is immaterial to the validation: every EDiD point estimate in
`tests/test_efficient_did_validation.py::TestHRSReplication` matches the published Table 6 value
to within **0.03 of the published standard error**.

**Regeneration:** Requires the Dobkin et al. replication kit (`.gitignore`d as `replication_data/`).

```python
import pandas as pd, numpy as np
df = pd.read_stata("replication_data/116186-V1/Replication-Kit/HRS/Data/HRS_long.dta")
sub = df[df["wave"].isin([7, 8, 9, 10, 11])]
balanced = sub.groupby("hhidpn")["wave"].nunique()
sub = sub[sub["hhidpn"].isin(balanced[balanced == 5].index)]
sub = sub[sub["hhidpn"].isin(sub[sub["first_hosp"].notna()]["hhidpn"].unique())]
fh = sub.groupby("hhidpn")["first_hosp"].first()
sub = sub[sub["hhidpn"].isin(fh[fh >= 8].index)]
ages = sub.groupby("hhidpn")["age_hosp"].first()
sub = sub[sub["hhidpn"].isin(ages[(ages >= 50) & (ages <= 59)].index)]
sub = sub[sub["wave"] <= 10]
sub["first_treat"] = sub["first_hosp"].apply(lambda x: np.inf if x == 11 else int(x))
out = sub[["hhidpn", "wave", "oop_spend", "first_treat"]].copy()
out.columns = ["unit", "time", "outcome", "first_treat"]
out["unit"] = out["unit"].astype(int)
out["time"] = out["time"].astype(int)
out.sort_values(["unit", "time"]).reset_index(drop=True).to_csv(
    "tests/data/hrs_edid_validation.csv", index=False
)
```
