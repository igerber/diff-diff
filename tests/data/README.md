# Test Data Fixtures

## hrs_edid_validation.csv

**Source:** Dobkin, C., Finkelstein, A., Kluender, R., & Notowidigdo, M. J. (2018).
"The Economic Consequences of Hospital Admissions." *American Economic Review*, 108(2), 308-352.
Replication kit: https://www.openicpsr.org/openicpsr/project/116186/version/V1/view

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

## lpdidtestdata1_core.csv

**Source:** Official example data bundled with Daniele Girardi's Stata `lpdid` package.

- Help-file URL: `http://fmwww.bc.edu/repec/bocode/l/lpdidtestdata1.dta`
- Estimation target: absorbing-treatment main path
- Python columns retained: `unit`, `time`, `Y`, `treat`

This fixture is used to validate that `diff_diff.LPDiD` reproduces the Stata
package's event-study and pooled estimates for the official absorbing example.

## lpdidtestdata1_event_stata.csv

Stata `lpdid` event-study output for `lpdidtestdata1_core.csv`, exported after:

```stata
use http://fmwww.bc.edu/repec/bocode/l/lpdidtestdata1.dta, clear
lpdid Y, time(time) unit(unit) treat(treat) pre_window(5) post_window(10) nograph
```

The exported columns are the rounded Stata benchmark values used by
`tests/test_lpdid.py`.

## lpdidtestdata1_pooled_stata.csv

Stata `lpdid` pooled pre/post output for the same run and sample as above.
This fixture locks down the default pooled-window behavior for the absorbing
main path without requiring CI to call Stata directly.

## lpdidtestdata1_nocomp_event_stata.csv

Stata `lpdid` event-study output for the same official example with `nocomp`
enabled:

```stata
use http://fmwww.bc.edu/repec/bocode/l/lpdidtestdata1.dta, clear
lpdid Y, time(time) unit(unit) treat(treat) pre_window(5) post_window(10) nocomp nograph
```

This fixture validates the common-composition sample restriction for the
absorbing main path without reweighting.

## lpdidtestdata1_nocomp_pooled_stata.csv

Stata `lpdid` pooled pre/post output for the same `nocomp` run as above.

## lpdidtestdata1_rw_event_stata.csv

Stata `lpdid` event-study output for the same official example with the `rw`
option enabled:

```stata
use http://fmwww.bc.edu/repec/bocode/l/lpdidtestdata1.dta, clear
lpdid Y, time(time) unit(unit) treat(treat) pre_window(5) post_window(10) rw nograph
```

This fixture validates the equally weighted ATT path for the absorbing design.

## lpdidtestdata1_rw_pooled_stata.csv

Stata `lpdid` pooled pre/post output for the same `rw` run as above.

## lpdidtestdata1_rw_nocomp_event_stata.csv

Stata `lpdid` event-study output for the same official example with both `rw`
and `nocomp` enabled:

```stata
use http://fmwww.bc.edu/repec/bocode/l/lpdidtestdata1.dta, clear
lpdid Y, time(time) unit(unit) treat(treat) pre_window(5) post_window(10) rw nocomp nograph
```

This fixture locks down the common-composition sample restriction under the
equally weighted ATT path.

## lpdidtestdata1_rw_nocomp_pooled_stata.csv

Stata `lpdid` pooled pre/post output for the same `rw nocomp` run as above.
