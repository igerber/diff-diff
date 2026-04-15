from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd


@dataclass
class LPDiDResults:
    event_study: Optional[pd.DataFrame]
    pooled: Optional[pd.DataFrame]
    n_obs: int
    n_treated_units: int
    n_control_units: int
    pre_window: int
    post_window: int
    control_group: str
    reweight: bool
    no_composition: bool
    pmd: Optional[Union[str, int]]
    alpha: float = 0.05

    def to_dataframe(self, kind: str = "event") -> pd.DataFrame:
        if kind == "event":
            return self.event_study.copy() if self.event_study is not None else pd.DataFrame()
        if kind == "pooled":
            return self.pooled.copy() if self.pooled is not None else pd.DataFrame()
        raise ValueError("kind must be 'event' or 'pooled'")

    def summary(self) -> str:
        return "LPDiDResults(summary not implemented yet)"

    def print_summary(self) -> None:
        print(self.summary())
