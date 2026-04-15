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

    def __repr__(self) -> str:
        return (
            "LPDiDResults("
            f"n_obs={self.n_obs}, "
            f"n_treated_units={self.n_treated_units}, "
            f"n_control_units={self.n_control_units}, "
            f"pre_window={self.pre_window}, "
            f"post_window={self.post_window}, "
            f"control_group={self.control_group!r})"
        )

    def to_dataframe(self, level: str = "event") -> pd.DataFrame:
        if level == "event":
            if self.event_study is None:
                raise ValueError("event_study dataframe was not computed")
            return self.event_study.copy()
        if level == "pooled":
            if self.pooled is None:
                raise ValueError("pooled dataframe was not computed")
            return self.pooled.copy()
        raise ValueError("level must be 'event' or 'pooled'")

    def summary(self, alpha: Optional[float] = None) -> str:
        alpha = self.alpha if alpha is None else alpha
        return f"LPDiDResults(summary not implemented yet, alpha={alpha:.3g})"

    def print_summary(self, alpha: Optional[float] = None) -> None:
        print(self.summary(alpha=alpha))
