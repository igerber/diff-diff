from typing import Any, Dict, Optional, Union

from diff_diff.lpdid_results import LPDiDResults

__all__ = ["LPDiD", "LPDiDResults"]


class LPDiD:
    def __init__(
        self,
        pre_window: int = 2,
        post_window: int = 0,
        control_group: str = "clean",
        reweight: bool = False,
        no_composition: bool = False,
        pmd: Optional[Union[str, int]] = None,
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        rank_deficient_action: str = "warn",
    ):
        self.pre_window = pre_window
        self.post_window = post_window
        self.control_group = control_group
        self.reweight = reweight
        self.no_composition = no_composition
        self.pmd = pmd
        self.alpha = alpha
        self.cluster = cluster
        self.rank_deficient_action = rank_deficient_action
        self._validate_params()
        self.is_fitted_ = False
        self.results_: Optional[LPDiDResults] = None

    def _validate_params(self) -> None:
        if self.control_group not in ("clean", "never_treated"):
            raise ValueError("control_group must be 'clean' or 'never_treated'")
        if self.rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError("rank_deficient_action must be 'warn', 'error', or 'silent'")

    def get_params(self) -> Dict[str, Any]:
        return {
            "pre_window": self.pre_window,
            "post_window": self.post_window,
            "control_group": self.control_group,
            "reweight": self.reweight,
            "no_composition": self.no_composition,
            "pmd": self.pmd,
            "alpha": self.alpha,
            "cluster": self.cluster,
            "rank_deficient_action": self.rank_deficient_action,
        }

    def set_params(self, **params: Any) -> "LPDiD":
        previous_values = {}
        for key, value in params.items():
            if hasattr(self, key):
                previous_values[key] = getattr(self, key)
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        try:
            self._validate_params()
        except ValueError:
            for key, value in previous_values.items():
                setattr(self, key, value)
            raise
        return self
