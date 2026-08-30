"""Results container for the DMLDiD estimator (Chang 2020, staggered).

``DMLDiDResults`` subclasses :class:`~diff_diff.staggered_results.CallawaySantAnnaResults`
and inherits its kit-based post-fit ``aggregate()`` machinery (event study
with sup-t bands, group/simple aggregation, bootstrap replay; ``total`` on
panel fits — repeated-cross-section fits fail it closed). The subclass adds
the cross-fitting provenance the parent has no concept of: learner specs,
fold count, per-cell cross-fit diagnostics, and the inference-provenance
fields (``seed``/``n_bootstrap``/``bootstrap_weights``/``cband``) that move
estimates or inference.

``vcov_type`` stays at the inherited ``"hc1"``: in this library ``hc1`` with
``cluster=None`` IS the per-sampling-unit influence-function variance by
definition (REGISTRY.md "IF-based variance estimators..." — the default),
and DMLDiD's augmented-score SE ``sqrt(mean(psi_bar**2)/n)`` is exactly
that on NO-DESIGN fits — per UNIT on panel fits, per OBSERVATION on
repeated-cross-section fits (rows are the sampling units there). Under a
``survey_design=``/``cluster=`` the per-cell SE is design-based instead:
full-design TSL fits use the CR1 / weighted-IF variance (the CS
clustered-``hc1`` convention: ``SurveyDesign(psu=...)`` routed through the
shared stratified-PSU meat), while replicate-weight fits use IF-reweighting
via ``compute_replicate_if_variance`` on the same per-cell payload
(``df = rank(replicate matrix) - 1``; REGISTRY DMLDiD Note).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from diff_diff.results_base import _json_safe_label
from diff_diff.staggered_results import CallawaySantAnnaResults

__all__ = ["DMLDiDResults"]


def _serialize_cross_fit_diagnostics(
    diagnostics: Optional[Dict[Any, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Serialize tuple-keyed per-cell diagnostics to JSON-safe form.

    Tuple keys become ``"g={g},t={t}"`` strings; ndarray values become lists;
    numpy scalars become Python scalars. Nested dicts are converted
    recursively (the schema is two levels deep: per-cell dict of stage
    sub-dicts / scalars).
    """
    if diagnostics is None:
        return None

    def _convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        return _json_safe_label(value)

    return {f"g={g},t={t}": _convert(entry) for (g, t), entry in diagnostics.items()}


@dataclass
class DMLDiDResults(CallawaySantAnnaResults):
    """Results from DMLDiD (Chang 2020 DML DiD, staggered ATT(g,t)).

    Inherits the full Callaway-Sant'Anna results surface — ``att``/``se``
    aliases, ``to_dataframe``, post-fit ``aggregate()`` (simple / event_study
    / group, plus total on panel non-survey fits; repeated-cross-section
    AND declared-survey fits fail ``total`` closed) with bootstrap replay —
    and adds the DML provenance fields below. ``cluster_name``/
    ``n_clusters``/``df_inference``/``survey_metadata`` are populated on
    survey/``cluster=`` fits (CS conventions: ``survey_metadata`` marks a
    DECLARED design; ``df_inference`` carries the bare-cluster df). Every
    inherited CS-only field that DMLDiD never populates (``epv_*``,
    ``pscore_fallback``, ``allow_unbalanced_panel``,
    ``used_rc_on_unbalanced_panel``, ``influence_functions``,
    ``event_study_effects``/``event_study_vcov``/``event_study_vcov_index``/
    ``event_study_df``) stays at its inherited default and is inert.
    ``group_effects`` stays ``None`` permanently — ``aggregate("group")``
    returns a separate container and never mutates this object. ``panel``
    is NOT inert: it carries the fit's DECLARED design — ``True`` for the
    Case 1 panel lane, ``False`` for the Case 2 repeated-cross-section lane
    (``DMLDiD(panel=False)``) — and report rendering plus the aggregation
    ``n_kind`` read it for "units" vs "observations" semantics.

    Attributes
    ----------
    propensity_learner, outcome_learner : str
        The learner specs as configured — the string name verbatim; a
        passed LIBRARY-native learner object keeps its configuration-aware
        ``repr()``; a FOREIGN object is recorded as its qualified class
        name only (its ``__repr__`` could embed sensitive content). Never
        the object itself: result pickles must not retain arbitrary user
        objects.
    n_folds : int
        Cross-fitting fold count K as REQUESTED (the configuration value).
    effective_n_folds : int, optional
        The REALIZED fold count when it differs from ``n_folds`` — set only
        when a coarse survey/cluster PSU design reduced K to the global PSU
        count to preserve cluster cohesion (warned at fit); ``None``
        otherwise.
    cross_fit_diagnostics : dict, optional
        Per-``(g, t)`` cross-fit diagnostics: per-stage fold losses and fit
        counts, ``p_hat``, propensity clip count, and the fold-seed
        derivation. Cells that failed before fold assignment carry no entry.
    seed : int, optional
        Root seed for the per-cell fold draws (and the bootstrap when
        enabled). With ``seed=None`` the recorded per-fit entropy is drawn
        from OS randomness.
    n_bootstrap, bootstrap_weights, cband
        Multiplier-bootstrap configuration used at fit.
    """

    propensity_learner: Any = "logit"
    outcome_learner: Any = "linear"
    n_folds: int = 5
    effective_n_folds: Optional[int] = None
    cross_fit_diagnostics: Optional[Dict[Any, Dict[str, Any]]] = field(default=None, repr=False)
    seed: Optional[int] = None
    n_bootstrap: int = 0
    bootstrap_weights: Optional[str] = None
    cband: bool = True

    def __repr__(self) -> str:
        folds = f"n_folds={self.n_folds}"
        if self.effective_n_folds is not None:
            folds += f" (effective {self.effective_n_folds})"
        return (
            f"DMLDiDResults(att={self.overall_att:.6g}, se={self.overall_se:.6g}, "
            f"n_cells={len(self.group_time_effects)}, {folds}, "
            f"propensity_learner={self.propensity_learner!r}, "
            f"outcome_learner={self.outcome_learner!r})"
        )

    def _seed_display(self) -> str:
        if self.seed is not None:
            return str(self.seed)
        if self.cross_fit_diagnostics:
            for entry in self.cross_fit_diagnostics.values():
                fold_seed = entry.get("fold_seed")
                if fold_seed and "entropy" in fold_seed:
                    return f"None (entropy {fold_seed['entropy']})"
        return "None"

    def summary(self, alpha: Optional[float] = None) -> str:
        """Formatted summary with the DML banner and cross-fit header block.

        Post-processes the parent summary: replaces the hardcoded
        Callaway-Sant'Anna banner line, relabels the normal-theory
        statistic columns (z, not t), and prepends the DML provenance
        header (never reimplements the parent body).

        Parameters
        ----------
        alpha : float, optional
            Accepted for signature uniformity. The stored intervals were
            computed at fit time; a value different from the stored
            ``alpha`` raises rather than silently relabeling fit-time
            intervals (percentile bootstrap intervals in particular cannot
            be reconstructed from the reported SE).
        """
        if alpha is not None and alpha != self.alpha:
            raise ValueError(
                f"This result stores intervals computed at alpha="
                f"{self.alpha}; summary() never recomputes or relabels "
                f"stored inference (requested alpha={alpha}). Re-fit with "
                "the desired alpha (bootstrap percentile intervals cannot "
                "be reconstructed from the reported SE)."
            )
        base = super().summary(alpha)
        cs_banner = "Callaway-Sant'Anna Staggered Difference-in-Differences Results"
        dml_banner = "DML DiD (Chang 2020) Staggered Difference-in-Differences Results"
        if self.n_bootstrap > 0:
            # Bootstrap fits carry PERCENTILE p-values and intervals: the
            # statistic column is effect/SE (a standardized ratio, not the
            # p-value's source), so label the p column honestly.
            base = base.replace("t-stat", "z-stat").replace("P>|t|", "Boot. p")
        elif self.survey_metadata is None and self.df_inference is None:
            # Analytical NO-DESIGN inference is normal-theory throughout
            # (safe_inference with df=None): relabel the parent's t columns.
            # Survey/bare-cluster fits use finite-df t inference
            # (df=df_survey / df_inference) — the parent's t labels are
            # CORRECT there and must not be relabeled.
            base = base.replace("t-stat", "z-stat").replace("P>|t|", "P>|z|")
        lines = base.split("\n")
        for i, line in enumerate(lines):
            if cs_banner in line:
                lines[i] = dml_banner.center(85)
                break

        n_degenerate = 0
        if self.cross_fit_diagnostics is not None:
            n_degenerate = sum(
                1
                for entry in self.cross_fit_diagnostics.values()
                if entry.get("skip_reason") is not None
            )
        header = []
        if self.panel is False:
            # Conditional line only (panel output stays byte-stable): the
            # declared design changes score, variance, and count semantics.
            header.append(f"{'Design:':<30} {'repeated cross sections':>10}")
        header += [
            f"{'Propensity learner:':<30} {self.propensity_learner!r:>10}",
            f"{'Outcome learner:':<30} {self.outcome_learner!r:>10}",
            f"{'Cross-fitting folds (K):':<30} {self.n_folds:>10}",
        ]
        if self.effective_n_folds is not None:
            header.append(f"{'Effective folds (PSU-reduced):':<30} {self.effective_n_folds:>10}")
        header += [
            # The seed line renders UNCONDITIONALLY: the fold draw moves
            # point estimates on every fit, bootstrap or not. With
            # seed=None the OS-drawn fold entropy (recorded per cell in
            # cross_fit_diagnostics) is surfaced so the fit stays
            # auditable/reproducible.
            f"{'Seed:':<30} {self._seed_display():>10}",
        ]
        if n_degenerate:
            header.append(f"{'Degenerate cells skipped:':<30} {n_degenerate:>10}")
        if self.n_bootstrap > 0:
            header.append(f"{'Bootstrap iterations:':<30} {self.n_bootstrap:>10}")
            header.append(f"{'Bootstrap weights:':<30} {str(self.bootstrap_weights):>10}")
            header.append(f"{'Uniform bands (cband):':<30} {str(self.cband):>10}")
            header.append(
                "Inference: multiplier-bootstrap PERCENTILE p-values and "
                "intervals (the z-stat column is the effect/SE ratio, not "
                "the p-value's source)."
            )

        # Insert the DML header right after the banner block (banner line is
        # framed by '=' rules at indices 0 and 2; the blank line is index 3).
        insert_at = 4
        return "\n".join(lines[:insert_at] + header + [""] + lines[insert_at:])

    def to_dict(self) -> Dict[str, Any]:
        """Headline dict extended with DML provenance (JSON-serializable)."""
        result = super().to_dict()
        # Fields already hold the repr-of-spec STRING (fit stores names
        # verbatim and objects as repr) — no double-repr.
        result["propensity_learner"] = str(self.propensity_learner)
        result["outcome_learner"] = str(self.outcome_learner)
        result["n_folds"] = int(self.n_folds)
        result["effective_n_folds"] = (
            None if self.effective_n_folds is None else int(self.effective_n_folds)
        )
        result["seed"] = None if self.seed is None else int(self.seed)
        result["n_bootstrap"] = int(self.n_bootstrap)
        result["bootstrap_weights"] = self.bootstrap_weights
        result["cband"] = bool(self.cband)
        # The inherited serializer emits only the headline dict; pscore_trim
        # moves nuisance clipping (and so estimates/inference) — surface it.
        result["pscore_trim"] = float(self.pscore_trim)
        if self.reference_event_times is not None:
            result["reference_event_times"] = [
                _json_safe_label(v) for v in self.reference_event_times
            ]
        result["cross_fit_diagnostics"] = _serialize_cross_fit_diagnostics(
            self.cross_fit_diagnostics
        )
        return result
