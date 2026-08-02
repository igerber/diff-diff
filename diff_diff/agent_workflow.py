"""Stateless orchestrator: print the recommended diff-diff workflow with
the caller's column names wired in.

This module exists to give LLM agents a single, recognizable entrypoint
that names the rest of the agent-facing workflow (`profile_panel`,
`get_llm_guide`, `practitioner_next_steps`, `BusinessReport`). The
function does not fit, inspect, or recommend — it templates a copy-
pasteable script.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Pattern → df-callable estimator class names. Flat union below is the
# `fit_candidates` field of the returned dict; each name must remain a
# valid `hasattr(diff_diff, name)` (locked by the contract test in
# tests/test_agent_discoverability.py and tests/test_agent_workflow.py).
# Patterns intentionally exclude post-fit and pre-fit diagnostics
# (PreTrendsPower takes pre-treatment coefficients, HonestDiD takes a
# fitted results object); those are mentioned separately in the
# templated Step 4 of the script.
_WORKFLOW_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "Staggered adoption + binary treatment + has_never_treated control",
        ("CallawaySantAnna", "SunAbraham", "ImputationDiD"),
    ),
    (
        "Continuous treatment dose (non-binary numeric intensity)",
        ("ContinuousDiD",),
    ),
    (
        "Heterogeneous adoption intensity across treated units",
        ("HeterogeneousAdoptionDiD",),
    ),
    (
        "Simple 2x2 DiD (binary treatment, two periods, no staggering)",
        ("DifferenceInDifferences",),
    ),
)


def _safe_kwarg(name: str, value: Optional[str]) -> Optional[str]:
    """Render ``name=<python-literal>`` using repr() for source-safety.

    Column labels containing quotes, backslashes, or other special
    characters must not break the emitted "copy-pasteable" script.
    Python's built-in ``repr()`` produces a valid string literal for
    any str input (including embedded quotes / backslashes /
    newlines), so ``f"{name}={value!r}"`` is injection-safe by
    construction. ``None`` returns ``None`` so the caller can drop
    the kwarg.
    """
    if value is None:
        return None
    return f"{name}={value!r}"


def _join_kwargs(**kwargs: Optional[str]) -> str:
    parts = [_safe_kwarg(k, v) for k, v in kwargs.items()]
    return ", ".join(p for p in parts if p is not None)


def agent_workflow(
    df: Any,
    *,
    unit: str,
    time: str,
    treatment: str,
    outcome: str,
    first_treat: Optional[str] = None,
    df_name: str = "df",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Print the recommended diff-diff workflow with your column names wired in.

    Stateless orchestrator. Calls nothing internally. Returns a dict;
    optionally prints a copy-pasteable script (``verbose=True``, the
    default). ``df`` is not inspected — column names are templated
    verbatim into the output.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel data. Not inspected; included so the agent
        can pass the same handle along to the next call.
    unit : str
        Column identifying the cross-sectional unit.
    time : str
        Column identifying the time period.
    treatment : str
        Column holding the treatment indicator or dose.
    outcome : str
        Column holding the outcome variable.
    first_treat : str, optional
        Column with each unit's first-treatment period (or NaN for
        never-treated controls). When supplied, the templated Step 3
        switches from a ``DifferenceInDifferences.fit(treatment=...)``
        example to a ``CallawaySantAnna().fit(first_treat=...)``
        example, matching the actual fit signatures (passing
        ``treatment=`` to CallawaySantAnna's ``.fit()`` would raise
        TypeError).
    df_name : str, default ``"df"``
        Identifier under which the caller's dataframe is bound in
        their namespace. Templated verbatim into the emitted script
        as the first positional argument of every call
        (``profile_panel({df_name}, ...)``,
        ``<Estimator>().fit({df_name}, ...)``) so the script is
        directly executable when the caller's local variable matches.
        If the caller has ``panel = pd.read_parquet(...)``, passing
        ``df_name="panel"`` produces a script that references
        ``panel`` instead of ``df``. Must be a valid Python identifier
        (not enforced; non-identifier values produce a script that
        won't parse).
    verbose : bool, default True
        If True, print the script to stdout. The dict is always
        returned regardless.

    Returns
    -------
    dict
        Keys:

        - ``"profile_call"`` (str): call signature for
          :func:`diff_diff.profile_panel`.
        - ``"guide_call"`` (str): call signature for
          :func:`diff_diff.get_llm_guide`.
        - ``"fit_candidates"`` (list of str): flat union of estimator /
          diagnostic class names referenced in the workflow patterns.
          Every name resolves on the top-level ``diff_diff`` namespace.
        - ``"validation_calls"`` (list of str): call signatures for the
          post-fit validation step.
        - ``"reporting_call"`` (str): call signature for
          :class:`diff_diff.BusinessReport`.
        - ``"script"`` (str): printable multi-line workflow.

    Examples
    --------
    >>> import pandas as pd
    >>> import diff_diff
    >>> df = pd.DataFrame({
    ...     "firm_id": [1, 1, 2, 2],
    ...     "year": [0, 1, 0, 1],
    ...     "treated": [0, 0, 1, 1],
    ...     "logwage": [0.1, 0.2, 0.1, 0.9],
    ... })
    >>> out = diff_diff.agent_workflow(df, unit="firm_id", time="year",
    ...                                treatment="treated", outcome="logwage",
    ...                                verbose=False)
    >>> "profile_panel" in out["script"]
    True
    """
    del df  # intentionally unused: orchestrator templates from column names only

    profile_call = (
        f"diff_diff.profile_panel({df_name}, "
        f"{_join_kwargs(unit=unit, time=time, treatment=treatment, outcome=outcome)})"
    )
    guide_call = 'diff_diff.get_llm_guide("autonomous")'

    # Step 3 example: branch on first_treat presence.
    # - With first_treat: a staggered structure is strongly implied, BUT
    #   `first_treat` does not by itself identify which estimator to use:
    #   CallawaySantAnna (binary staggered), ContinuousDiD (continuous-
    #   dose with first_treat), and HeterogeneousAdoptionDiD event-study
    #   (heterogeneous intensity with first_treat) all accept it.
    #   Show CallawaySantAnna as the binary-staggered canonical example
    #   and list the alternatives for continuous / heterogeneous designs
    #   so an agent isn't steered to the wrong estimator.
    # - Without first_treat: the orchestrator does not inspect df, so it
    #   CANNOT infer whether the panel is 2x2 binary vs continuous-dose
    #   vs heterogeneous-adoption. Show a DifferenceInDifferences call
    #   as the "simple 2x2" example and label it explicitly conditional
    #   on that shape.
    if first_treat is not None:
        fit_example_kwargs = _join_kwargs(
            outcome=outcome, unit=unit, time=time, first_treat=first_treat
        )
        fit_example_call = f"diff_diff.CallawaySantAnna().fit({df_name}, {fit_example_kwargs})"
        step3_label_lines = [
            "Step 3 - Fit. Your data has `first_treat` -> staggered structure.",
            "`first_treat` alone does NOT identify a single estimator; pick by",
            "treatment shape:",
            "  - Binary staggered  : CallawaySantAnna (shown) / SunAbraham / ImputationDiD",
            "  - Continuous dose   : ContinuousDiD (also takes first_treat=)",
            "  - Heterogeneous adoption intensity:",
            "                        HeterogeneousAdoptionDiD (event study,",
            "                        takes first_treat=; first_treat_col= is a",
            "                        deprecated alias slated for 4.0 removal)",
        ]
    else:
        fit_example_kwargs = _join_kwargs(
            outcome=outcome, unit=unit, time=time, treatment=treatment
        )
        fit_example_call = (
            f"diff_diff.DifferenceInDifferences().fit({df_name}, {fit_example_kwargs})"
        )
        step3_label_lines = [
            "Step 3 - Fit. Pick a candidate from Step 2's patterns based on your",
            "treatment/time shape. The example below shows the simple 2x2 case",
            "(binary treatment + binary time); substitute ContinuousDiD /",
            "HeterogeneousAdoptionDiD / etc. when your design is not 2x2",
            "(DifferenceInDifferences.fit() validates and rejects non-binary",
            "treatment or time).",
        ]
    step3_comment_block = "\n".join(f"# {line}" for line in step3_label_lines)

    validation_calls = [
        "diff_diff.practitioner_next_steps(result)",
    ]
    reporting_call = "diff_diff.BusinessReport(result).full_report()"

    fit_candidates: List[str] = []
    pattern_lines: List[str] = []
    for label, names in _WORKFLOW_PATTERNS:
        pattern_lines.append(f"#   - {label}")
        pattern_lines.append(f"#       candidates: {', '.join(names)}")
        for n in names:
            if n not in fit_candidates:
                fit_candidates.append(n)
    pattern_block = "\n".join(pattern_lines)

    diagnostics_block = (
        "# Parallel-trends sensitivity / power (take a fitted result or\n"
        "# pre-trend coefficients, NOT df+columns):\n"
        "#     diff_diff.PreTrendsPower / diff_diff.HonestDiD"
    )

    # Templated output is a valid Python script: every prose line is a
    # `#` comment, every code line stands at column 0 and runs as-is.
    # Step 5 wraps full_report() in print() so end-to-end execution
    # actually produces the stakeholder narrative.
    script = f"""# diff_diff workflow for your data
# =================================
#
# Step 1 - Describe the panel:
profile = {profile_call}
print(profile)

# Step 2 - Choose an estimator. Consult the routing matrix:
print({guide_call})

# Routing patterns (df-callable estimators):
{pattern_block}
#
{diagnostics_block}

{step3_comment_block}
result = {fit_example_call}

# Step 4 - Validate:
{validation_calls[0]}

# Step 5 - Report:
print({reporting_call})

# Full reference: diff_diff.get_llm_guide("full")
# Practitioner recipe: diff_diff.get_llm_guide("practitioner")
"""

    if verbose:
        print(script)

    return {
        "profile_call": profile_call,
        "guide_call": guide_call,
        "fit_candidates": fit_candidates,
        "validation_calls": validation_calls,
        "reporting_call": reporting_call,
        "script": script,
    }
