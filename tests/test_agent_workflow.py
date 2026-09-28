"""Tests for the stateless agent_workflow() orchestrator.

Mirrors the content-stability pattern from tests/test_guides.py: assert
fingerprint strings appear in the output rather than pinning exact
formatting. The orchestrator's stability contract is that it names the
five canonical workflow primitives in a copy-pasteable script.
"""

import pandas as pd
import pytest

import diff_diff


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "firm_id": [1, 1, 2, 2],
            "year": [0, 1, 0, 1],
            "treated": [0, 0, 1, 1],
            "logwage": [0.1, 0.2, 0.1, 0.9],
        }
    )


def test_returns_dict_with_canonical_keys(df):
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=False,
    )
    expected = {
        "profile_call",
        "guide_call",
        "fit_candidates",
        "validation_calls",
        "reporting_call",
        "script",
    }
    assert expected <= set(out.keys())


def test_script_names_canonical_workflow(df):
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=False,
    )
    script = out["script"]
    for name in (
        "profile_panel",
        "get_llm_guide",
        "practitioner_next_steps",
        "BusinessReport",
    ):
        assert name in script, f"{name!r} missing from script"


def test_templates_column_names(df):
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        first_treat="cohort",
        verbose=False,
    )
    script = out["script"]
    for col in ("firm_id", "year", "treated", "logwage", "cohort"):
        assert col in script, f"column {col!r} missing from templated script"


def test_first_treat_omitted_when_none(df):
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=False,
    )
    # Without first_treat=, the templated Step 3 should NOT mention it.
    assert "first_treat=" not in out["script"]


def test_first_treat_appears_when_provided(df):
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        first_treat="cohort_year",
        verbose=False,
    )
    # repr() produces single-quoted string literals for simple labels.
    assert "first_treat='cohort_year'" in out["script"]


def test_first_treat_switches_step3_estimator(df):
    """Step 3 must showcase a fit signature compatible with the data shape.

    - first_treat=None  -> DifferenceInDifferences (takes `treatment=`,
      does NOT take `first_treat=`)
    - first_treat=<col> -> CallawaySantAnna (takes `first_treat=`,
      does NOT take `treatment=`)
    """
    no_ft = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=False,
    )
    assert "DifferenceInDifferences" in no_ft["script"]
    assert "diff_diff.CallawaySantAnna().fit" not in no_ft["script"]

    with_ft = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        first_treat="cohort",
        verbose=False,
    )
    assert "diff_diff.CallawaySantAnna().fit" in with_ft["script"]
    # Staggered fit must not pass `treatment=` (would TypeError).
    step3_lines = [
        line for line in with_ft["script"].split("\n") if "CallawaySantAnna().fit" in line
    ]
    assert step3_lines, "Step 3 line missing"
    assert "treatment=" not in step3_lines[0]


def test_no_first_treat_step3_does_not_overclaim_match(df):
    """Without `first_treat`, the orchestrator cannot infer panel shape.

    The Step 3 label must NOT claim the emitted DiD example is "matched to
    your data shape" — for continuous-dose or heterogeneous-adoption
    designs without first_treat, DifferenceInDifferences would reject at
    fit time. The label must instead frame the example as conditional on
    the simple 2x2 case and tell the agent to substitute the matching
    candidate from Step 2 for other shapes.
    """
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=False,
    )
    script = out["script"]
    # Negative: should not claim universal match.
    assert "matched to your data shape" not in script
    # Positive: must qualify with the 2x2 conditionality + substitution hint.
    assert "2x2" in script
    assert "substitute" in script.lower()
    # Other workflow patterns must remain enumerated so the agent can substitute.
    for name in ("ContinuousDiD", "HeterogeneousAdoptionDiD"):
        assert name in script, f"{name} routing pattern missing from Step 2 hints"


def test_first_treat_step3_names_non_binary_alternatives(df):
    """first_treat alone doesn't pick the estimator.

    ContinuousDiD.fit and HeterogeneousAdoptionDiD (event-study) BOTH take
    `first_treat` (via `first_treat=` and `first_treat_col=`). The Step 3
    label must name those alternatives so an agent on a continuous-dose or
    heterogeneous-intensity panel isn't silently steered to CS21.
    """
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        first_treat="cohort",
        verbose=False,
    )
    script = out["script"]
    # The CS21 example remains the canonical binary-staggered demo.
    assert "diff_diff.CallawaySantAnna().fit" in script
    # The Step 3 commentary must name the non-binary alternatives so the
    # agent knows when to switch.
    assert "ContinuousDiD" in script
    assert "HeterogeneousAdoptionDiD" in script
    # The HAD distinction (first_treat_col vs first_treat) must be named
    # so the agent doesn't try to pass first_treat= to HAD's event study.
    assert "first_treat_col" in script


def test_emitted_script_parses_as_python_module(df):
    """The "script" output must parse as a complete Python module.

    Prior contract was "copy-pasteable" but the script had bare prose
    lines (`Step 1 - ...`) that would SyntaxError on execution. This
    locks the full-script parseability so the contract stays honest.
    """
    import ast

    for ft in (None, "cohort"):
        out = diff_diff.agent_workflow(
            df,
            unit="firm_id",
            time="year",
            treatment="treated",
            outcome="logwage",
            first_treat=ft,
            verbose=False,
        )
        # ast.parse with default mode='exec' verifies the WHOLE script
        # parses as a Python module, not just individual call expressions.
        ast.parse(out["script"])


def test_emitted_script_prints_report(df):
    """Step 5 must wrap BusinessReport in print() so end-to-end script
    execution actually produces the stakeholder narrative. full_report()
    returns a str; the previous template discarded it.
    """
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=False,
    )
    script = out["script"]
    assert "print(diff_diff.BusinessReport" in script
    assert ".full_report())" in script


def test_df_name_templates_into_script(df):
    """Caller can rename the dataframe symbol in the emitted script.

    Default (df_name="df"): script references `df`.
    Custom (df_name="panel"): every emitted call uses `panel` and no
    bare `df` identifier appears in the runnable code paths.
    """
    import ast

    out_default = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=False,
    )
    out_panel = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        df_name="panel",
        verbose=False,
    )
    # Default behavior preserved.
    assert "profile_panel(df," in out_default["script"]
    # Custom name flows through profile_call AND fit_example_call.
    assert "profile_panel(panel," in out_panel["script"]
    assert ".fit(panel," in out_panel["script"]
    # Static reference scan: parse the panel-script and confirm no `df`
    # Name node exists — catches template drift where a `df` reference
    # slips in outside the templated points.
    tree = ast.parse(out_panel["script"])
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "df" not in names, (
        f"emitted script with df_name='panel' still references `df`: "
        f"identifier names found = {sorted(names)}"
    )
    assert "panel" in names


def test_df_name_panel_script_executes_in_panel_namespace(df):
    """The emitted script must resolve all names in a namespace where
    `panel` exists and `df` does not. We stub out `diff_diff` with a
    MagicMock so calls don't actually fit; the test is purely about
    symbol resolution, not numerical correctness — if the script still
    referenced `df` anywhere in runnable code, exec() would NameError.
    """
    import unittest.mock

    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        df_name="panel",
        verbose=False,
    )
    ns = {
        "diff_diff": unittest.mock.MagicMock(),
        "panel": "sentinel_df_object",
        # Deliberately no `df` key — script must not reference it.
    }
    exec(compile(out["script"], "<test_df_name>", "exec"), ns)


def test_does_not_inspect_df():
    # Pure orchestrator: a structurally-empty DataFrame must still produce
    # the templated script (no df inspection happens).
    out = diff_diff.agent_workflow(
        pd.DataFrame(),
        unit="a",
        time="b",
        treatment="c",
        outcome="d",
        verbose=False,
    )
    assert "profile_panel" in out["script"]
    # repr() produces single-quoted literals.
    assert "unit='a'" in out["script"]


def test_emitted_calls_are_valid_python():
    """The advertised "copy-pasteable" script must actually parse as Python.

    Walks each line starting with `profile =` or `result =` and asserts the
    RHS parses with ast.parse(..., mode='eval'). Guards against future
    template drift that would silently emit invalid syntax.
    """
    import ast

    base = pd.DataFrame({"u": [1], "t": [0], "tr": [0], "y": [0.0]})
    for ft in (None, "cohort_col"):
        out = diff_diff.agent_workflow(
            base,
            unit="u",
            time="t",
            treatment="tr",
            outcome="y",
            first_treat=ft,
            verbose=False,
        )
        rhs_lines = []
        for line in out["script"].split("\n"):
            s = line.strip()
            if s.startswith("profile =") or s.startswith("result ="):
                rhs_lines.append(s[s.index("=") + 1 :].strip())
        assert rhs_lines, f"no parseable call lines emitted (first_treat={ft})"
        for rhs in rhs_lines:
            ast.parse(rhs, mode="eval")


@pytest.mark.parametrize(
    "label",
    [
        'firm"id',  # embedded double quote
        "year'col",  # embedded single quote
        "name\\with\\slash",  # backslashes
        "x\\nname",  # backslash-n (not a real newline)
        'unit"); evil()  #',  # injection attempt
        "with space",  # whitespace
    ],
)
def test_adversarial_column_labels_produce_valid_python(label):
    """Any str column label must produce a script that parses as Python.

    Uses repr() under the hood, so any input str becomes a valid Python
    string literal in the templated output. Locks the P0 contract that
    column names can never inject statements into the "copy-pasteable"
    script.
    """
    import ast

    df_local = pd.DataFrame({label: [1]} if " " not in label else {"u": [1]})
    out = diff_diff.agent_workflow(
        df_local,
        unit=label,
        time="t",
        treatment="tr",
        outcome="y",
        verbose=False,
    )
    for line in out["script"].split("\n"):
        s = line.strip()
        if s.startswith("profile =") or s.startswith("result ="):
            rhs = s[s.index("=") + 1 :].strip()
            ast.parse(rhs, mode="eval")


def test_fit_candidates_all_importable(df):
    """Every estimator name in fit_candidates must remain importable.

    Catches the drift case where an estimator is renamed but the
    orchestrator's candidates list still references the old name.
    """
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=False,
    )
    for name in out["fit_candidates"]:
        assert hasattr(diff_diff, name), (
            f"agent_workflow advertises {name!r} but it's not on the "
            f"public surface — rename detected without orchestrator update."
        )


def test_verbose_true_prints_script(df, capsys):
    out = diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "profile_panel" in captured.out
    assert out["script"] in captured.out


def test_verbose_false_silent(df, capsys):
    diff_diff.agent_workflow(
        df,
        unit="firm_id",
        time="year",
        treatment="treated",
        outcome="logwage",
        verbose=False,
    )
    captured = capsys.readouterr()
    assert captured.out == ""
