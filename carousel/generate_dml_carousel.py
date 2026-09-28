#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for the DMLDiD launch (v3.11.0).

Mirrors the POSTER-MODE architecture of ``generate_lwdid_carousel.py``
(magazine sidebar with progress tick, light gradient background, split-color
logo, footer wordmark, dark slides, soft card shadows; poster type floors:
body >= 16pt, captions >= 13pt, headlines 32-54pt, code >= 15pt; one idea +
one visual + roughly 25 words per slide).

Palette: "Circuit" - graphite/slate structure with a single SIGNAL GREEN
accent reserved for the ML/curved-truth motif (the deck's visual thesis:
the relationship the fixed-form model cannot bend to). The misspecified
linear fit is always warm gray. Two dark slides: the ridge twist (slide 4)
and the code slide (slide 9).

Narrative spine (ML-FORWARD hook - user decision 2026-08-29: lead with the
capability, then earn it with the confidently-wrong receipt):

1.  Cover          -- "Plug any ML model into your DiD." Curve-vs-line
                      motif (illustrative); subtitle names DMLDiD + the
                      3.11 release (version derived from pyproject, never
                      hardcoded).
2.  The bet        -- covariate adjustment is a modeling bet: classic DiD
                      estimators fix the nuisance forms (logit + linear
                      OLS); when assignment or trends bend in X, the bet
                      loses silently.
3.  The receipt    -- simulated staggered panel, truth known by
                      construction (said on-slide): linear reads 2.5909
                      (reported SE 0.0743) against a true 2.2388 - almost
                      five standard errors off, behind a narrow nominal
                      interval (nominal-inference caveat on-slide).
4.  Twist (DARK)   -- ridge won't save you: 2.5898. A penalty on the wrong
                      functional form is still the wrong functional form.
5.  The paper      -- Chang (2020), The Econometrics Journal: two ideas -
                      Neyman-orthogonal scores + cross-fitting.
                      ATTRIBUTED PARAPHRASE ONLY, no quote marks: the
                      paper review (docs/methodology/papers/
                      chang-2020-review.md) is pinned to the arXiv v3
                      layout and the published PDF has not been
                      cross-checked word-for-word, so this deck carries NO
                      verbatim pull quote (unlike the CiC deck, whose
                      quote was verified against its published PDF).
6.  The math       -- the Case 1 score annotated term-by-term (matplotlib
                      mathtext, big type), plus the two one-liners:
                      orthogonality (nuisance error enters at second
                      order) and cross-fitting (no unit scored by a
                      learner that saw it).
7.  The payoff     -- the four-learner chart: linear/ridge miss high,
                      sieve 2.2804 and a hand-rolled PolynomialRidge
                      2.2818 recover the truth line. The double-robustness
                      beat rides ON the slide: the propensity model is
                      misspecified in every arm; the flexible OUTCOME
                      model alone rescues the point estimate.
8.  When to use it -- the honest decision split (mirrors
                      docs/choosing_estimator.rst): nonlinear /
                      high-dimensional covariates -> DMLDiD; a plausible
                      logit + linear spec -> CallawaySantAnna (fewer
                      moving parts). Closing band (user emphasis,
                      2026-08-29): four learners BUILT IN with no extra
                      installs, or bring your own - scikit-learn
                      REGRESSORS already fit the duck-typed
                      fit()/predict() outcome contract, and propensity
                      CLASSIFIERS additionally need predict_proba()
                      (both contracts stated on-slide).
9.  Code (DARK)    -- the tutorial's own fit (sieve, n_folds=5, seed=42),
                      >= 15pt, plus the bring-your-own swap rendered as a
                      real sklearn import + constructor call.
10. Production     -- feature grid (post-fit staggered surface: event
                      study / group / HonestDiD / sup-t bands; survey
                      designs + clustering with PSU-cohesive folds;
                      repeated cross-sections; per-cell cross-fit
                      diagnostics) + validation strip (Chang Case 1 score
                      anchored to DoubleML at machine precision; committed
                      spike).
11. CTA            -- pip install; tutorial 32 mentioned ONCE, here.

Claim discipline (verified against the committed executed notebook
``docs/tutorials/32_dml_did.ipynb`` and ``docs/methodology/REGISTRY.md``):

- EVERY estimate on the deck is a committed, seed-locked tutorial-32 value
  from a SIMULATED example with truth known by construction - labeled
  on-slide wherever a truth value appears. notebook <-> deck sync is
  pinned by ``tests/test_dml_carousel_claims.py`` (parses this module's
  constants via ``ast`` and locates each on the committed notebook
  surface). The 30-seed robustness sweep from prototyping is deliberately
  NOT on the deck (it is not on the committed notebook surface).
- "Almost five standard errors" mirrors the tutorial's own wording and is
  a DISTANCE in reported-SE units; the deck never presents the simulated
  SEs as valid coverage. The rate-condition caveat is SLIDE-LOCAL wherever
  reported uncertainty appears (review round 1): slide 3 labels the value
  "reported SE" / "narrow nominal interval" and carries the
  nominal-not-theory-backed caption, slide 4 says "nominal precision" with
  its own qualifier line, and the payoff slide keeps the
  illustrative-inference caption.
- The math slide labels the displayed expression as the UNCENTERED summand
  ``s`` (its cell mean is the ATT; the centered eq. 3.1 score is
  ``psi = s - ATT`` - said in the caption), and scopes orthogonality to
  the LEARNED nuisances g and l only (the treated share p carries the
  Theorem 2 augmented-score variance correction, per the REGISTRY
  chang_panel_score note) - review round 1 correction.
- The repeated-cross-section card states Chang Assumption 2.3's
  same-target-population requirement ON the card ("fresh samples from the
  SAME target population each wave") - review round 1 correction.
- The COVER motif is stylized art (a curved relationship vs a straight
  fit), labeled "illustrative" on-slide; it plots no tutorial data.
- CallawaySantAnna is named twice, neutrally and honestly: slide 2 (the
  fixed-form nuisances it and classic DR estimators use - stated as a
  design choice, not a defect) and slide 8 (the choosing_estimator
  decision rule RECOMMENDING it when a parametric spec is plausible). No
  "only"/"first" absolutes anywhere (claims test bans them). No external
  competitor is named.
- The math slide renders the ``chang_panel_score`` summand exactly as the
  REGISTRY documents it (Equation 3.1 as implemented); g-hat is the
  conditional propensity, p-hat the unconditional treated share (the
  REGISTRY notation split).
- The DoubleML anchor is scoped ON-SLIDE to the Case 1 score (committed
  spike ``benchmarks/doubleml/chang_case1_parity.py``, ATT diff 4.4e-16
  per the REGISTRY DR-score families note) - never an end-to-end
  staggered-parity claim.
- Survey/cluster support and the staggered ATT(g,t) construction are
  documented library extensions of Chang (REGISTRY DMLDiD Notes); the
  production slide says "documented extension" rather than attributing
  them to the paper.
- The learner-flexibility claims are library-surface facts, not
  comparisons: the four built-in learners live in ``diff_diff/_learners.py``
  on the library's own numpy/scipy solvers (no additional dependency - the
  claims test asserts the module imports no external ML package), and
  scikit-learn compatibility is the documented duck-typed protocol
  (regressor fit()/predict(); classifier fit()/predict_proba())
  (``docs/api/dml_did.rst`` names scikit-learn; sklearn is an ecosystem
  library the estimator interoperates with, never a compared-against
  competitor). The sklearn import on the code slide is illustrative of the
  contract; the notebook's own custom-learner demo is a numpy-only object
  (the notebooks CI environment deliberately excludes sklearn).
- Tutorial 32 is referenced ONCE, on the CTA slide (house precedent).

Run with::

    python carousel/generate_dml_carousel.py

Produces ``carousel/diff-diff-dml-carousel.pdf``. Generation requires
``fpdf2``, ``Pillow``, and ``matplotlib`` (carousel-only dependencies, not
part of the library's install extras).
"""

import os
import re
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from fpdf import FPDF  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

# Page dimensions (4:5 portrait -- same as the other decks)
WIDTH = 270  # mm
HEIGHT = 337.5  # mm

# Version label derived from pyproject.toml (regex, not tomllib: py>=3.9).
_PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"
_m = re.search(r'^version\s*=\s*"([^"]+)"', _PYPROJECT.read_text(encoding="utf-8"), re.MULTILINE)
if _m is None:
    raise RuntimeError(f"could not parse project version from {_PYPROJECT}")
VERSION_LABEL = "v" + _m.group(1)
RELEASE_LABEL = ".".join(_m.group(1).split(".")[:2])

TOTAL_SLIDES = 11

# -------------------------------------------------------------------------
# "Circuit" palette -- graphite/slate structure, SIGNAL GREEN = the ML /
# curved-truth motif. The wrong linear fit is always warm gray.
# -------------------------------------------------------------------------
GRAPHITE = (30, 41, 59)  # #1e293b  primary accent / structure
SLATE = (51, 65, 85)  # #334155
SIGNAL = (22, 163, 74)  # #16a34a  THE CURVED TRUTH / ML (chart grammar)
SIGNAL_BRIGHT = (74, 222, 128)  # #4ade80  signal on the dark slides
SIGNAL_TINT = (240, 253, 244)  # #f0fdf4  gradient start
MINT = (187, 247, 208)  # #bbf7d0  light fills / dark-slide text pop
WRONG_GRAY = (120, 113, 108)  # #78716c  the misspecified fit (warm gray)
SHADOW = (203, 213, 225)  # #cbd5e1  soft card shadow

# Text + structural (shared with the other decks for legibility)
NAVY = (15, 23, 42)  # #0f172a  primary text; dark-slide gradient start
GRAY = (100, 116, 139)  # #64748b
LIGHT_GRAY = (148, 163, 184)  # #94a3b8
WHITE = (255, 255, 255)
DEEP_GREEN = (20, 46, 34)  # #142e22  dark-slide gradient end
PANEL_NAVY = (30, 41, 59)  # #1e293b  code panel fill
GREEN_CODE = (74, 222, 128)  # #4ade80  code string/number literals
SLATE_CODE = (148, 163, 184)  # #94a3b8

GRAPHITE_HEX = "#1e293b"
SIGNAL_HEX = "#16a34a"
SIGNAL_BRIGHT_HEX = "#4ade80"
MINT_HEX = "#bbf7d0"
WRONG_GRAY_HEX = "#78716c"
NAVY_HEX = "#0f172a"
GRAY_HEX = "#64748b"
LIGHT_GRAY_HEX = "#94a3b8"

# -------------------------------------------------------------------------
# Seed-locked numbers from the committed executed tutorial
# (docs/tutorials/32_dml_did.ipynb). Every constant below is located on the
# committed notebook surface by tests/test_dml_carousel_claims.py - never
# edit one without re-running it. The example is SIMULATED with truth known
# by construction (stated on every slide that shows a truth value).
# -------------------------------------------------------------------------

TRUTH = 2.2388  # DGP-implied overall ATT (the fit's own aggregation weights)
LINEAR = (2.5909, 0.0743)  # outcome_learner="linear": ATT, SE
RIDGE = (2.5898, 0.0736)  # outcome_learner="ridge"
SIEVE = (2.2804, 0.0438)  # outcome_learner="sieve"
POLY = (2.2818, 0.0438)  # hand-rolled duck-typed PolynomialRidge object

N_UNITS, N_PERIODS = 600, 6
N_NEVER = 354  # never-treated units in the simulated panel
COHORTS = (4, 5)

# Survey lane provenance (RCS + SurveyDesign fit)
SURVEY_DF, SURVEY_PSU, SURVEY_STRATA = 16, 20, 4

# Tutorial fit config rendered on the code slide (claims test pins these
# against the notebook's code cells)
CODE_LEARNER, CODE_FOLDS, CODE_SEED = "sieve", 5, 42

# DoubleML anchor (REGISTRY DR-score families note; committed spike
# benchmarks/doubleml/chang_case1_parity.py) - Case 1 SCORE parity only.
DOUBLEML_ATT_DIFF = "4.4e-16"


class DMLCarouselPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format=(WIDTH, HEIGHT))
        self.set_auto_page_break(False)
        self._temp_files = []

    def cleanup(self):
        for f in self._temp_files:
            try:
                os.unlink(f)
            except OSError:
                pass

    # -----------------------------------------------------------------
    # Magazine vertical sidebar -- GRAPHITE bar, SIGNAL progress tick.
    # -----------------------------------------------------------------
    def _draw_vertical_sidebar(self, slide_number, total=TOTAL_SLIDES, dark=False):
        bar_x = 14
        bar_y_top = 45
        bar_y_bottom = 275
        self.set_draw_color(*(LIGHT_GRAY if dark else GRAPHITE))
        self.set_line_width(0.6)
        self.line(bar_x, bar_y_top, bar_x, bar_y_bottom)

        ratio = (slide_number - 1) / (total - 1) if total > 1 else 0.0
        tick_y = bar_y_top + ratio * (bar_y_bottom - bar_y_top)
        self.set_draw_color(*(SIGNAL_BRIGHT if dark else SIGNAL))
        self.set_line_width(1.2)
        self.line(bar_x - 4, tick_y, bar_x + 7, tick_y)

    # -----------------------------------------------------------------
    # Backgrounds + footer
    # -----------------------------------------------------------------
    def light_gradient_background(self):
        """Green tint #f0fdf4 fading to white."""
        steps = 50
        r0, g0, b0 = SIGNAL_TINT
        for i in range(steps):
            ratio = i / steps
            self.set_fill_color(
                int(r0 + (255 - r0) * ratio),
                int(g0 + (255 - g0) * ratio),
                int(b0 + (255 - b0) * ratio),
            )
            y = i * HEIGHT / steps
            self.rect(0, y, WIDTH, HEIGHT / steps + 1, "F")

    def dark_gradient_background(self):
        """Near-black navy #0f172a fading to deep green #142e22."""
        steps = 50
        r0, g0, b0 = NAVY
        r1, g1, b1 = DEEP_GREEN
        for i in range(steps):
            ratio = i / steps
            self.set_fill_color(
                int(r0 + (r1 - r0) * ratio),
                int(g0 + (g1 - g0) * ratio),
                int(b0 + (b1 - b0) * ratio),
            )
            y = i * HEIGHT / steps
            self.rect(0, y, WIDTH, HEIGHT / steps + 1, "F")

    def add_footer(self, dark=False):
        """Centered split-color ``diff-diff vX.Y.Z`` wordmark."""
        self.set_font("Helvetica", "B", 12)
        dd_text = "diff-diff "
        v_text = VERSION_LABEL
        dd_w = self.get_string_width(dd_text)
        v_w = self.get_string_width(v_text)
        start_x = (WIDTH - dd_w - v_w) / 2

        self.set_xy(start_x, HEIGHT - 18)
        self.set_text_color(*(LIGHT_GRAY if dark else GRAY))
        self.cell(dd_w, 10, dd_text)
        self.set_text_color(*(SIGNAL_BRIGHT if dark else SIGNAL))
        self.cell(v_w, 10, v_text)

    # -----------------------------------------------------------------
    # Text helpers
    # -----------------------------------------------------------------
    def centered_text(self, y, text, size=28, bold=True, color=NAVY, italic=False):
        self.set_xy(0, y)
        style = ("B" if bold else "") + ("I" if italic else "")
        self.set_font("Helvetica", style, size)
        self.set_text_color(*color)
        self.cell(WIDTH, size * 0.5, text, align="C")

    def _kicker(self, y, text, color=SIGNAL):
        """Editorial section label: letter-spaced caps with flanking rules."""
        spaced = " ".join(text.upper())
        self.set_font("Helvetica", "B", 14)
        tw = self.get_string_width(spaced)
        mid_y = y + 3
        rule = 20
        gap = 8
        self.set_draw_color(*color)
        self.set_line_width(0.7)
        self.line(WIDTH / 2 - tw / 2 - gap - rule, mid_y, WIDTH / 2 - tw / 2 - gap, mid_y)
        self.line(WIDTH / 2 + tw / 2 + gap, mid_y, WIDTH / 2 + tw / 2 + gap + rule, mid_y)
        self.set_xy(0, y)
        self.set_text_color(*color)
        self.cell(WIDTH, 6, spaced, align="C")

    def draw_split_logo(self, y, size=18):
        """Split-color diff-diff logo with SIGNAL middle dash."""
        self.set_xy(0, y)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="R")
        self.set_text_color(*SIGNAL)
        self.cell(10, 10, "-", align="C")
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="L")

    # -----------------------------------------------------------------
    # Shadowed card helper (poster mode: used sparingly, big type inside)
    # -----------------------------------------------------------------
    def _shadow_rect(self, x, y, w, h):
        self.set_fill_color(*SHADOW)
        self.rect(x + 1.4, y + 1.4, w, h, "F")

    def _stat_card(self, x, y, w, h, headline, sub_lines, accent, headline_size=40):
        """One shadowed stat card: big number + caption lines (>= 14pt)."""
        self._shadow_rect(x, y, w, h)
        self.set_fill_color(*WHITE)
        self.set_draw_color(220, 220, 220)
        self.set_line_width(0.5)
        self.rect(x, y, w, h, "DF")
        self.set_fill_color(*accent)
        self.rect(x, y, w, 3.2, "F")

        self.set_xy(x, y + 13)
        self.set_font("Helvetica", "B", headline_size)
        self.set_text_color(*accent)
        self.cell(w, 16, headline, align="C")

        ly = y + 36
        for line, emphasize in sub_lines:
            self.set_xy(x + 6, ly)
            self.set_font("Helvetica", "B" if emphasize else "", 14)
            self.set_text_color(*(NAVY if emphasize else GRAY))
            self.cell(w - 12, 8, line, align="C")
            ly += 10.5

    # -----------------------------------------------------------------
    # Figure helpers (matplotlib -> PNG -> fpdf image)
    # -----------------------------------------------------------------
    def _save_fig(self, fig, dpi=200, transparent=False, facecolor="white"):
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        self._temp_files.append(path)
        try:
            fig.savefig(
                path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.1,
                transparent=transparent,
                facecolor=None if transparent else facecolor,
            )
        finally:
            plt.close(fig)
        with PILImage.open(path) as img:
            pw, ph = img.size
        return path, pw, ph

    # -----------------------------------------------------------------
    # Cover motif -- a curved covariate relationship (signal green) that
    # a straight gray fit cannot bend to. Stylized art, not a data claim
    # (it plots no tutorial data) - labeled "illustrative" on-slide.
    # -----------------------------------------------------------------
    def _render_cover_motif(self):
        rng = np.random.default_rng(2020)
        x = np.linspace(-2.6, 2.6, 140)
        truth = 1.6 * (x**2 - 1.0) * 0.55 + 0.6

        fig, ax = plt.subplots(figsize=(10, 3.7))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        xs = rng.uniform(-2.6, 2.6, 170)
        ys = 1.6 * (xs**2 - 1.0) * 0.55 + 0.6 + rng.normal(0, 0.42, xs.size)
        ax.plot(xs, ys, "o", color=GRAPHITE_HEX, markersize=5.5, alpha=0.30)

        coef = np.polyfit(xs, ys, 1)
        ax.plot(
            x,
            np.polyval(coef, x),
            color=WRONG_GRAY_HEX,
            linewidth=3.2,
            linestyle=(0, (6, 3)),
        )
        ax.plot(x, truth, color=SIGNAL_HEX, linewidth=4.2, zorder=5)

        ax.text(
            2.68,
            float(truth[-1]) - 0.15,
            "what an ML learner\ncan fit",
            fontsize=14,
            color=SIGNAL_HEX,
            fontweight="bold",
            va="center",
        )
        ax.text(
            2.68,
            float(np.polyval(coef, 2.6)) - 1.05,
            "what a linear\nmodel fits",
            fontsize=13.5,
            color=WRONG_GRAY_HEX,
            fontweight="bold",
            va="center",
        )
        ax.text(
            3.9,
            -1.62,
            "illustrative",
            fontsize=15,
            color=LIGHT_GRAY_HEX,
            style="italic",
            ha="right",
        )

        ax.set_xlim(-2.8, 4.0)
        ax.set_ylim(-1.75, 3.3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=0.2)
        return self._save_fig(fig, transparent=True)

    # -----------------------------------------------------------------
    # Math slide equation (matplotlib mathtext -> crisp big type)
    # -----------------------------------------------------------------
    def _render_score_equation(self, dark=False):
        """Equation composed from measured fragments, with the second
        fraction built MANUALLY (numerator text, drawn bar, denominator
        text) so the annotation arrows can anchor to center-placed text
        objects instead of mathtext bounding boxes - mathtext bboxes carry
        asymmetric slop (worst around \\left[...\\right]), which pushed
        every arrow visibly right of its target in the first two cuts."""
        fg = "white" if dark else NAVY_HEX
        accent = SIGNAL_BRIGHT_HEX if dark else SIGNAL_HEX
        gray = LIGHT_GRAY_HEX if dark else GRAY_HEX

        fig, ax = plt.subplots(figsize=(10.8, 4.1))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        y_eq = 6.1
        frags = [
            ("prefix", r"$s \,=$"),
            ("bracket", r"$\left[\,D \;-\; \hat g(X)\,\frac{1-D}{1-\hat g(X)}\,\right]$"),
            ("dot", r"$\cdot$"),
        ]
        texts = {}
        cursor = 0.0
        lead_gap = {"prefix": 0.0, "bracket": -0.12, "dot": 0.18}
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        boxes = {}
        for name, tex in frags:
            t = ax.text(
                cursor + lead_gap[name], y_eq, tex, fontsize=30, color=fg, ha="left", va="center"
            )
            fig.canvas.draw()
            bb = t.get_window_extent(renderer=renderer)
            (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            texts[name] = t
            boxes[name] = [x0, y0, x1, y1]
            cursor = x1

        # Manual fraction: numerator / bar / p-hat, all centered on one x
        # so the arrows can anchor EXACTLY (ha="center" text at a known x).
        num_t = ax.text(
            cursor + 0.35,
            y_eq + 0.28,
            r"$\Delta Y \;-\; \hat\ell(X)$",
            fontsize=25,
            color=fg,
            ha="left",
            va="bottom",
        )
        fig.canvas.draw()
        bb = num_t.get_window_extent(renderer=renderer)
        (nx0, _ny0), (nx1, _ny1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        frac_cx = (nx0 + nx1) / 2
        bar_pad = 0.12
        bar = ax.plot(
            [nx0 - bar_pad, nx1 + bar_pad],
            [y_eq + 0.08, y_eq + 0.08],
            color=fg,
            linewidth=2.0,
            solid_capstyle="butt",
        )[0]
        den_t = ax.text(
            frac_cx, y_eq - 0.16, r"$\hat p$", fontsize=25, color=fg, ha="center", va="top"
        )
        frac_parts = [num_t, bar, den_t]

        # Recenter the assembled equation on x = 5.
        span_mid = (boxes["prefix"][0] + (nx1 + bar_pad)) / 2
        shift = 5.0 - span_mid
        for name, _ in frags:
            x_old, y_old = texts[name].get_position()
            texts[name].set_position((x_old + shift, y_old))
            boxes[name][0] += shift
            boxes[name][2] += shift
        for artist in frac_parts:
            if artist is bar:
                xs = bar.get_xdata()
                bar.set_xdata([xs[0] + shift, xs[1] + shift])
            else:
                x_old, y_old = artist.get_position()
                artist.set_position((x_old + shift, y_old))
        frac_cx += shift
        nx0 += shift
        nx1 += shift

        # Bracket arrow center: the bbox RIGHT edge carries \right]-related
        # slop, so use the reliable left edge and the dot fragment's left
        # edge (which visually abuts the closing bracket) as the span.
        bracket_cx = (boxes["bracket"][0] + boxes["dot"][0] - lead_gap["dot"]) / 2
        bracket_bottom = boxes["bracket"][1]

        pad = 0.45
        arrow_kw = dict(arrowstyle="->", color=accent, lw=1.8)
        # propensity weight -> the bracket, from below
        ax.annotate(
            "",
            xy=(bracket_cx, bracket_bottom - 0.15),
            xytext=(bracket_cx, bracket_bottom - pad - 0.9),
            arrowprops=arrow_kw,
        )
        ax.text(
            bracket_cx,
            bracket_bottom - pad - 1.05,
            "propensity weight\n(treated vs reweighted controls)",
            fontsize=13.5,
            color=gray,
            ha="center",
            va="top",
        )
        # outcome-model residual -> the numerator, from above
        num_top = y_eq + 1.22
        ax.annotate(
            "",
            xy=(frac_cx, num_top + 0.1),
            xytext=(frac_cx, num_top + pad + 0.85),
            arrowprops=arrow_kw,
        )
        ax.text(
            frac_cx,
            num_top + pad + 1.0,
            "outcome-model residual\n(what the trend model missed)",
            fontsize=13.5,
            color=gray,
            ha="center",
            va="bottom",
        )
        # treated share -> p-hat, from below (den_t is ha="center" at
        # frac_cx, so this arrow hits the glyph center exactly)
        den_bottom = y_eq - 1.28
        ax.annotate(
            "",
            xy=(frac_cx, den_bottom + 0.1),
            xytext=(frac_cx, den_bottom - pad - 0.75),
            arrowprops=arrow_kw,
        )
        ax.text(
            frac_cx,
            den_bottom - pad - 0.9,
            "treated share\n(normalizer)",
            fontsize=13.5,
            color=gray,
            ha="center",
            va="top",
        )

        ax.text(
            5.0,
            0.15,
            r"$\hat g(X)$: propensity learner    "
            r"$\hat\ell(X)$: outcome learner    "
            r"$\hat p$: treated share",
            fontsize=14.5,
            color=gray,
            ha="center",
        )
        fig.tight_layout(pad=0.2)
        return self._save_fig(fig, transparent=True)

    # -----------------------------------------------------------------
    # Payoff chart -- the four learner estimates vs the truth line.
    # ONLY committed tutorial values (point estimates + 1.96*SE whiskers,
    # labeled illustrative-inference on-slide).
    # -----------------------------------------------------------------
    def _render_payoff_chart(self):
        fig, ax = plt.subplots(figsize=(10.2, 4.1))

        learners = [
            ("linear", LINEAR, WRONG_GRAY_HEX),
            ("ridge", RIDGE, WRONG_GRAY_HEX),
            ("sieve", SIEVE, SIGNAL_HEX),
            ("custom\nPolynomialRidge", POLY, SIGNAL_HEX),
        ]
        xs = np.arange(len(learners))
        ax.axhline(TRUTH, color=GRAPHITE_HEX, linewidth=2.2, linestyle=(0, (5, 3)))
        ax.text(
            3.52,
            TRUTH - 0.028,
            f"true effect {TRUTH:.4f}\n(simulated - known\nby construction)",
            fontsize=13.5,
            color=GRAPHITE_HEX,
            fontweight="bold",
            va="top",
        )

        for i, (name, (att, se), color) in enumerate(learners):
            ax.errorbar(
                [i],
                [att],
                yerr=[1.96 * se],
                fmt="o",
                color=color,
                markersize=14,
                capsize=7,
                linewidth=2.8,
                capthick=2.4,
            )
            ax.annotate(
                f"{att:.4f}",
                xy=(i, att),
                xytext=(i + 0.09, att + 0.035),
                fontsize=15,
                fontweight="bold",
                color=color,
            )

        ax.set_xticks(xs)
        ax.set_xticklabels([n for n, _, _ in learners], fontsize=14.5, color=NAVY_HEX)
        ax.set_ylabel("estimated overall ATT", fontsize=14, color=NAVY_HEX)
        ax.tick_params(axis="y", labelsize=13, colors=GRAY_HEX)
        ax.set_xlim(-0.5, 4.6)
        ax.set_ylim(2.1, 2.82)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("bottom", "left"):
            ax.spines[s].set_color(LIGHT_GRAY_HEX)
        fig.tight_layout(pad=0.3)
        return self._save_fig(fig, facecolor="white")

    # -----------------------------------------------------------------
    # Code-block helper (dark panel with token-colored lines)
    # -----------------------------------------------------------------
    def _add_code_block(
        self, x, y, w, token_lines, font_size=15, line_height=12.5, fill=PANEL_NAVY
    ):
        n_lines = len(token_lines)
        total_h = n_lines * line_height + 26

        self.set_fill_color(*fill)
        self.rect(x, y, w, total_h, "F")

        self.set_font("Courier", "", font_size)
        char_w = self.get_string_width("M")

        pad_x = 14
        pad_y = 13
        for i, tokens in enumerate(token_lines):
            cx = x + pad_x
            cy = y + pad_y + i * line_height
            for text, color in tokens:
                if not text:
                    continue
                self.set_xy(cx, cy)
                self.set_text_color(*color)
                self.cell(char_w * len(text), 10, text)
                cx += char_w * len(text)
        return total_h

    # -----------------------------------------------------------------
    # Slides
    # -----------------------------------------------------------------
    def slide_01_cover(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(1)

        self.draw_split_logo(26, size=34)

        self.centered_text(60, "Plug any ML model", size=52)
        self.centered_text(90, "into your DiD.", size=52, color=SIGNAL)

        motif_path, _pw, _ph = self._render_cover_motif()
        motif_w = 224
        self.image(motif_path, (WIDTH - motif_w) / 2, 122, motif_w)

        self.centered_text(
            212,
            f"New in diff-diff {RELEASE_LABEL}: the DMLDiD estimator.",
            size=22,
            color=GRAPHITE,
        )
        self.centered_text(
            230,
            "Double/debiased machine learning DiD (Chang 2020).",
            size=19,
            bold=False,
        )
        self.add_footer()

    def slide_02_bet(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(2)

        self._kicker(42, "Why You'd Want This")

        self.centered_text(78, "Covariate adjustment", size=44)
        self.centered_text(104, "is a modeling bet.", size=44, color=GRAPHITE)

        self.set_xy(28, 138)
        self.set_font("Helvetica", "", 20)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 56,
            11,
            "Classic doubly-robust DiD (CallawaySantAnna and friends) fixes the "
            "nuisance models up front: a logit for treatment, a linear "
            "regression for trends.",
            align="C",
        )

        self.set_draw_color(*SIGNAL)
        self.set_line_width(1.4)
        self.line(WIDTH / 2 - 30, 186, WIDTH / 2 + 30, 186)

        self.set_xy(28, 200)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*GRAPHITE)
        self.multi_cell(
            WIDTH - 56,
            12,
            "When treatment or trends bend in X, that bet loses - and nothing "
            "in the output tells you.",
            align="C",
        )
        self.add_footer()

    def slide_03_receipt(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(3)

        self._kicker(36, "The Receipt")

        self.centered_text(62, "Confidently wrong.", size=48)

        card_w, card_h = 108, 78
        gap = 14
        left_x = (WIDTH - 2 * card_w - gap) / 2
        y0 = 100
        self._stat_card(
            left_x,
            y0,
            card_w,
            card_h,
            f"{LINEAR[0]:.4f}",
            [
                (f"reported SE {LINEAR[1]:.4f}", True),
                ("linear outcome model -", False),
                ("a narrow nominal interval", False),
            ],
            WRONG_GRAY,
        )
        self._stat_card(
            left_x + card_w + gap,
            y0,
            card_w,
            card_h,
            f"{TRUTH:.4f}",
            [
                ("the true effect", True),
                ("simulated staggered panel -", False),
                ("truth known by construction", False),
            ],
            SIGNAL,
        )

        self.centered_text(198, "Almost five standard errors off.", size=30, color=GRAPHITE)

        self.set_xy(30, 222)
        self.set_font("Helvetica", "", 18)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 60,
            9.5,
            f"{N_UNITS} units, {N_PERIODS} periods, cohorts treated at t = {COHORTS[0]} "
            f"and {COHORTS[1]}, {N_NEVER} never treated. Assignment and trends both "
            "bend in X - the linear model cannot. (SEs here are nominal, not "
            "theory-backed: the DGP deliberately breaks the rate conditions.)",
            align="C",
        )
        self.add_footer()

    def slide_04_twist(self):
        self.add_page()
        self.dark_gradient_background()
        self._draw_vertical_sidebar(4, dark=True)

        self._kicker(40, "It Gets Worse", color=SIGNAL_BRIGHT)

        self.centered_text(84, "Ridge won't save you.", size=48, color=WHITE)

        self.centered_text(136, f"{RIDGE[0]:.4f}", size=64, color=LIGHT_GRAY)
        self.centered_text(
            166,
            f"reported SE {RIDGE[1]:.4f}  -  same miss, same nominal precision",
            size=19,
            bold=False,
            color=LIGHT_GRAY,
        )

        self.set_xy(30, 204)
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(*MINT)
        self.multi_cell(
            WIDTH - 60,
            13,
            "A penalty on the wrong functional form\nis still the wrong functional form.",
            align="C",
        )

        self.set_xy(30, 250)
        self.set_font("Helvetica", "", 16)
        self.set_text_color(*LIGHT_GRAY)
        self.multi_cell(
            WIDTH - 60,
            8.5,
            "Regularization tunes a model family. It cannot leave the family. "
            "(Reported SEs are nominal - illustrative under this DGP.)",
            align="C",
        )
        self.add_footer(dark=True)

    def slide_05_paper(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(5)

        self._kicker(38, "The Paper")

        self.centered_text(66, "Chang (2020):", size=40)
        self.centered_text(92, "stop betting. Learn them.", size=40, color=SIGNAL)

        y0 = 128
        for num, head, body in [
            (
                "1",
                "Neyman-orthogonal scores",
                "built so small nuisance errors hit the estimate only at second order",
            ),
            (
                "2",
                "Cross-fitting",
                "nuisances learned on K-1 folds, applied to the held-out fold - "
                "flexible learners without overfitting bias",
            ),
        ]:
            self.set_fill_color(*SIGNAL)
            self.set_xy(34, y0)
            self.set_font("Helvetica", "B", 26)
            self.set_text_color(*SIGNAL)
            self.cell(14, 12, num)
            self.set_xy(52, y0)
            self.set_font("Helvetica", "B", 23)
            self.set_text_color(*NAVY)
            self.cell(WIDTH - 86, 12, head)
            self.set_xy(52, y0 + 14)
            self.set_font("Helvetica", "", 17)
            self.set_text_color(*GRAY)
            self.multi_cell(WIDTH - 92, 9, body)
            y0 += 52

        self.set_xy(30, 244)
        self.set_font("Helvetica", "I", 15)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 60,
            8,
            "Double/debiased machine learning for difference-in-differences models "
            "- The Econometrics Journal, 23(2).",
            align="C",
        )
        self.add_footer()

    def slide_06_math(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(6)

        self._kicker(34, "The Math")

        self.centered_text(58, "One orthogonal score per cell.", size=36)

        eq_path, _pw, _ph = self._render_score_equation()
        eq_w = 232
        self.image(eq_path, (WIDTH - eq_w) / 2, 84, eq_w)

        y0 = 182
        for head, body in [
            (
                "Orthogonal.",
                "The score's derivative in the LEARNED nuisances g and l is "
                "zero at the truth - learner error is second-order. (The "
                "treated share p gets its own variance correction.)",
            ),
            (
                "Cross-fit.",
                "No unit is scored by a learner that trained on it - "
                "flexibility without self-overfitting.",
            ),
        ]:
            self.set_xy(34, y0)
            self.set_font("Helvetica", "B", 20)
            self.set_text_color(*SIGNAL)
            head_w = self.get_string_width(head) + 4
            self.cell(head_w, 10, head)
            self.set_xy(34 + head_w, y0)
            self.set_font("Helvetica", "", 17)
            self.set_text_color(*NAVY)
            self.multi_cell(WIDTH - 68 - head_w, 9.4, body)
            y0 += 34

        self.set_xy(30, 256)
        self.set_font("Helvetica", "I", 13.5)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 60,
            7,
            "Chang eq. 3.1 (panel case) as implemented: the mean of the summand "
            "s over a (cohort, period) cell is that cell's ATT; the centered "
            "score is psi = s - ATT.",
            align="C",
        )
        self.add_footer()

    def slide_07_payoff(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(7)

        self._kicker(34, "The Payoff")

        self.centered_text(58, "Same data. Same propensity.", size=32)
        self.centered_text(78, "Flexible outcome model: truth.", size=32, color=SIGNAL)

        chart_path, _pw, _ph = self._render_payoff_chart()
        chart_w = 226
        self.image(chart_path, (WIDTH - chart_w) / 2, 98, chart_w)

        self.set_xy(28, 232)
        self.set_font("Helvetica", "", 16.5)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 56,
            9,
            "That is double robustness doing real work: the propensity model is "
            "misspecified in every arm - a flexible outcome model alone rescues "
            "the point estimate.",
            align="C",
        )
        self.set_xy(28, 262)
        self.set_font("Helvetica", "I", 13)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 56,
            7,
            "Seed-locked simulated example; whiskers are 1.96 SE, illustrative "
            "under the deliberately misspecified propensity.",
            align="C",
        )
        self.add_footer()

    def slide_08_when(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(8)

        self._kicker(36, "When To Reach For It")

        col_w = 106
        gap = 16
        left_x = (WIDTH - 2 * col_w - gap) / 2
        y0 = 66
        col_h = 118

        self._shadow_rect(left_x, y0, col_w, col_h)
        self.set_fill_color(*WHITE)
        self.set_draw_color(220, 220, 220)
        self.rect(left_x, y0, col_w, col_h, "DF")
        self.set_fill_color(*SIGNAL)
        self.rect(left_x, y0, col_w, 3.2, "F")
        self.set_xy(left_x, y0 + 10)
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(*SIGNAL)
        self.cell(col_w, 12, "DMLDiD", align="C")
        self.set_xy(left_x + 8, y0 + 28)
        self.set_font("Helvetica", "", 16)
        self.set_text_color(*NAVY)
        self.multi_cell(
            col_w - 16,
            9,
            "Covariates with nonlinear or high-dimensional relationships to "
            "treatment or trends.",
            align="C",
        )
        self.set_xy(left_x + 8, y0 + 82)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*GRAPHITE)
        self.multi_cell(
            col_w - 16,
            8.5,
            "Bring your own learners:\nregressors fit() / predict(),\nclassifiers predict_proba().",
            align="C",
        )

        rx = left_x + col_w + gap
        self._shadow_rect(rx, y0, col_w, col_h)
        self.set_fill_color(*WHITE)
        self.set_draw_color(220, 220, 220)
        self.rect(rx, y0, col_w, col_h, "DF")
        self.set_fill_color(*GRAPHITE)
        self.rect(rx, y0, col_w, 3.2, "F")
        self.set_xy(rx, y0 + 10)
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(*GRAPHITE)
        self.cell(col_w, 12, "CallawaySantAnna", align="C")
        self.set_xy(rx + 8, y0 + 28)
        self.set_font("Helvetica", "", 16)
        self.set_text_color(*NAVY)
        self.multi_cell(
            col_w - 16,
            9,
            "A logit + linear spec is plausible for your covariates.",
            align="C",
        )
        self.set_xy(rx + 8, y0 + 82)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*GRAPHITE)
        self.multi_cell(col_w - 16, 8.5, "Fewer moving parts.\nStill doubly robust.", align="C")

        self.set_xy(28, y0 + col_h + 20)
        self.set_font("Helvetica", "", 18)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 56,
            10,
            "Same staggered surface either way: ATT(g,t) cells, event studies, "
            "group effects, HonestDiD.",
            align="C",
        )

        band_y = y0 + col_h + 52
        self.set_fill_color(*GRAPHITE)
        self.rect(28, band_y, WIDTH - 56, 40, "F")
        self.set_xy(34, band_y + 6)
        self.set_font("Helvetica", "B", 17)
        self.set_text_color(*SIGNAL_BRIGHT)
        self.cell(
            WIDTH - 68,
            9,
            "Four learners built in: linear, ridge, sieve, logit - no extra installs.",
            align="C",
        )
        self.set_xy(34, band_y + 19)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*WHITE)
        self.multi_cell(
            WIDTH - 68,
            8.5,
            "Or bring your own: scikit-learn regressors already fit the fit() /\n"
            "predict() contract - classifiers add predict_proba().",
            align="C",
        )
        self.add_footer()

    def slide_09_code(self):
        self.add_page()
        self.dark_gradient_background()
        self._draw_vertical_sidebar(9, dark=True)

        self._kicker(34, "The Code", color=SIGNAL_BRIGHT)

        self.centered_text(58, "Three lines. Any learner.", size=40, color=WHITE)

        W_ = WHITE
        G_ = GREEN_CODE
        S_ = SLATE_CODE
        code = [
            [("est = DMLDiD(", W_)],
            [("    outcome_learner=", W_), (f'"{CODE_LEARNER}"', G_), (",", W_)],
            [
                ("    n_folds=", W_),
                (str(CODE_FOLDS), G_),
                (", seed=", W_),
                (str(CODE_SEED), G_),
                (",", W_),
            ],
            [("    base_period=", W_), ('"universal"', G_), (",", W_)],
            [(")", W_)],
            [
                ("res = est.fit(df, outcome=", W_),
                ('"y"', G_),
                (", unit=", W_),
                ('"unit"', G_),
                (", time=", W_),
                ('"time"', G_),
                (",", W_),
            ],
            [
                ("              first_treat=", W_),
                ('"first_treat"', G_),
                (", covariates=[", W_),
                ('"x1"', G_),
                (", ", W_),
                ('"x2"', G_),
                ("])", W_),
            ],
            [("res.aggregate(", W_), ('"event_study"', G_), (")", W_)],
            [("", W_)],
            [("# or bring your own - sklearn fits the contract:", S_)],
            [("from sklearn.ensemble import GradientBoostingRegressor", W_)],
            [("DMLDiD(outcome_learner=GradientBoostingRegressor())", W_)],
        ]
        block_w = 226
        self._add_code_block(
            (WIDTH - block_w) / 2, 88, block_w, code, font_size=15, line_height=11.8
        )

        self.set_xy(30, 252)
        self.set_font("Helvetica", "", 16)
        self.set_text_color(*LIGHT_GRAY)
        self.multi_cell(
            WIDTH - 60,
            8.5,
            "seed= pins the fold draws - the shown call reproduces the shown numbers exactly.",
            align="C",
        )
        self.add_footer(dark=True)

    def slide_10_production(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(10)

        self._kicker(34, "Built For Production")

        self.centered_text(56, "The full staggered toolkit.", size=36)

        features = [
            (
                "Post-fit aggregation",
                "event studies, group effects,\nHonestDiD, uniform sup-t bands",
            ),
            (
                "Survey data",
                f"designs + clustering: PSU-cohesive\nfolds; design df: {SURVEY_PSU} PSUs\n"
                f"- {SURVEY_STRATA} strata = {SURVEY_DF}",
            ),
            (
                "Repeated cross-sections",
                "panel=False runs Chang's Case 2:\nfresh samples from the SAME\ntarget population each wave",
            ),
            (
                "Cross-fit diagnostics",
                "per-cell fold losses, overlap,\ntrimming - inspect every nuisance",
            ),
        ]
        card_w, card_h = 108, 62
        gap_x, gap_y = 14, 12
        x0 = (WIDTH - 2 * card_w - gap_x) / 2
        y0 = 82
        for i, (head, body) in enumerate(features):
            cx = x0 + (i % 2) * (card_w + gap_x)
            cy = y0 + (i // 2) * (card_h + gap_y)
            self._shadow_rect(cx, cy, card_w, card_h)
            self.set_fill_color(*WHITE)
            self.set_draw_color(220, 220, 220)
            self.rect(cx, cy, card_w, card_h, "DF")
            self.set_fill_color(*SIGNAL)
            self.rect(cx, cy, card_w, 3, "F")
            self.set_xy(cx + 6, cy + 9)
            self.set_font("Helvetica", "B", 17)
            self.set_text_color(*GRAPHITE)
            self.cell(card_w - 12, 9, head, align="C")
            self.set_xy(cx + 6, cy + 22)
            self.set_font("Helvetica", "", 13.5)
            self.set_text_color(*GRAY)
            self.multi_cell(card_w - 12, 7.5, body, align="C")

        strip_y = y0 + 2 * card_h + gap_y + 16
        self.set_fill_color(*GRAPHITE)
        self.rect(28, strip_y, WIDTH - 56, 30, "F")
        self.set_xy(34, strip_y + 5)
        self.set_font("Helvetica", "B", 15.5)
        self.set_text_color(*SIGNAL_BRIGHT)
        self.cell(WIDTH - 68, 8, f"Chang Case 1 score vs DoubleML: ATT diff {DOUBLEML_ATT_DIFF}")
        self.set_xy(34, strip_y + 15)
        self.set_font("Helvetica", "", 13)
        self.set_text_color(*LIGHT_GRAY)
        self.cell(
            WIDTH - 68,
            7,
            "committed parity spike; staggered cells + survey lanes are documented "
            "extensions of the paper",
        )
        self.add_footer()

    def slide_11_cta(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(11)

        self.draw_split_logo(38, size=34)

        self.centered_text(80, "Your covariates aren't linear.", size=38)
        self.centered_text(106, "Your DiD can keep up now.", size=38, color=SIGNAL)

        chip_w = 150
        chip_x = (WIDTH - chip_w) / 2
        self.set_fill_color(*PANEL_NAVY)
        self.rect(chip_x, 142, chip_w, 22, "F")
        self.set_xy(chip_x, 148)
        self.set_font("Courier", "B", 19)
        self.set_text_color(*GREEN_CODE)
        self.cell(chip_w, 10, "pip install diff-diff", align="C")

        self.set_xy(30, 184)
        self.set_font("Helvetica", "", 17)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 60,
            9,
            "Tutorial 32 reproduces every number on this deck - learner "
            "comparison, HonestDiD, and the survey lane included:",
            align="C",
        )
        self.centered_text(
            218,
            "diff-diff.readthedocs.io/en/latest/tutorials/32_dml_did.html",
            size=14,
            color=SLATE,
        )

        self.centered_text(242, "github.com/igerber/diff-diff", size=17)

        self.set_xy(0, 264)
        self.set_font("Helvetica", "I", 13)
        self.set_text_color(*GRAY)
        self.cell(
            WIDTH,
            6,
            "Chang (2020), The Econometrics Journal 23(2) - doi:10.1093/ectj/utaa001.",
            align="C",
        )
        self.add_footer()


def main():
    pdf = DMLCarouselPDF()
    try:
        pdf.slide_01_cover()
        pdf.slide_02_bet()
        pdf.slide_03_receipt()
        pdf.slide_04_twist()
        pdf.slide_05_paper()
        pdf.slide_06_math()
        pdf.slide_07_payoff()
        pdf.slide_08_when()
        pdf.slide_09_code()
        pdf.slide_10_production()
        pdf.slide_11_cta()

        output_path = Path(__file__).parent / "diff-diff-dml-carousel.pdf"
        pdf.output(str(output_path))
        print(f"PDF saved to: {output_path}")
    finally:
        pdf.cleanup()


if __name__ == "__main__":
    main()
