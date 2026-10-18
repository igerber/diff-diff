#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for the bad-controls launch (v3.12).

Mirrors the POSTER-MODE architecture of ``generate_dml_carousel.py``
(magazine sidebar with progress tick, light gradient background, split-color
logo, footer wordmark, dark slides, soft card shadows; poster type floors:
body >= 16pt, captions >= 13pt, headlines 32-54pt, code >= 15pt; one idea +
one visual + roughly 25 words per slide).

Palette: "Junction" - ink/stone structure with a single AMBER accent reserved
for the bad control itself and for the third road out of the fork (the
deck's visual thesis: include-or-drop is a false choice; condition on the
covariate's UNTREATED path instead). The two wrong roads (include at t, drop)
are always warm gray, matching the DML deck's misspecified-fit convention.
Two dark slides: the "drop it" twist (slide 4) and the code slide (slide 9).

Narrative spine (practitioner-dilemma hook - user decision 2026-10-18: lead
with the include-or-drop choice every practitioner faces, name the covariate
explicitly on the cover, then show both roads miss):

1.  Cover          -- "Include the covariate, or drop it? Both can be wrong."
                      ("can", not "are": the paper's bias terms vanish when
                      Condition 1 or 2 fails - user feedback 2026-10-18).
                      Fork motif (illustrative): one road splits into two
                      gray roads that bend away from the true effect and
                      one amber road that reaches it. Subtitle names the
                      3.12 release that shipped the lane (FEATURE_RELEASE,
                      pinned to the CHANGELOG by the claims test; the
                      FOOTER version is derived from pyproject, never
                      hardcoded) and the paper.
2.  The dilemma    -- a covariate treatment can change (occupation, hours,
                      insurance, firm size). The paper's Definition 1: a
                      bad control matters for the untreated outcome trend
                      (Condition 1) AND treatment affects it (Condition 2).
3.  The receipt    -- tutorial 33's naive within-TWFE with the bad control
                      at t: 0.4886 (SE 0.0142) against a true 1.00 on the
                      paper's own DGP 1 made staggered (truth known by
                      construction, said on-slide); the half of the effect
                      that runs through the covariate is netted out; the
                      paper's Table S2 reports the same -0.501 bias.
4.  Twist (DARK)   -- dropping it is still wrong: the covariate's untreated
                      path carries trend information (Section 3.2). NO
                      number on this slide on purpose: tutorial 33 states
                      the result and deliberately does not run a drop-X
                      arm, and the deck never shows a number the committed
                      notebook does not (the earlier on-slide "no number"
                      caption was cut on user feedback 2026-10-18).
5.  The paper      -- Caetano, Callaway, Payne & Sant'Anna (2026), arXiv
                      preprint (status on-slide; the paper is first NAMED on
                      slide 2 so slides 3-4 may say "the paper's"; the
                      Hugo-not-Pedro disambiguation stays off-slide): two
                      ways out -
                      condition on the pre-treatment value (Approach 1) or
                      model the covariate's untreated path (Approach 2).
                      ATTRIBUTED PARAPHRASE ONLY, no double-quoted pull
                      quote (the review is pinned to arXiv v2 of an
                      unpublished preprint).
6.  Route 1        -- already in the box: CallawaySantAnna reads covariates
                      at the base period on panel data; 1.0540 (SE 0.0235)
                      with [x, z], 0.9754 (SE 0.0274) with [x, z, w]. Which
                      conditioning set is right is an identification call,
                      stated on-slide.
7.  Route 2        -- the DMLDiD bad-control lane: the paper's
                      Neyman-orthogonal doubly-robust score, four
                      cross-fitted nuisances, two nested; 0.9766 (SE
                      0.0260). Payoff chart: three committed estimates
                      against the truth line - TWFE gray, BOTH routes amber
                      (both are valid; user feedback 2026-10-18). Body says
                      what Route 2 buys: W enters only the covariate's path
                      model, never parallel trends (Remark 2), plus the DR /
                      ML score and the pre-test.
8.  The pre-test   -- bad_control_summary(): the six ATT_X(g,t) rows. Pre
                      rows (t < g) pre-test the path assumptions and sit
                      near zero; post rows check that treatment moved the
                      covariate and sit near the true 0.50. Both caveats
                      on-slide: a nonzero pre row is NOT evidence of a bad
                      control; a zero post row does not prove the converse.
9.  Code (DARK)    -- the tutorial's own lane fit (seed=0, W = ["w"]) plus
                      the Route 1 one-liner, >= 15pt.
10. In the wild    -- the paper's NLSY79 job-displacement application (THE
                      AUTHORS' estimates, labeled as such, not reproduced
                      here), told wrong-then-right: the intro states that
                      displacement moved the covariate (ATT_X -0.0261, SE
                      0.0092: occupations paying about 2.6% less); the gray
                      card is naive TWFE ("almost -10%", the paper's words,
                      with or without the occupation score); the amber card
                      is the paper's bad-control estimator (ATT -0.0672, SE
                      0.0242, about -7%); the line below says naive TWFE is
                      roughly 40% larger in magnitude and WHY the right one
                      is right (parallel trends conditional on the
                      occupation a worker would have had without
                      displacement). Plus the
                      restrictions + validation strip (panel only, cluster
                      yes / survey no, CS simple weighting for the headline
                      ATT, numpy oracle at 1e-12, black-box goldens vs the
                      authors' R package).
11. CTA            -- pip install; tutorial 33 mentioned ONCE, here.

Claim discipline (verified against the committed executed notebook
``docs/tutorials/33_bad_controls.ipynb``, the paper review
``docs/methodology/papers/caetano-2026-review.md`` and
``docs/methodology/REGISTRY.md``):

- EVERY simulated estimate on the deck is a committed, seed-locked
  tutorial-33 value from the paper's DGP 1 made staggered, truth known by
  construction - labeled on-slide wherever a truth value appears. notebook
  <-> deck sync is pinned by ``tests/test_bad_controls_carousel_claims.py``
  (parses this module's constants via ``ast`` and locates each on the
  committed notebook surface).
- The application numbers on slide 10 are the PAPER'S estimates (Section 7,
  Figures 6-8 as transcribed in the paper review), labeled "the authors'
  estimates" on-slide; the library has not reproduced them and the deck
  never says it has. The "roughly 40% larger" line is the paper's own text
  statement as recorded in the review.
- The drop-X slide carries no number: the tutorial does not run that arm
  and the deck never shows a number the committed notebook does not.
- The pre-test slide carries both Remark 6 caveats ON the slide: pre rows
  flag a possible assumption violation, not a bad control; a zero post row
  does not establish that treatment left the covariate alone.
- The Route 1 slide says which conditioning set is right is an
  identification call (the tutorial's own caveat: on this DGP conditioning
  on W alone already recovers the truth, so the deck does not claim "you
  need the lane").
- The headline ATT keeps the Callaway-Sant'Anna simple weighting (not the
  paper's Remark 4 overall) - said on the production strip. Panel-only /
  cluster-only / anticipation=0 / varying base period - said there too.
- The cover motif is stylized art, labeled "illustrative" on-slide; it
  plots no tutorial data.
- No "only"/"first"/"unbiased"/"detects bad controls" absolutes anywhere
  (claims test bans them). No external competitor is named; the authors'
  R package is named once, as the black-box parity oracle it is.
- CallawaySantAnna is named honestly as the Route 1 host (Approach 1 needs
  no new code) - the lane is an addition, not a replacement.
- Tutorial 33 is referenced ONCE, on the CTA slide (house precedent).

Run with::

    python carousel/generate_bad_controls_carousel.py

Produces ``carousel/diff-diff-bad-controls-carousel.pdf``. Generation
requires ``fpdf2``, ``Pillow``, and ``matplotlib`` (carousel-only
dependencies, not part of the library's install extras).
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

# The release that SHIPPED the bad-control lane (CHANGELOG 3.12.0 section).
# Deliberately not derived from pyproject: the deck may be built from a later
# version, and the cover's "New in" claim is about the shipping release.
# Pinned to the CHANGELOG by tests/test_bad_controls_carousel_claims.py.
FEATURE_RELEASE = "3.12"

TOTAL_SLIDES = 11

# -------------------------------------------------------------------------
# "Junction" palette -- ink/stone structure, AMBER = the bad control / the
# third road. The two wrong roads are always warm gray.
# -------------------------------------------------------------------------
INK = (28, 25, 23)  # #1c1917  primary structure / code panel
STONE = (68, 64, 60)  # #44403c  secondary structure
AMBER = (217, 119, 6)  # #d97706  THE BAD CONTROL / THE THIRD ROAD
AMBER_BRIGHT = (251, 191, 36)  # #fbbf24  amber on the dark slides
AMBER_TINT = (255, 251, 235)  # #fffbeb  gradient start
CREAM = (254, 243, 199)  # #fef3c7  light fills / dark-slide text pop
WRONG_GRAY = (120, 113, 108)  # #78716c  the two wrong roads (warm gray)
SHADOW = (214, 211, 209)  # #d6d3d1  soft card shadow

# Text + structural (shared with the other decks for legibility)
NAVY = (15, 23, 42)  # #0f172a  primary text; dark-slide gradient start
GRAY = (100, 116, 139)  # #64748b
LIGHT_GRAY = (148, 163, 184)  # #94a3b8
WHITE = (255, 255, 255)
DEEP_AMBER = (46, 30, 10)  # #2e1e0a  dark-slide gradient end
PANEL_INK = (28, 25, 23)  # #1c1917  code panel fill
AMBER_CODE = (251, 191, 36)  # #fbbf24  code string/number literals
SLATE_CODE = (168, 162, 158)  # #a8a29e  code comments

INK_HEX = "#1c1917"
STONE_HEX = "#44403c"
AMBER_HEX = "#d97706"
AMBER_BRIGHT_HEX = "#fbbf24"
WRONG_GRAY_HEX = "#78716c"
NAVY_HEX = "#0f172a"
GRAY_HEX = "#64748b"
LIGHT_GRAY_HEX = "#94a3b8"

# -------------------------------------------------------------------------
# Seed-locked numbers from the committed executed tutorial
# (docs/tutorials/33_bad_controls.ipynb, seed 5). Every constant below is
# located on the committed notebook surface by
# tests/test_bad_controls_carousel_claims.py - never edit one without
# re-running it. The example is SIMULATED (the paper's DGP 1 made
# staggered) with truth known by construction: ATT = 1.00 and ATT_X = 0.50
# in every post-treatment cell, 0 before.
# -------------------------------------------------------------------------

TRUTH = 1.00  # lambda + delta
TRUTH_X = 0.50  # lambda - the effect of treatment on the bad control
TWFE_INCLUDE = (0.4886, 0.0142)  # within-TWFE with the bad control at t: ATT, SE
CS_A1 = (1.0540, 0.0235)  # CallawaySantAnna, covariates=["x", "z"]
CS_A1_W = (0.9754, 0.0274)  # CallawaySantAnna, covariates=["x", "z", "w"]
LANE_W = (0.9766, 0.0260)  # DMLDiD bad-control lane, W = ["w"]
PAPER_S2_BIAS = -0.501  # the paper's Table S2, TWFE Include BC, n = 2000

ATT_X_ROWS = (  # (g, t, att_x, se_x) - results.bad_control_summary()
    (3, 2, 0.0019, 0.0166),
    (3, 3, 0.5137, 0.0156),
    (3, 4, 0.5055, 0.0299),
    (4, 2, 0.0120, 0.0169),
    (4, 3, 0.0061, 0.0190),
    (4, 4, 0.4764, 0.0263),
)

N_UNITS, N_PERIODS = 2000, 4
N_NEVER = 974  # never-treated units in the simulated panel
COHORT_SIZES = ((3, 531), (4, 495))

# Tutorial fit config rendered on the code slide (claims test pins these
# against the notebook's code cells)
CODE_SEED = 0
CODE_BAD_CONTROL = "x"
CODE_W = "w"
CODE_Z = "z"
CODE_CONTROL_GROUP = "not_yet_treated"

# The paper's NLSY79 application (Section 7; the AUTHORS' estimates as
# transcribed in docs/methodology/papers/caetano-2026-review.md - the
# library has not reproduced them and the deck says so on-slide).
NLSY_ATT_X = (-0.0261, 0.0092)  # overall ATT_X: occupation score
NLSY_ATT = (-0.0672, 0.0242)  # overall ATT: log earnings (baseline imputation)
NLSY_N_DISPLACED, NLSY_N_NOT = 748, 2483
NLSY_TWFE_TEXT = "almost -10%"  # the paper's own words for the naive TWFE arms

# Validation anchors (REGISTRY DMLDiD bad-control block / CHANGELOG)
ORACLE_TOL = "1e-12"
R_PACKAGE = "badcontrols"


class BadControlsCarouselPDF(FPDF):
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
    # Magazine vertical sidebar -- INK bar, AMBER progress tick.
    # -----------------------------------------------------------------
    def _draw_vertical_sidebar(self, slide_number, total=TOTAL_SLIDES, dark=False):
        bar_x = 14
        bar_y_top = 45
        bar_y_bottom = 275
        self.set_draw_color(*(LIGHT_GRAY if dark else INK))
        self.set_line_width(0.6)
        self.line(bar_x, bar_y_top, bar_x, bar_y_bottom)

        ratio = (slide_number - 1) / (total - 1) if total > 1 else 0.0
        tick_y = bar_y_top + ratio * (bar_y_bottom - bar_y_top)
        self.set_draw_color(*(AMBER_BRIGHT if dark else AMBER))
        self.set_line_width(1.2)
        self.line(bar_x - 4, tick_y, bar_x + 7, tick_y)

    # -----------------------------------------------------------------
    # Backgrounds + footer
    # -----------------------------------------------------------------
    def light_gradient_background(self):
        """Amber tint #fffbeb fading to white."""
        steps = 50
        r0, g0, b0 = AMBER_TINT
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
        """Near-black navy #0f172a fading to deep amber #2e1e0a."""
        steps = 50
        r0, g0, b0 = NAVY
        r1, g1, b1 = DEEP_AMBER
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
        self.set_text_color(*(AMBER_BRIGHT if dark else AMBER))
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

    def _kicker(self, y, text, color=AMBER):
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
        """Split-color diff-diff logo with AMBER middle dash."""
        self.set_xy(0, y)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="R")
        self.set_text_color(*AMBER)
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
    # Cover motif -- the fork. One road arrives from the left and splits:
    # two warm-gray roads ("include X at t", "drop X") bend away from the
    # true effect, one amber road ("condition on its untreated path")
    # reaches it. Stylized art, not a data claim (it plots no tutorial
    # data) - labeled "illustrative" on-slide.
    # -----------------------------------------------------------------
    def _render_cover_motif(self):
        fig, ax = plt.subplots(figsize=(10, 4.0))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        # Shared trunk
        xt = np.linspace(-3.0, 0.0, 40)
        ax.plot(xt, np.zeros_like(xt), color=INK_HEX, linewidth=5.0, solid_capstyle="round")
        ax.plot([0.0], [0.0], "o", color=INK_HEX, markersize=11, zorder=6)

        # Branches
        xb = np.linspace(0.0, 4.0, 160)
        bend = 1.35 * np.tanh(xb / 1.5) ** 1.6
        ax.plot(xb, bend, color=WRONG_GRAY_HEX, linewidth=4.0, linestyle=(0, (6, 3)))
        ax.plot(xb, -bend, color=WRONG_GRAY_HEX, linewidth=4.0, linestyle=(0, (6, 3)))
        ax.plot(xb, np.zeros_like(xb), color=AMBER_HEX, linewidth=5.0, zorder=5)

        # The true effect: a target ring at the end of the amber road
        ax.plot([4.0], [0.0], "o", color="white", markersize=22, zorder=7)
        ax.plot(
            [4.0],
            [0.0],
            "o",
            color=AMBER_HEX,
            markersize=22,
            zorder=7,
            fillstyle="none",
            markeredgewidth=3.0,
        )
        ax.plot([4.0], [0.0], "o", color=AMBER_HEX, markersize=9, zorder=8)

        # Labels
        ax.text(
            4.22,
            float(bend[-1]),
            "include it at t",
            fontsize=15,
            color=WRONG_GRAY_HEX,
            fontweight="bold",
            va="center",
        )
        ax.text(
            4.22,
            float(-bend[-1]),
            "drop it",
            fontsize=15,
            color=WRONG_GRAY_HEX,
            fontweight="bold",
            va="center",
        )
        ax.text(
            4.42,
            0.0,
            "the true effect",
            fontsize=15,
            color=AMBER_HEX,
            fontweight="bold",
            va="center",
        )
        ax.text(
            2.0,
            -0.26,
            "condition on its untreated path",
            fontsize=13.5,
            color=AMBER_HEX,
            ha="center",
            va="top",
            style="italic",
        )
        ax.text(
            -1.5,
            0.22,
            "your covariate",
            fontsize=13.5,
            color=INK_HEX,
            ha="center",
            va="bottom",
            style="italic",
        )
        ax.text(
            -3.1,
            -1.55,
            "illustrative",
            fontsize=15,
            color=LIGHT_GRAY_HEX,
            style="italic",
            ha="left",
        )

        ax.set_xlim(-3.2, 6.6)
        ax.set_ylim(-1.7, 1.7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=0.2)
        return self._save_fig(fig, transparent=True)

    # -----------------------------------------------------------------
    # Payoff chart -- the four committed estimates vs the truth line.
    # ONLY committed tutorial values (point estimates + 1.96*SE whiskers).
    # -----------------------------------------------------------------
    def _render_payoff_chart(self):
        fig, ax = plt.subplots(figsize=(10.2, 4.3))

        arms = [
            ("TWFE\ninclude X at t", TWFE_INCLUDE, WRONG_GRAY_HEX),
            ("Route 1: CallawaySantAnna\ncovariates=[x, z, w]", CS_A1_W, AMBER_HEX),
            ("Route 2: DMLDiD\nbad_control=x, W=[w]", LANE_W, AMBER_HEX),
        ]
        xs = np.arange(len(arms))
        ax.axhline(TRUTH, color=INK_HEX, linewidth=2.2, linestyle=(0, (5, 3)))
        ax.text(
            2.62,
            TRUTH + 0.02,
            f"true effect {TRUTH:.2f}\n(simulated - known\nby construction)",
            fontsize=13.5,
            color=INK_HEX,
            fontweight="bold",
            va="bottom",
        )

        for i, (_name, (att, se), color) in enumerate(arms):
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
                xytext=(i + 0.1, att + 0.03),
                fontsize=15,
                fontweight="bold",
                color=color,
            )

        ax.set_xticks(xs)
        ax.set_xticklabels([n for n, _, _ in arms], fontsize=12.5, color=NAVY_HEX)
        ax.set_ylabel("estimated overall ATT", fontsize=14, color=NAVY_HEX)
        ax.tick_params(axis="y", labelsize=13, colors=GRAY_HEX)
        ax.set_xlim(-0.6, 3.9)
        ax.set_ylim(0.38, 1.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("bottom", "left"):
            ax.spines[s].set_color(LIGHT_GRAY_HEX)
        fig.tight_layout(pad=0.3)
        return self._save_fig(fig, facecolor="white")

    # -----------------------------------------------------------------
    # Pre-test table (fpdf cells, big type) -- the six ATT_X(g,t) rows.
    # -----------------------------------------------------------------
    def _draw_att_x_table(self, x, y, w):
        cols = (("cohort g", 0.19), ("period t", 0.19), ("ATT_X(g,t)", 0.31), ("SE", 0.31))
        row_h = 13.5
        # Header
        self.set_fill_color(*INK)
        self.rect(x, y, w, row_h, "F")
        cx = x
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*WHITE)
        for label, frac in cols:
            self.set_xy(cx, y + 2.5)
            self.cell(w * frac, 8.5, label, align="C")
            cx += w * frac
        ry = y + row_h
        for g, t, att_x, se_x in ATT_X_ROWS:
            post = t >= g
            self.set_fill_color(*(CREAM if post else WHITE))
            self.set_draw_color(225, 222, 218)
            self.set_line_width(0.3)
            self.rect(x, ry, w, row_h, "DF")
            self.set_fill_color(*(AMBER if post else LIGHT_GRAY))
            self.rect(x, ry, 3.0, row_h, "F")
            cx = x
            vals = (str(g), str(t), f"{att_x:.4f}", f"{se_x:.4f}")
            for (label, frac), val in zip(cols, vals):
                self.set_xy(cx, ry + 2.5)
                bold = label.startswith("ATT_X")
                self.set_font("Helvetica", "B" if bold else "", 16)
                self.set_text_color(*((AMBER if post else GRAY) if bold else NAVY))
                self.cell(w * frac, 8.5, val, align="C")
                cx += w * frac
            ry += row_h
        return ry - y

    # -----------------------------------------------------------------
    # Code-block helper (dark panel with token-colored lines)
    # -----------------------------------------------------------------
    def _add_code_block(self, x, y, w, token_lines, font_size=15, line_height=12.5, fill=PANEL_INK):
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

        self.draw_split_logo(22, size=34)

        self.centered_text(50, "Include the covariate,", size=46)
        self.centered_text(74, "or drop it?", size=46)
        self.centered_text(98, "Both can be wrong.", size=46, color=AMBER)

        motif_path, _pw, _ph = self._render_cover_motif()
        motif_w = 244
        self.image(motif_path, (WIDTH - motif_w) / 2, 122, motif_w)

        self.centered_text(
            230,
            f"New in diff-diff {FEATURE_RELEASE}: DiD with bad controls.",
            size=22,
            color=INK,
        )
        self.centered_text(
            248,
            "Caetano, Callaway, Payne & Sant'Anna (2026).",
            size=19,
            bold=False,
        )
        self.add_footer()

    def slide_02_dilemma(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(2)

        self._kicker(40, "The Dilemma")

        self.centered_text(72, "A covariate that", size=44)
        self.centered_text(98, "treatment can change.", size=44, color=AMBER)

        self.set_xy(28, 130)
        self.set_font("Helvetica", "", 20)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 56,
            11,
            "Occupation. Hours worked. Health insurance. Firm size. Anything "
            "measured after treatment starts that treatment plausibly moves.",
            align="C",
        )

        self.set_draw_color(*AMBER)
        self.set_line_width(1.4)
        self.line(WIDTH / 2 - 30, 172, WIDTH / 2 + 30, 172)

        self.set_xy(28, 186)
        self.set_font("Helvetica", "B", 21)
        self.set_text_color(*INK)
        self.multi_cell(
            WIDTH - 56,
            11.5,
            "Caetano, Callaway, Payne & Sant'Anna (2026) call it a bad control "
            "when it matters for the untreated outcome trend (Condition 1) and "
            "treatment affects it (Condition 2).",
            align="C",
        )

        self.set_xy(28, 232)
        self.set_font("Helvetica", "", 17)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 56,
            9,
            "Condition on it at t and you compare units at different values of "
            "its untreated version. Drop it and you throw away what its path "
            "says about the trend.",
            align="C",
        )
        self.add_footer()

    def slide_03_receipt(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(3)

        self._kicker(36, "The Receipt")

        self.centered_text(62, "Include it at t.", size=48)

        card_w, card_h = 108, 78
        gap = 14
        left_x = (WIDTH - 2 * card_w - gap) / 2
        y0 = 100
        self._stat_card(
            left_x,
            y0,
            card_w,
            card_h,
            f"{TWFE_INCLUDE[0]:.4f}",
            [
                (f"reported SE {TWFE_INCLUDE[1]:.4f}", True),
                ("TWFE with the bad control at t -", False),
                ("unit-clustered", False),
            ],
            WRONG_GRAY,
        )
        self._stat_card(
            left_x + card_w + gap,
            y0,
            card_w,
            card_h,
            f"{TRUTH:.2f}",
            [
                ("the true effect", True),
                ("simulated staggered panel -", False),
                ("truth known by construction", False),
            ],
            AMBER,
        )

        self.centered_text(198, "Half the effect, netted out.", size=30, color=INK)

        self.set_xy(30, 222)
        self.set_font("Helvetica", "", 17.5)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 60,
            9.5,
            f"A simulated staggered panel built from the paper's first Monte Carlo design: "
            f"{N_UNITS:,} units, {N_PERIODS} periods, cohorts treated at t = "
            f"{COHORT_SIZES[0][0]} and {COHORT_SIZES[1][0]}, {N_NEVER} never treated. "
            f"Treatment moves the covariate by {TRUTH_X:.2f}, and that half of the effect "
            f"is exactly what the regression removes - the paper's own simulation reports "
            f"the same bias ({PAPER_S2_BIAS:.3f}).",
            align="C",
        )
        self.add_footer()

    def slide_04_twist(self):
        self.add_page()
        self.dark_gradient_background()
        self._draw_vertical_sidebar(4, dark=True)

        self._kicker(40, "It Gets Worse", color=AMBER_BRIGHT)

        self.centered_text(84, "Drop it? Still wrong.", size=48, color=WHITE)

        self.set_xy(30, 128)
        self.set_font("Helvetica", "", 21)
        self.set_text_color(*LIGHT_GRAY)
        self.multi_cell(
            WIDTH - 60,
            11.5,
            "The covariate's untreated path carries information about the "
            "outcome trend. Discard the covariate and that information leaves "
            "with it.",
            align="C",
        )

        self.set_xy(30, 188)
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(*CREAM)
        self.multi_cell(
            WIDTH - 60,
            13,
            "In the paper's simulations, dropping the covariate\nis inconsistent in every design.",
            align="C",
        )

        self.add_footer(dark=True)

    def slide_05_paper(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(5)

        self._kicker(36, "The Paper")

        self.centered_text(62, "Caetano, Callaway, Payne", size=36)
        self.centered_text(84, "& Sant'Anna (2026):", size=36)
        self.centered_text(106, "two ways out.", size=36, color=AMBER)

        y0 = 138
        for num, head, body in [
            (
                "1",
                "Condition on the pre-treatment value",
                "under simple covariate unconfoundedness, the target reduces to "
                "Callaway-Sant'Anna with the covariate read at the base period - "
                "existing tools apply",
            ),
            (
                "2",
                "Model the covariate's untreated path",
                "covariate unconfoundedness given its pre-treatment value, observed "
                "confounders W and Z. W explains the covariate's path only - never "
                "parallel trends - and a doubly-robust, ML-ready score does the rest",
            ),
        ]:
            self.set_xy(34, y0)
            self.set_font("Helvetica", "B", 26)
            self.set_text_color(*AMBER)
            self.cell(14, 12, num)
            self.set_xy(52, y0)
            self.set_font("Helvetica", "B", 22)
            self.set_text_color(*NAVY)
            self.cell(WIDTH - 86, 12, head)
            self.set_xy(52, y0 + 14)
            self.set_font("Helvetica", "", 16.5)
            self.set_text_color(*GRAY)
            self.multi_cell(WIDTH - 92, 8.8, body, align="L")
            y0 += 50

        self.set_xy(30, 246)
        self.set_font("Helvetica", "I", 14)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 60,
            7.5,
            "Difference-in-differences with 'bad controls' - arXiv:2608.03881, an "
            "unpublished preprint (v2, September 2026).",
            align="C",
        )
        self.add_footer()

    def slide_06_route1(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(6)

        self._kicker(34, "Route 1")

        self.centered_text(58, "Already in the box.", size=40)
        self.centered_text(82, "Read it at the base period.", size=30, color=AMBER)

        self.set_xy(28, 104)
        self.set_font("Helvetica", "", 17.5)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 56,
            9.5,
            "On panel data, CallawaySantAnna reads every covariate at the cell's "
            "base period, never at t. Put the bad control in covariates and you "
            "have the paper's Approach 1 - no new code.",
            align="C",
        )

        card_w, card_h = 108, 78
        gap = 14
        left_x = (WIDTH - 2 * card_w - gap) / 2
        y0 = 140
        self._stat_card(
            left_x,
            y0,
            card_w,
            card_h,
            f"{CS_A1[0]:.4f}",
            [
                (f"SE {CS_A1[1]:.4f}", True),
                ('covariates=["x", "z"] - W left out', False),
                ("misses the true 1.00 by two SE", False),
            ],
            STONE,
        )
        self._stat_card(
            left_x + card_w + gap,
            y0,
            card_w,
            card_h,
            f"{CS_A1_W[0]:.4f}",
            [
                (f"SE {CS_A1_W[1]:.4f}", True),
                ('covariates=["x", "z", "w"] - W in', False),
                ("within one SE of the true 1.00", False),
            ],
            AMBER,
        )

        self.set_xy(28, 232)
        self.set_font("Helvetica", "", 15.5)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 56,
            8.5,
            "This design has an observed confounder W of the covariate's path. Leave "
            "it out and Route 1 misses; put it in and Route 1 works - but which "
            "conditioning set is right is an identification call, not a fit statistic. "
            "Simulated example, truth known by construction.",
            align="C",
        )
        self.add_footer()

    def slide_07_route2(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(7)

        self._kicker(34, "Route 2")

        self.centered_text(58, "The bad-control lane.", size=40)
        self.centered_text(82, "What Route 2 buys you.", size=30, color=AMBER)

        chart_path, _pw, _ph = self._render_payoff_chart()
        chart_w = 226
        self.image(chart_path, (WIDTH - chart_w) / 2, 100, chart_w)

        self.set_xy(28, 206)
        self.set_font("Helvetica", "", 16.5)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 56,
            9,
            "Both routes are valid. Route 1 puts W into parallel trends. Route 2 "
            "uses W only to explain the covariate's untreated path - so the lagged "
            "outcome can be W without becoming a parallel-trends variable (the "
            "paper's Remark 2) - plus a doubly-robust, ML-ready score and the "
            f"pre-test on the next slide. Here: {LANE_W[0]:.4f} (SE {LANE_W[1]:.4f}).",
            align="C",
        )
        self.set_xy(28, 252)
        self.set_font("Helvetica", "I", 13)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 56,
            7,
            "Seed-locked simulated example; whiskers are 1.96 analytical SE. On this "
            "draw both routes land within one SE of the truth - Route 2 is a different "
            "set of assumptions, not a bigger number.",
            align="C",
        )
        self.add_footer()

    def slide_08_pretest(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(8)

        self._kicker(34, "The Pre-Test")

        self.centered_text(56, "Did treatment move the covariate?", size=34)

        self.set_xy(28, 80)
        self.set_font("Helvetica", "", 16.5)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 56,
            9,
            "Every cell also reports ATT_X(g,t): the effect of treatment on the "
            "bad control itself, with its own analytical SE - "
            "results.bad_control_summary().",
            align="C",
        )

        table_w = 190
        self._draw_att_x_table((WIDTH - table_w) / 2, 108, table_w)

        y0 = 208
        for head, body, color in [
            (
                "Pre rows (t < g)",
                "pre-test the path assumptions. They should be zero - and are, within SE.",
                GRAY,
            ),
            (
                "Post rows (t >= g)",
                f"check that treatment moves the covariate - near the true {TRUTH_X:.2f}.",
                AMBER,
            ),
        ]:
            self.set_xy(30, y0)
            self.set_font("Helvetica", "B", 16.5)
            self.set_text_color(*color)
            self.cell(WIDTH - 60, 9, head)
            self.set_xy(30, y0 + 9.5)
            self.set_font("Helvetica", "", 16)
            self.set_text_color(*NAVY)
            self.multi_cell(WIDTH - 60, 8.5, body, align="L")
            y0 += 22

        self.set_xy(28, 254)
        self.set_font("Helvetica", "I", 13.5)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 56,
            7,
            "A nonzero pre row flags a possible assumption violation - it is not "
            "evidence of a bad control. A zero post row does not prove treatment "
            "left the covariate alone. Simulated; truth known by construction.",
            align="C",
        )
        self.add_footer()

    def slide_09_code(self):
        self.add_page()
        self.dark_gradient_background()
        self._draw_vertical_sidebar(9, dark=True)

        self._kicker(34, "The Code", color=AMBER_BRIGHT)

        self.centered_text(58, "Name it. Name its confounders.", size=38, color=WHITE)

        W_ = WHITE
        A_ = AMBER_CODE
        S_ = SLATE_CODE
        code = [
            [
                ("res = DMLDiD(control_group=", W_),
                (f'"{CODE_CONTROL_GROUP}"', A_),
                (", seed=", W_),
                (str(CODE_SEED), A_),
                (").fit(", W_),
            ],
            [
                ("    df, outcome=", W_),
                ('"y"', A_),
                (", unit=", W_),
                ('"unit"', A_),
                (", time=", W_),
                ('"time"', A_),
                (",", W_),
            ],
            [("    first_treat=", W_), ('"first_treat"', A_), (",", W_)],
            [
                ("    covariates=[", W_),
                (f'"{CODE_Z}"', A_),
                ("],", W_),
                ("                  # Z: clean covariates", S_),
            ],
            [
                ("    bad_control=", W_),
                (f'"{CODE_BAD_CONTROL}"', A_),
                (",", W_),
                ("                   # the one treatment moves", S_),
            ],
            [
                ("    bad_control_covariates=[", W_),
                (f'"{CODE_W}"', A_),
                ("],", W_),
                ("      # W: confounders of its path", S_),
            ],
            [(")", W_)],
            [("res.bad_control_summary()", W_), ("     # ATT_X(g,t) per cell", S_)],
            [("res.aggregate(", W_), ('"event_study"', A_), (")", W_)],
            [("", W_)],
            [("# Route 1 instead - read it at the base period:", S_)],
            [
                ("CallawaySantAnna(control_group=", W_),
                (f'"{CODE_CONTROL_GROUP}"', A_),
                (").fit(", W_),
            ],
            [
                ("    df, ..., covariates=[", W_),
                (f'"{CODE_BAD_CONTROL}"', A_),
                (", ", W_),
                (f'"{CODE_Z}"', A_),
                (", ", W_),
                (f'"{CODE_W}"', A_),
                ("])", W_),
            ],
        ]
        block_w = 232
        self._add_code_block(
            (WIDTH - block_w) / 2, 82, block_w, code, font_size=15, line_height=11.0
        )

        self.set_xy(30, 258)
        self.set_font("Helvetica", "", 15.5)
        self.set_text_color(*LIGHT_GRAY)
        self.multi_cell(
            WIDTH - 60,
            8.5,
            "Pass the bad control through bad_control=, never also through covariates= "
            "(the fit rejects a column in both). seed= pins the folds.",
            align="C",
        )
        self.add_footer(dark=True)

    def slide_10_wild(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(10)

        self._kicker(34, "In The Wild")

        self.centered_text(56, "Job displacement, NLSY79.", size=36)

        self.set_xy(28, 76)
        self.set_font("Helvetica", "", 15.5)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 56,
            8.4,
            f"The paper's application: {NLSY_N_DISPLACED:,} displaced workers, "
            f"{NLSY_N_NOT:,} not. The bad control is the occupation score - displacement "
            f"moves workers into occupations paying about {-NLSY_ATT_X[0] * 100:.1f}% less "
            f"(ATT_X {NLSY_ATT_X[0]:.4f}, SE {NLSY_ATT_X[1]:.4f}). The authors' estimates, "
            "not reproduced here:",
            align="C",
        )

        card_w, card_h = 108, 74
        gap = 14
        left_x = (WIDTH - 2 * card_w - gap) / 2
        y0 = 120
        self._stat_card(
            left_x,
            y0,
            card_w,
            card_h,
            f"{NLSY_TWFE_TEXT}",
            [
                ("naive TWFE on log earnings", True),
                ("with or without the", False),
                ("occupation score", False),
            ],
            WRONG_GRAY,
            headline_size=38,
        )
        self._stat_card(
            left_x + card_w + gap,
            y0,
            card_w,
            card_h,
            "about -7%",
            [
                (f"ATT {NLSY_ATT[0]:.4f}, SE {NLSY_ATT[1]:.4f}", True),
                ("the paper's bad-control", False),
                ("estimator on log earnings", False),
            ],
            AMBER,
            headline_size=38,
        )

        self.set_xy(28, 202)
        self.set_font("Helvetica", "", 15.5)
        self.set_text_color(*INK)
        self.multi_cell(
            WIDTH - 56,
            8.4,
            "Naive TWFE reads roughly 40% larger in magnitude. The bad-control "
            "estimator holds parallel trends conditional on the occupation a worker "
            "would have had without displacement - the covariate's untreated path.",
            align="C",
        )

        strip_y = 232
        self.set_fill_color(*INK)
        self.rect(28, strip_y, WIDTH - 56, 46, "F")
        self.set_xy(34, strip_y + 4)
        self.set_font("Helvetica", "B", 14.5)
        self.set_text_color(*AMBER_BRIGHT)
        self.multi_cell(
            WIDTH - 68,
            7.5,
            f"Validated: numpy oracle of the paper's algorithm at {ORACLE_TOL}; "
            f"tolerance goldens vs the authors' R {R_PACKAGE} package, run black-box.",
            align="C",
        )
        self.set_xy(34, strip_y + 23)
        self.set_font("Helvetica", "", 13)
        self.set_text_color(*LIGHT_GRAY)
        self.multi_cell(
            WIDTH - 68,
            7,
            "Lane scope: panel data only; cluster= yes, survey_design= no; no anticipation; "
            "varying base period. The headline ATT keeps the Callaway-Sant'Anna simple "
            "weighting, not the paper's Remark 4 overall.",
            align="C",
        )
        self.add_footer()

    def slide_11_cta(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(11)

        self.draw_split_logo(38, size=34)

        self.centered_text(80, "Your covariate moved.", size=38)
        self.centered_text(106, "Your DiD can handle it now.", size=38, color=AMBER)

        chip_w = 150
        chip_x = (WIDTH - chip_w) / 2
        self.set_fill_color(*PANEL_INK)
        self.rect(chip_x, 142, chip_w, 22, "F")
        self.set_xy(chip_x, 148)
        self.set_font("Courier", "B", 19)
        self.set_text_color(*AMBER_CODE)
        self.cell(chip_w, 10, "pip install diff-diff", align="C")

        self.set_xy(30, 184)
        self.set_font("Helvetica", "", 17)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 60,
            9,
            "Tutorial 33 reproduces every simulated number on this deck - both "
            "routes, the pre-test table, the event study and a ridge refit:",
            align="C",
        )
        self.centered_text(
            218,
            "diff-diff.readthedocs.io/en/latest/tutorials/33_bad_controls.html",
            size=14,
            color=STONE,
        )

        self.centered_text(242, "github.com/igerber/diff-diff", size=17)

        self.set_xy(0, 264)
        self.set_font("Helvetica", "I", 13)
        self.set_text_color(*GRAY)
        self.cell(
            WIDTH,
            6,
            "Caetano, Callaway, Payne & Sant'Anna (2026), arXiv:2608.03881 (preprint).",
            align="C",
        )
        self.add_footer()


def main():
    pdf = BadControlsCarouselPDF()
    try:
        pdf.slide_01_cover()
        pdf.slide_02_dilemma()
        pdf.slide_03_receipt()
        pdf.slide_04_twist()
        pdf.slide_05_paper()
        pdf.slide_06_route1()
        pdf.slide_07_route2()
        pdf.slide_08_pretest()
        pdf.slide_09_code()
        pdf.slide_10_wild()
        pdf.slide_11_cta()

        output_path = Path(__file__).parent / "diff-diff-bad-controls-carousel.pdf"
        pdf.output(str(output_path))
        print(f"PDF saved to: {output_path}")
    finally:
        pdf.cleanup()


if __name__ == "__main__":
    main()
