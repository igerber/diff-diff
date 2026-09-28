#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for the LWDiD launch (v3.10.0).

Mirrors the architecture of ``generate_mmm_carousel.py`` (magazine sidebar
with progress tick, light gradient background, split-color logo, footer
wordmark, dark slides, soft card shadows) but in POSTER MODE - the user
direction for this deck (2026-08-22): say less per slide with larger type so
the deck is legible on a phone. Poster type floors: body >= 16pt, captions
>= 13pt, headlines 32-54pt, code >= 15pt; one idea + one visual + roughly 25
words per slide.

Palette: "Ledger" - indigo primary (LWDiD / diff-diff accent) + gold
reserved for the SINGLE TREATED UNIT motif (the deck's visual thesis), on an
indigo-tint-to-white gradient. Two dark slides: the transformation reveal
(slide 3) and the code slide (slide 7). Truth values in simulated stats are
labeled "(simulated)".

Narrative spine (INVERTED arc - user decision 2026-08-22, round 2: lead
with the one-treated-unit drama, keep the transformation as the reveal; 8 slides with
the user-manual middle consolidated):

1.  Cover          -- "One treated unit. An exact p-value." California-style
                      one-gold-line-vs-donor-pool motif; subtitle names the
                      new estimator + the 3.10 release (version derived from
                      pyproject, never hardcoded).
2.  The gap        -- an estimate with one treated unit is easy (SyntheticDiD
                      does it); an exact p-value with one treated unit is
                      not. NO TWFE/Bacon villain arc (staggered bias is
                      already solved elsewhere in the library) and NO
                      few-treated-units-is-unsolved claim (SDID owns that
                      paradigm) - the gap is exact finite-sample INFERENCE.
                      The closing tagline is scoped to the classical
                      regression path (review R6: ipw/dr/psm exist; exact
                      t / RI live on the reg path). SyntheticDiD is named
                      once, neutrally, here only.
3.  The trick (DARK)- subtract each unit's own pre-treatment path; what's
                      left is a cross-section, where exact inference works
                      with one treated unit (N1 = 1; total N >= 3 per the
                      REGISTRY guards). Schematic centerpiece; the exact-t
                      assumption qualifier rides WITH the claim.
4.  Proof, N1 = 1  -- Prop 99, California vs 38 states: exact p = 0.0209
                      (df = 37) and randomization inference p = 0.0540,
                      both on the detrended spec (LW 2026 Table 3).
5.  At scale       -- Walmart entry, 1,277 counties, 14 cohorts: overall ATT
                      0.0199 (SE 0.0090); near-lead magnitudes stated as
                      bounds (review R9: never "flat", an inferential word)
                      WITH the far-lead disclosure. Shows it is not just a small-N1 device.
6.  Use it well    -- consolidated practice slide (typography, no charts):
                      demean-vs-detrend (drifting-units simulation + the
                      Prop 99 factor-of-two) and clustering (hc1 vs CR1 +
                      wild cluster bootstrap), closed by the diagnostics
                      call labeled DESCRIPTIVE, not an assumption test
                      (reviews R5/R6 - the tutorial's own framing).
7.  Code (DARK)    -- the Prop 99 fit + exact p + seeded randomization test,
                      >= 15pt, with the df/reps/seed literals rendered from
                      the pinned constants. The staggered caption scopes
                      honestly BOTH ways (review R4): the randomization test
                      is common-timing only, most staggered aggregates use
                      IF/bootstrap inference, and the eligible classical
                      composite KEEPS exact t (REGISTRY per-surface
                      reference-distribution Note).
8.  CTA            -- pip install; tutorial 31 mentioned ONCE, here.

Claim discipline (verified against the committed executed notebook
``docs/tutorials/31_lwdid.ipynb``):

- EVERY number on the deck is a committed, seed-locked tutorial-31 value
  (the real-data cells additionally assert their ``lwdid_ssc_ancillary``
  provenance in the notebook itself). notebook <-> deck sync is pinned by
  ``tests/test_lwdid_carousel_claims.py`` (parses this module's constants
  via ``ast`` and locates each on the committed notebook surface).
- The COVER motif is stylized art (one gold treated trajectory diverging
  from a donor pool), labeled "illustrative" on-slide; it plots no Prop 99
  data and makes no data claim - same convention as the CiC/MMM covers.
- Slide 2 mentions SyntheticDiD ONCE, neutrally, as the in-library
  estimator for the few-treated ESTIMATION paradigm; the deck never ranks
  it, never claims its inference is wrong, and carries no "only"/"first"
  absolutes (claims test bans them). The gap claim is scoped to EXACT
  finite-sample inference.
- The exact-t claim carries its FULL assumption stack ON THE SLIDE THAT
  MAKES IT (slide 3, reviews R1/R3/R5): the DiD identification assumptions
  (no anticipation, parallel- or unit-linear-trends restriction, overlap)
  plus classical errors (independent, mean-zero, conditionally normal,
  homoskedastic in the collapsed cross-section; LW 2026, REGISTRY
  "Small-sample (exact) inference layer") plus the sample-size guard (one
  treated unit needs at least two controls, N >= 3). Randomization inference is the
  complementary check testing
  Fisher's sharp null; slide 4 states its complete-randomization
  justification and shows exact p = 0.021 and RI p = 0.054 side by side
  WITHOUT calling them agreement (they straddle 0.05; the honesty is the
  point), and repeats the classical-error qualifier locally (review R8:
  the proof slide circulates on its own).
- Slide 4's "reproduce LW 2026 Table 3" claim is scoped to ATT/SE/exact p;
  the RI card is the seed-locked tutorial value under the authors'-package
  inclusive convention (REGISTRY RI note), said on-slide. The causal
  reading is stated as conditional on the detrending (CHT) and design
  assumptions, and "about a 20% reduction" is verbatim from the committed
  tutorial markdown.
- Slide 5's lead-flatness claim is SCOPED to the printed near-lead window
  (r = -7..-3, max |WATT| = 0.0059) AND discloses the full lead window's
  max (0.0329 at r = -22) on the same slide, LABELED as a magnitude
  (|WATT| - review R8: the committed surface pins max |WATT(r)|, not the
  signed estimate, so the deck never implies a sign) - the far lead is
  never flattened away. The chart draws only the committed values, and the
  near-lead band is drawn in DATA coordinates from the window constant
  (review R4: axes-fraction shading had silently covered r ~ -8.2..-2).
- Slide 6 frames demean-vs-detrend as diagnosis (``get_transformation_
  diagnostics``), scopes detrending to UNIT-SPECIFIC LINEAR drift (CHT is
  a linear-trends model, REGISTRY), and labels every simulated stat
  "(simulated)". The clustering copy's "barely half" derives from the
  committed SEs 0.131 vs 0.237 (claims test requires ratio <= 0.6); the
  Prop 99 "nearly a factor of two" derives from -0.4222 vs -0.2270
  (claims test bounds the ratio). The wild-cluster-bootstrap capability is
  scoped ON-SLIDE to clustered common-timing fits (review R3).
- Slide 7's code is the tutorial's own Prop 99 invocation (common-timing
  ``reg`` fit with ``vcov_type='classical'``) - the ONLY mode where
  ``randomization_test()`` is defined - with the tutorial's seed displayed
  so the shown call reproduces the shown p-value; the claims test pins the
  displayed fragments to the notebook's code cells.
- Tutorial 31 is referenced ONCE, on the CTA slide (house precedent).

Run with::

    python carousel/generate_lwdid_carousel.py

Produces ``carousel/diff-diff-lwdid-carousel.pdf``. Generation requires
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
# Cover subtitle needs the minor release ("3.10"), also derived, never typed.
RELEASE_LABEL = ".".join(_m.group(1).split(".")[:2])

TOTAL_SLIDES = 8

# -------------------------------------------------------------------------
# "Ledger" palette -- indigo primary, gold = the single treated unit.
# -------------------------------------------------------------------------
INDIGO = (67, 56, 202)  # #4338ca  primary accent (LWDiD / diff-diff)
INDIGO_DARK = (49, 46, 129)  # #312e81
INDIGO_BRIGHT = (129, 140, 248)  # #818cf8  accents on the dark slides
PERIWINKLE = (199, 210, 254)  # #c7d2fe  light fills / dark-slide text pop
GOLD = (202, 138, 4)  # #ca8a04  THE TREATED UNIT (chart grammar) + kickers
GOLD_BRIGHT = (250, 204, 21)  # #facc15  gold on the dark slides
INDIGO_TINT = (238, 242, 255)  # #eef2ff  gradient start
SHADOW = (203, 213, 225)  # #cbd5e1  soft card shadow

# Text + structural (shared with the other decks for legibility)
NAVY = (15, 23, 42)  # #0f172a  primary text; dark-slide gradient start
GRAY = (100, 116, 139)  # #64748b
LIGHT_GRAY = (148, 163, 184)  # #94a3b8
WHITE = (255, 255, 255)
DEEP_INDIGO = (30, 27, 75)  # #1e1b4b  dark-slide gradient end
PANEL_NAVY = (30, 41, 59)  # #1e293b  code panel fill
GOLD_CODE = (250, 204, 21)  # #facc15  code string/number literals
SLATE_CODE = (148, 163, 184)  # #94a3b8

INDIGO_HEX = "#4338ca"
INDIGO_BRIGHT_HEX = "#818cf8"
PERIWINKLE_HEX = "#c7d2fe"
GOLD_HEX = "#ca8a04"
GOLD_BRIGHT_HEX = "#facc15"
NAVY_HEX = "#0f172a"
GRAY_HEX = "#64748b"
LIGHT_GRAY_HEX = "#94a3b8"

# -------------------------------------------------------------------------
# Seed-locked numbers from the committed executed tutorial
# (docs/tutorials/31_lwdid.ipynb). Every constant below is located on the
# committed notebook surface by tests/test_lwdid_carousel_claims.py -
# never edit one without re-running it.
# -------------------------------------------------------------------------

# Walmart entry (real data; detrended staggered fit on log retail employment)
WM_ATT = (0.0199, 0.0090)  # overall ATT, SE
WM_COUNTIES, WM_YEARS, WM_COHORTS, WM_NEVER = 1277, 23, 14, 391
WM_COHORT_SPAN = (1986, 1999)
WM_WATT = {0: (0.0072, 0.0037), 1: (0.0322, 0.0051), 5: (0.0164, 0.0109)}
WM_NEAR_LEADS_MAX = 0.0059  # max |WATT(r)| over WM_NEAR_LEADS_WINDOW (as printed)
WM_NEAR_LEADS_WINDOW = (-7, -3)  # the printed near-lead window; band drawn from it
WM_ALL_LEADS_MAX, WM_ALL_LEADS_AT = 0.0329, -22  # full lead window, disclosed on-slide

# Prop 99 (real data; one treated state, exact + randomization inference)
P99_STATES, P99_YEARS = 39, 31
P99_SPAN = (1970, 2000)
P99_DETREND = (-0.2270, 0.0941)  # ATT, SE (log per-capita cigarette sales)
P99_EXACT_P, P99_DF = 0.0209, 37
P99_RI_P = 0.0540  # randomization inference, detrended fit (RI_REPS, RI_SEED)
RI_REPS, RI_SEED = 9999, 42  # the tutorial RI call; slide-7 code renders from these
P99_DEMEAN = (-0.4222, 0.1208)  # the demeaned spec (slide-6 sensitivity beat)

# Drifting-units simulation (demean vs detrend under differential trends)
TREND_DEMEAN = (3.376, 0.099)
TREND_DETREND = (1.098, 0.202)
TREND_TRUTH = 1.0

# Clustered-shock simulation (hc1 vs CR1 vs wild cluster bootstrap)
CL_ATT = 1.048
CL_NAIVE_SE = 0.131
CL_CR1_SE, CL_G = 0.237, 12
CL_TRUTH = 1.2
WCB_P, WCB_LO, WCB_HI = 0.0025, 0.503, 1.614

# Staggered simulation check (slide-5 caption)
SIM_STAG = (2.953, 0.060, 2.870)  # ATT, SE, truth


class LWDiDCarouselPDF(FPDF):
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
    # Magazine vertical sidebar -- INDIGO bar, GOLD progress tick.
    # -----------------------------------------------------------------
    def _draw_vertical_sidebar(self, slide_number, total=TOTAL_SLIDES, dark=False):
        bar_x = 14
        bar_y_top = 45
        bar_y_bottom = 275
        self.set_draw_color(*(INDIGO_BRIGHT if dark else INDIGO))
        self.set_line_width(0.6)
        self.line(bar_x, bar_y_top, bar_x, bar_y_bottom)

        ratio = (slide_number - 1) / (total - 1) if total > 1 else 0.0
        tick_y = bar_y_top + ratio * (bar_y_bottom - bar_y_top)
        self.set_draw_color(*(GOLD_BRIGHT if dark else GOLD))
        self.set_line_width(1.2)
        self.line(bar_x - 4, tick_y, bar_x + 7, tick_y)

    # -----------------------------------------------------------------
    # Backgrounds + footer
    # -----------------------------------------------------------------
    def light_gradient_background(self):
        """Indigo tint #eef2ff fading to white."""
        steps = 50
        r0, g0, b0 = INDIGO_TINT
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
        """Near-black navy #0f172a fading to deep indigo #1e1b4b."""
        steps = 50
        r0, g0, b0 = NAVY
        r1, g1, b1 = DEEP_INDIGO
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
        self.set_text_color(*(INDIGO_BRIGHT if dark else INDIGO))
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

    def _kicker(self, y, text, color=GOLD):
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
        """Split-color diff-diff logo with INDIGO middle dash."""
        self.set_xy(0, y)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="R")
        self.set_text_color(*INDIGO)
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
    # Cover motif -- one gold treated trajectory pulling away from a
    # donor pool of thin indigo lines. Stylized art, not a data claim
    # (it plots no Prop 99 data) - labeled "illustrative" on-slide.
    # -----------------------------------------------------------------
    def _render_cover_motif(self):
        rng = np.random.default_rng(1989)
        t = np.linspace(0, 10, 80)
        pre_end = 6.0

        fig, ax = plt.subplots(figsize=(10, 3.6))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        for _ in range(38):
            base = rng.uniform(1.4, 4.8)
            slope = rng.uniform(-0.03, 0.06)
            y = base + slope * t + rng.normal(0, 0.05, t.size)
            ax.plot(t, y, color=INDIGO_BRIGHT_HEX, linewidth=1.1, alpha=0.35)

        y_treat = 3.1 + 0.03 * t + rng.normal(0, 0.05, t.size)
        y_treat[t >= pre_end] -= 0.42 * (t[t >= pre_end] - pre_end)
        ax.plot(t, y_treat, color=GOLD_HEX, linewidth=3.6, zorder=5)

        ax.axvline(pre_end, color=GRAY_HEX, linewidth=1.4, linestyle=(0, (4, 3)))
        ax.text(pre_end - 0.12, 5.45, "treatment", fontsize=13.5, color=GRAY_HEX, ha="right")
        ax.text(
            10.15,
            float(y_treat[-1]),
            "the one\ntreated unit",
            fontsize=14,
            color=GOLD_HEX,
            fontweight="bold",
            va="center",
        )
        ax.text(
            10.15,
            4.55,
            "the donor pool",
            fontsize=13.5,
            color=INDIGO_BRIGHT_HEX,
            fontweight="bold",
            va="center",
        )
        ax.text(
            12.6, 0.3, "illustrative", fontsize=15, color=LIGHT_GRAY_HEX, style="italic", ha="right"
        )

        ax.set_xlim(-0.2, 12.7)
        ax.set_ylim(0.0, 5.9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=0.2)
        return self._save_fig(fig, transparent=True)

    # -----------------------------------------------------------------
    # Transformation schematic (slide 3, dark) -- labeled a schematic
    # on-slide; illustrates the rolling transformation, plots no data.
    # -----------------------------------------------------------------
    def _render_transform_schematic(self):
        rng = np.random.default_rng(3110)
        t = np.linspace(0, 10, 50)
        pre = t < 6

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.6))
        for ax in (ax1, ax2):
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

        # Left: raw panel - each unit with its own dashed pre-path fit,
        # extrapolated past treatment (the subtraction benchmark).
        for base, slope, color, lw in [
            (0.6, 0.06, INDIGO_BRIGHT_HEX, 2.0),
            (1.9, 0.14, INDIGO_BRIGHT_HEX, 2.0),
            (3.4, 0.02, INDIGO_BRIGHT_HEX, 2.0),
            (4.6, 0.10, GOLD_BRIGHT_HEX, 3.0),
        ]:
            y = base + slope * t + rng.normal(0, 0.07, t.size)
            if color == GOLD_BRIGHT_HEX:
                y[~pre] += 0.85
            ax1.plot(t, y, color=color, linewidth=lw)
            ax1.plot(
                t,
                base + slope * t,
                color=color,
                linewidth=1.4,
                linestyle=(0, (3, 2)),
                alpha=0.8,
            )
        ax1.axvline(6, color=LIGHT_GRAY_HEX, linewidth=1.2, linestyle=(0, (4, 3)))
        ax1.set_title(
            "fit each pre-treatment path (dashed)",
            fontsize=14.5,
            color="white",
            fontweight="bold",
        )
        ax1.set_ylim(0, 6.7)

        # Right: the collapsed cross-section (post-period residuals).
        xs = [-0.2, 0.0, 0.2, 1.0]
        ys = [0.05, -0.12, 0.10, 0.88]
        colors = [INDIGO_BRIGHT_HEX] * 3 + [GOLD_BRIGHT_HEX]
        for x, yv, c in zip(xs, ys, colors):
            ax2.plot([x], [yv], "o", color=c, markersize=16 if c == GOLD_BRIGHT_HEX else 13)
        ax2.axhline(0, color=LIGHT_GRAY_HEX, linewidth=1.2)
        ax2.text(
            0.0,
            -0.62,
            "controls",
            fontsize=14,
            color=INDIGO_BRIGHT_HEX,
            ha="center",
            fontweight="bold",
        )
        ax2.text(
            1.0,
            -0.62,
            "treated",
            fontsize=14,
            color=GOLD_BRIGHT_HEX,
            ha="center",
            fontweight="bold",
        )
        ax2.annotate(
            "the gap is the ATT",
            xy=(0.93, 0.84),
            xytext=(-0.42, 1.3),
            fontsize=14,
            color="white",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="white", lw=1.6),
        )
        ax2.set_title(
            "subtract it: the collapsed cross-section",
            fontsize=14.5,
            color="white",
            fontweight="bold",
        )
        ax2.set_xlim(-0.7, 1.6)
        ax2.set_ylim(-0.9, 1.55)

        fig.text(
            0.99,
            0.01,
            "schematic",
            fontsize=15,
            color=LIGHT_GRAY_HEX,
            ha="right",
            style="italic",
        )
        fig.tight_layout(pad=0.6)
        return self._save_fig(fig, transparent=True)

    # -----------------------------------------------------------------
    # Walmart event-study chart (slide 5) -- ONLY committed values: the
    # three printed WATT(r) points with their SEs, plus the near-leads
    # band at the printed max, labeled with its window.
    # -----------------------------------------------------------------
    def _render_walmart_event_study(self):
        fig, ax = plt.subplots(figsize=(10, 4.0))

        band = WM_NEAR_LEADS_MAX
        w0, w1 = WM_NEAR_LEADS_WINDOW
        ax.fill_between([w0, w1], -band, band, color=PERIWINKLE_HEX, alpha=0.55, linewidth=0)
        ax.text(
            (w0 + w1) / 2,
            band + 0.003,
            f"near leads (r = {w0}..{w1}):\nmax |WATT| = {band:.4f}\n(magnitude envelope)",
            fontsize=13.5,
            color=GRAY_HEX,
            ha="center",
        )
        ax.axhline(0, color=LIGHT_GRAY_HEX, linewidth=1.2)
        ax.axvline(-0.85, color=LIGHT_GRAY_HEX, linewidth=1.3, linestyle=(0, (4, 3)))

        for r, (eff, se) in WM_WATT.items():
            ax.errorbar(
                [r],
                [eff],
                yerr=[1.96 * se],
                color=INDIGO_HEX,
                fmt="o",
                markersize=12,
                capsize=6,
                linewidth=2.6,
                capthick=2.2,
            )
            ax.annotate(
                f"{eff:.3f}",
                xy=(r, eff),
                xytext=(r + 0.28, eff + 0.004),
                fontsize=14.5,
                fontweight="bold",
                color=INDIGO_HEX,
            )

        ax.text(
            2.9,
            -0.021,
            "WATT(r): average effect r years after entry,\ncohorts weighted by treated mass",
            fontsize=13,
            color=GRAY_HEX,
            ha="center",
        )

        ax.set_xlim(-8.2, 6.6)
        ax.set_ylim(-0.032, 0.052)
        ax.set_xticks([-7, -5, -3, 0, 1, 5])
        ax.set_xlabel("years since Walmart entry", fontsize=14, color=NAVY_HEX)
        ax.set_ylabel("effect on log retail employment", fontsize=13.5, color=NAVY_HEX)
        ax.tick_params(labelsize=13, colors=GRAY_HEX)
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

        self.centered_text(60, "One treated unit.", size=54)
        self.centered_text(90, "An exact p-value.", size=54, color=GOLD)

        motif_path, _pw, _ph = self._render_cover_motif()
        motif_w = 222
        self.image(motif_path, (WIDTH - motif_w) / 2, 124, motif_w)

        self.centered_text(
            212,
            f"New in diff-diff {RELEASE_LABEL}: the LWDiD estimator.",
            size=22,
            color=INDIGO,
        )
        self.centered_text(
            230, "Rolling-transformation DiD (Lee & Wooldridge).", size=19, bold=False
        )
        self.add_footer()

    def slide_02_gap(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(2)

        self._kicker(42, "Why Another Estimator")

        self.centered_text(80, "An estimate with", size=42)
        self.centered_text(104, "one treated unit is easy.", size=42)
        self.centered_text(132, "SyntheticDiD does it.", size=22, bold=False, color=GRAY)

        self.set_draw_color(*GOLD)
        self.set_line_width(1.4)
        self.line(WIDTH / 2 - 30, 158, WIDTH / 2 + 30, 158)

        self.centered_text(176, "An exact p-value with", size=42, color=INDIGO)
        self.centered_text(200, "one treated unit is not.", size=42, color=INDIGO)

        self.set_xy(30, 244)
        self.set_font("Helvetica", "", 18)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 60,
            9.5,
            "LWDiD closes that gap on its classical path: one ordinary "
            "cross-sectional regression.",
            align="C",
        )
        self.add_footer()

    def slide_03_trick(self):
        self.add_page()
        self.dark_gradient_background()
        self._draw_vertical_sidebar(3, dark=True)

        self._kicker(32, "The Trick", color=GOLD_BRIGHT)

        self.centered_text(58, "Subtract each unit's own", size=44, color=WHITE)
        self.centered_text(84, "pre-treatment path.", size=44, color=GOLD_BRIGHT)

        schematic_path, _pw, _ph = self._render_transform_schematic()
        sch_w = 226
        self.image(schematic_path, (WIDTH - sch_w) / 2, 110, sch_w)

        self.centered_text(218, "What's left is a cross-section -", size=22, color=WHITE)
        self.centered_text(
            234, "where exact inference works with one treated unit.", size=22, color=GOLD_BRIGHT
        )

        self.set_xy(28, 256)
        self.set_font("Helvetica", "I", 13)
        self.set_text_color(*PERIWINKLE)
        self.multi_cell(
            WIDTH - 56,
            6.5,
            "Assumes the DiD design assumptions (no anticipation, the parallel- "
            "or unit-linear-trends restriction, overlap) plus classical errors: "
            "independent, mean-zero, conditionally normal, homoskedastic in the "
            "collapsed cross-section. One treated unit needs at least two "
            "controls (N >= 3).",
            align="C",
        )
        self.add_footer(dark=True)

    def slide_04_prop99(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(4)

        self._kicker(30, "Proof With One Treated Unit")
        self.centered_text(48, "Proposition 99:", size=40)
        self.centered_text(72, f"California vs. {P99_STATES - 1} states.", size=40)

        card_w, card_h = 106, 74
        gap = 14
        x0 = (WIDTH - card_w * 2 - gap) / 2
        self._stat_card(
            x0,
            104,
            card_w,
            card_h,
            f"p = {P99_EXACT_P:.3f}",
            [("exact t", True), (f"df = {P99_DF}", False)],
            INDIGO,
        )
        self._stat_card(
            x0 + card_w + gap,
            104,
            card_w,
            card_h,
            f"p = {P99_RI_P:.3f}",
            [("randomization", True), ("inference", False)],
            GOLD,
        )

        self.set_xy(0, 192)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*NAVY)
        self.cell(
            WIDTH,
            10,
            f"ATT {P99_DETREND[0]:.3f} (SE {P99_DETREND[1]:.3f}) on log cigarette sales -",
            align="C",
        )
        self.centered_text(208, "about a 20% reduction.", size=22, color=INDIGO)

        self.set_xy(25, 232)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 50,
            7,
            f"{P99_STATES} states x {P99_YEARS} years ({P99_SPAN[0]}-{P99_SPAN[1]}), "
            "detrended; ATT, SE, and exact p reproduce LW 2026 Table 3. RI tests "
            "Fisher's sharp null under the complete-randomization assignment "
            "mechanism (authors' package convention). Two roads to inference with a "
            "single treated state - conditional on the detrending (CHT) and design "
            "assumptions; the exact t additionally assumes independent, "
            "conditionally normal, homoskedastic collapsed errors.",
            align="C",
        )
        self.add_footer()

    def slide_05_walmart(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(5)

        self._kicker(30, "And At Scale")
        self.centered_text(48, f"{WM_COUNTIES:,} counties.", size=40)
        self.centered_text(72, f"{WM_COHORTS} Walmart entry cohorts.", size=40)

        chart_path, _pw, _ph = self._render_walmart_event_study()
        chart_w = 222
        self.image(chart_path, (WIDTH - chart_w) / 2, 96, chart_w)

        self.set_xy(0, 206)
        self.set_font("Helvetica", "B", 26)
        self.set_text_color(*INDIGO)
        self.cell(WIDTH, 12, f"overall ATT {WM_ATT[0]:.4f}  (SE {WM_ATT[1]:.4f})", align="C")

        self.set_xy(25, 226)
        self.set_font("Helvetica", "", 16)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 50,
            8.5,
            f"Staggered adoption ({WM_COHORT_SPAN[0]}-{WM_COHORT_SPAN[1]}), "
            f"{WM_NEVER} never-treated counties, detrended. Near leads stay below "
            f"|WATT| = {WM_NEAR_LEADS_MAX:.3f}; distant leads reach "
            f"|WATT| = {WM_ALL_LEADS_MAX:.3f} "
            f"(at r = {WM_ALL_LEADS_AT}).",
            align="C",
        )
        self.set_xy(0, 254)
        self.set_font("Helvetica", "", 13)
        self.set_text_color(*GRAY)
        self.cell(
            WIDTH,
            6,
            f"Simulation check: staggered ATT {SIM_STAG[0]:.3f} (SE {SIM_STAG[1]:.3f}) "
            f"vs truth {SIM_STAG[2]:.2f} (simulated).",
            align="C",
        )
        self.add_footer()

    def slide_06_use_it_well(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(6)

        self._kicker(34, "Use It Well")

        self.centered_text(58, "Units drifting apart? Detrend.", size=32)
        self.set_xy(30, 78)
        self.set_font("Helvetica", "", 16)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 60,
            8.5,
            f"Detrending fits unit-specific linear drift. Under drift, demeaning "
            f"estimated {TREND_DEMEAN[0]:.2f} vs detrending {TREND_DETREND[0]:.2f} - "
            f"truth {TREND_TRUTH:.1f} (simulated). On Prop 99 the two disagree by "
            f"nearly a factor of two ({P99_DEMEAN[0]:.2f} vs {P99_DETREND[0]:.2f}).",
            align="C",
        )

        self.set_draw_color(*GOLD)
        self.set_line_width(1.4)
        self.line(WIDTH / 2 - 30, 130, WIDTH / 2 + 30, 130)

        self.centered_text(146, "Shared shocks? Cluster.", size=32)
        self.set_xy(30, 166)
        self.set_font("Helvetica", "", 16)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 60,
            8.5,
            f"The naive SE was barely half the honest one: {CL_NAIVE_SE:.3f} "
            f"unclustered vs {CL_CR1_SE:.3f} CR1 with G = {CL_G} regions "
            f"(simulated). Few clusters? Wild cluster bootstrap is built in "
            f"for clustered common-timing reg fits: "
            f"p = {WCB_P:.4f}, 95% CI [{WCB_LO:.3f}, {WCB_HI:.3f}].",
            align="C",
        )

        self.set_xy(0, 226)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*INDIGO)
        self.cell(
            WIDTH,
            8,
            "Inspect the transformation - descriptive, not an assumption test:",
            align="C",
        )
        self.set_xy(0, 240)
        self.set_font("Courier", "B", 16)
        self.set_text_color(*NAVY)
        self.cell(WIDTH, 8, "get_transformation_diagnostics()", align="C")
        self.add_footer()

    def slide_07_code(self):
        self.add_page()
        self.dark_gradient_background()
        self._draw_vertical_sidebar(7, dark=True)

        self._kicker(32, "The Whole Workflow", color=GOLD_BRIGHT)
        self.centered_text(52, "A fit, an exact p-value,", size=36, color=WHITE)
        self.centered_text(74, "a permutation test.", size=36, color=WHITE)

        kw, s, g, w = INDIGO_BRIGHT, GOLD_CODE, SLATE_CODE, WHITE
        token_lines = [
            [
                ("fit", w),
                (" = ", g),
                ("LWDiD", kw),
                ("(rolling=", w),
                ('"detrend"', s),
                (",", w),
            ],
            [("            vcov_type=", w), ('"classical"', s), (").fit(", w)],
            [("    prop99, outcome=", w), ('"lcigsale"', s), (",", w)],
            [("    unit=", w), ('"state"', s), (", time=", w), ('"year"', s), (",", w)],
            [("    treatment=", w), ('"treated"', s), (")", w)],
            [("", w)],
            [("fit.p_value", w), (f"        # exact t, df = {P99_DF}", g)],
            [
                ("fit.randomization_test(n_reps=", w),
                (str(RI_REPS), s),
                (", seed=", w),
                (str(RI_SEED), s),
                (")", w),
            ],
        ]
        margin = 34
        self._add_code_block(
            margin, 98, WIDTH - margin * 2, token_lines, font_size=15, line_height=13
        )

        self.set_xy(0, 242)
        self.set_font("Helvetica", "B", 19)
        self.set_text_color(*GOLD_BRIGHT)
        self.cell(WIDTH, 8, "Staggered? Same fit - just add first_treat=.", align="C")
        self.set_xy(28, 256)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*PERIWINKLE)
        self.multi_cell(
            WIDTH - 56,
            7,
            "The randomization test is common-timing reg only; most staggered "
            "aggregates use influence-function or bootstrap inference (an "
            "eligible classical composite keeps exact t).",
            align="C",
        )
        self.add_footer(dark=True)

    def slide_08_cta(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(8)

        self.draw_split_logo(38, size=34)

        self.centered_text(80, "Panels in.", size=46)
        self.centered_text(108, "Cross-sections out.", size=46, color=INDIGO)

        chip_w = 150
        chip_x = (WIDTH - chip_w) / 2
        self.set_fill_color(*PANEL_NAVY)
        self.rect(chip_x, 142, chip_w, 22, "F")
        self.set_xy(chip_x, 148)
        self.set_font("Courier", "B", 19)
        self.set_text_color(*GOLD_CODE)
        self.cell(chip_w, 10, "pip install diff-diff", align="C")

        self.set_xy(30, 184)
        self.set_font("Helvetica", "", 17)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 60,
            9,
            "Tutorial 31 reproduces every number on this deck - Prop 99 and " "Walmart included:",
            align="C",
        )
        self.centered_text(
            218,
            "diff-diff.readthedocs.io/en/latest/tutorials/31_lwdid.html",
            size=14,
            color=INDIGO_DARK,
        )

        self.centered_text(242, "github.com/igerber/diff-diff", size=17)

        self.set_xy(0, 264)
        self.set_font("Helvetica", "I", 13)
        self.set_text_color(*GRAY)
        self.cell(
            WIDTH,
            6,
            "Lee & Wooldridge (2025; 2026) - SSRN 4516518 + 5325686.",
            align="C",
        )
        self.add_footer()


def main():
    pdf = LWDiDCarouselPDF()
    try:
        pdf.slide_01_cover()
        pdf.slide_02_gap()
        pdf.slide_03_trick()
        pdf.slide_04_prop99()
        pdf.slide_05_walmart()
        pdf.slide_06_use_it_well()
        pdf.slide_07_code()
        pdf.slide_08_cta()

        output_path = Path(__file__).parent / "diff-diff-lwdid-carousel.pdf"
        pdf.output(str(output_path))
        print(f"PDF saved to: {output_path}")
    finally:
        pdf.cleanup()


if __name__ == "__main__":
    main()
