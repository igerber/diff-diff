#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for the ChangesInChanges (CiC) launch.

Mirrors the architecture of ``generate_lpdid_carousel.py`` (magazine sidebar
with progress tick, light gradient background, split-color logo, footer
wordmark, serif pull quotes, one dark twist slide, soft card shadows, phone
type floors: body >= 13pt, headlines 38-52) with the "Tide" palette: deep
ocean navy / teal / seafoam on a light aqua gradient, coral kickers, amber
highlight band for the significant quantile range. Cover motif:
distribution-as-water - the counterfactual and treated spend densities, the
lower tail visibly lifting.

Narrative spine (villain-first: the effect mean DiD cannot see; grounded in
the Athey-Imbens 2006 paper):

1.  Cover          -- "The effect your DiD can't see." Real programs move
                      parts of the distribution; the mean averages them away
2.  The problem    -- a real effect, averaged into nothing: mean DiD $0.22
                      (p = 0.90) vs a known true effect of $3.01
3.  Twist (DARK)   -- it gets worse: same data in logs reads +14.0%
                      (p = 0.001); the scale choice IS the assumption
4.  The paper      -- Athey & Imbens (2006): compare ranks, not scales;
                      verbatim abstract pull quote + the equivariance
                      receipt (dollars fit == exp(logs fit))
5.  The math       -- the Theorem 3.1 composition, annotated step by step,
                      plus the QTE definition
6.  The payoff     -- the QTE profile: the same program, now visible
7.  Joint claims   -- sup-t uniform bands: "the bottom half moved" as one
                      simultaneous claim
8.  The code       -- fit() to the whole distribution in three lines
9.  Covariates     -- confidently wrong 9.38 -> truth-covering 6.53 with
                      covariates=['tenure']
10. Production     -- feature grid + validation strip (qte 1.3.1 parity)
11. CTA            -- pip install; the ONE tutorial mention; GitHub, RTD

Claim discipline (verified against docs/methodology/REGISTRY.md, the parity
suite header in tests/test_changes_in_changes_parity.py, and Tutorial 27 +
its drift test):

- EVERY number on the deck is the tutorial's seed-locked value (simulated
  example with truth known by construction - stated on-slide). Values
  marked "pin" below are asserted by
  tests/test_t27_cic_distributional_effects_drift.py; values marked
  "surface" are quoted from the committed executed notebook output.
  tests/test_cic_carousel_claims.py syncs the deck constants against that
  committed notebook surface.
- Tutorial 27 is referenced ONCE, on the CTA slide - never as the framing
  device of earlier slides (user decision, 2026-07-19).
- The equivariance claim carries "(unconditional fits)" ON the slide; the
  covariate branch's linear quantile regressions are not equivariant, and
  ATT/means are never claimed transform-invariant (means do not commute
  with transforms for any estimator).
- Parity wording: unconditional end-to-end point estimates MATCH the R qte
  1.3.1 golden files to 1e-10 (means/type-7 interpolation differ from R by
  ~1 ulp); "bit-exact" is used ONLY for the type-1 quantile arithmetic
  pinned at atol=0. The covariate branch is layered parity per the REGISTRY
  Deviation note: conventions exact given R inputs, LP solver proven
  optimal per tau - never called an "exact port" end to end.
- The market trend in the simulated example is a POWER-LAW transform
  (y -> e^a * y^1.06): additive on neither scale across differently
  composed groups. Never described as "multiplicative" (that would be
  log-parallel).
- The QTE profile fades to "nothing distinguishable from zero above the
  70th percentile" (inference-scoped; the pinned point estimate at
  tau = 0.75 is +$0.59, so "~0 above the median" would overstate).
- Above the median the uniform bands are silent, not exonerating; the
  tau = 0.55 pointwise blip (p = 0.044, surface) is named as what joint
  bands protect against.
- The pull quote is VERBATIM from the paper's abstract (Econometrica
  74(2), p. 431), checked word-for-word against the published PDF.
- Bootstrap-only inference and the fixed-95% sup-t band level are stated
  as qte-parity conventions. NO speed claims. QDiD is deliberately not
  mentioned anywhere on this deck (user decision, 2026-07-18).

Run with::

    python carousel/generate_cic_carousel.py

Produces ``carousel/diff-diff-cic-carousel.pdf``. Generation requires
``fpdf2``, ``Pillow``, and ``matplotlib`` (carousel-only dependencies, not
part of the library's install extras) plus ``scipy`` (already a library
runtime dependency) for the cover-density KDE.
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

plt.rcParams["mathtext.fontset"] = "cm"

# Page dimensions (4:5 portrait -- same as the other decks)
WIDTH = 270  # mm
HEIGHT = 337.5  # mm

# Version label derived from pyproject.toml (regex, not tomllib: py>=3.9).
_PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"
_m = re.search(r'^version\s*=\s*"([^"]+)"', _PYPROJECT.read_text(encoding="utf-8"), re.MULTILINE)
if _m is None:
    raise RuntimeError(f"could not parse project version from {_PYPROJECT}")
VERSION_LABEL = "v" + _m.group(1)

TOTAL_SLIDES = 11

# -------------------------------------------------------------------------
# "Tide" palette
# -------------------------------------------------------------------------
TEAL = (13, 148, 136)  # #0d9488  primary accent (after / CiC)
TEAL_DARK = (15, 118, 110)  # #0f766e
TEAL_BRIGHT = (45, 212, 191)  # #2dd4bf  accents on the dark slide
SEAFOAM = (153, 246, 228)  # #99f6e4  light fills / dark-slide text pop
OCEAN = (22, 66, 91)  # #16425b  deep ocean navy (before / counterfactual)
CORAL = (225, 29, 72)  # #e11d48  kickers + warning accents
AMBER = (245, 158, 11)  # #f59e0b  significant-band highlight
AQUA_TINT = (236, 254, 255)  # #ecfeff  gradient start
SHADOW = (203, 213, 225)  # #cbd5e1  soft card shadow

# Text + structural (shared with the other decks for legibility)
NAVY = (15, 23, 42)  # #0f172a  primary text; dark-slide gradient start
GRAY = (100, 116, 139)  # #64748b
LIGHT_GRAY = (148, 163, 184)  # #94a3b8
WHITE = (255, 255, 255)
DARK_OCEAN = (22, 46, 66)  # #162e42  dark-slide gradient end
AMBER_CODE = (252, 211, 77)  # #fcd34d  code string literals
SLATE_CODE = (148, 163, 184)  # #94a3b8

TEAL_HEX = "#0d9488"
TEAL_BRIGHT_HEX = "#2dd4bf"
SEAFOAM_HEX = "#99f6e4"
OCEAN_HEX = "#16425b"
CORAL_HEX = "#e11d48"
AMBER_HEX = "#f59e0b"
NAVY_HEX = "#0f172a"
GRAY_HEX = "#64748b"
LIGHT_GRAY_HEX = "#94a3b8"

# -------------------------------------------------------------------------
# Seed-locked numbers from Tutorial 27 (docs/tutorials/
# 27_cic_distributional_effects.ipynb, seed 27, n=901/cell).
#
# "pin"     = asserted by tests/test_t27_cic_distributional_effects_drift.py
# "surface" = quoted from the committed executed notebook output (the same
#             rendered surface the drift test cross-checks)
# -------------------------------------------------------------------------
SEED_LABEL = 27  # pin (tutorial DGP seed)
N_CELL_LABEL = 901  # pin (obs per group x period cell)
BLIP_TAU, BLIP_P = 0.55, 0.044  # surface (pointwise blip the bands kill)
MAX_REL_DIFF_LABEL = "0.00e+00"  # surface (receipt slide, this run)
QR_GRID_TAUS = 99  # pin (covariate-branch QR tau grid size)
DID_ATT = 0.22  # pin (mean DiD, dollars)
DID_P = 0.90  # pin
DID_CI = (-3.18, 3.62)  # surface (tutorial prose + summary output)
TRUE_MEAN_EFFECT = 3.01  # pin (Monte-Carlo truth, known by construction)
CIC_ATT = 3.05  # pin (3.0476 in the output table)
CIC_ATT_P = 0.004  # surface
LOG_DID_PCT = 14.0  # pin (att_log 0.1307 -> +14.0%)
LOG_DID_P = 0.001  # surface (tutorial prose; drift pin is p < 0.005)

# Full 19-point QTE profile: (tau, qte, conf_low, conf_high) -- surface
# (summary() table in the committed output; taus 0.05/0.25/0.50/0.75 pinned).
QTE_ROWS = [
    (0.05, 4.9087, 3.2914, 6.5260),
    (0.10, 5.4037, 3.6154, 7.1920),
    (0.15, 5.4575, 3.5787, 7.3364),
    (0.20, 5.3724, 3.7483, 6.9965),
    (0.25, 5.4433, 3.6953, 7.1913),
    (0.30, 5.5808, 3.5504, 7.6112),
    (0.35, 4.2059, 2.1936, 6.2181),
    (0.40, 4.1270, 2.2871, 5.9668),
    (0.45, 4.1575, 2.4318, 5.8831),
    (0.50, 3.8598, 1.6632, 6.0564),
    (0.55, 2.9191, 0.0782, 5.7599),
    (0.60, 1.7319, -0.9116, 4.3755),
    (0.65, 0.7939, -1.9821, 3.5699),
    (0.70, 0.3729, -2.7341, 3.4799),
    (0.75, 0.5906, -2.5447, 3.7258),
    (0.80, 1.3649, -1.8030, 4.5327),
    (0.85, 1.0424, -3.7510, 5.8357),
    (0.90, -0.3910, -5.0450, 4.2631),
    (0.95, 2.7981, -3.8035, 9.3997),
]
QTE_SES = [
    0.8252,
    0.9124,
    0.9586,
    0.8286,
    0.8918,
    1.0359,
    1.0267,
    0.9387,
    0.8804,
    1.1207,
    1.4494,
    1.3488,
    1.4164,
    1.5852,
    1.5996,
    1.6163,
    2.4456,
    2.3746,
    3.3682,
]  # surface (summary() table)
SUP_T_CRIT = 3.260  # surface ("sup-t critical value: 3.260" in the output)
# Uniform bands exclude zero for EXACTLY tau = 0.05..0.50 -- pin.

# Receipt (levels-fit vs exp(log-fit) counterfactual quantiles) -- surface.
RECEIPT_ROWS = [
    (0.15, 13.9021),
    (0.50, 20.3166),
    (0.75, 29.9053),
    (0.95, 45.2715),
]


# Covariate section -- pins.
COV_TRUTH = 6.0
COV_UNC = (9.38, 8.32, 10.45)  # ATT, CI low, CI high (CI excludes truth)
COV_COND = (6.53, 5.35, 7.70)  # ATT, CI low, CI high (CI covers truth)

# DGP shape constants for the cover density motif ONLY (same values as the
# tutorial's locked DGP; the motif is illustrative art, not a data claim).
_MU_LOG, _SIGMA_LOG = 3.4, 0.75
_GAMMA, _ALPHA = 1.06, -0.17
_U_LO, _U_HI = 0.05, 0.90
_BETA_A, _BETA_B = 1.2, 2.2
_LIFT_MAX, _LIFT_MID, _LIFT_SCALE = 0.65, 0.22, 0.07


class CiCCarouselPDF(FPDF):
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
    # Magazine vertical sidebar -- TEAL bar, CORAL progress tick.
    # -----------------------------------------------------------------
    def _draw_vertical_sidebar(self, slide_number, total=TOTAL_SLIDES, dark=False):
        bar_x = 14
        bar_y_top = 45
        bar_y_bottom = 275
        self.set_draw_color(*(TEAL_BRIGHT if dark else TEAL))
        self.set_line_width(0.6)
        self.line(bar_x, bar_y_top, bar_x, bar_y_bottom)

        ratio = (slide_number - 1) / (total - 1) if total > 1 else 0.0
        tick_y = bar_y_top + ratio * (bar_y_bottom - bar_y_top)
        self.set_draw_color(*CORAL)
        self.set_line_width(1.2)
        self.line(bar_x - 4, tick_y, bar_x + 7, tick_y)

    # -----------------------------------------------------------------
    # Backgrounds + footer
    # -----------------------------------------------------------------
    def light_gradient_background(self):
        """Aqua #ecfeff fading to white."""
        steps = 50
        r0, g0, b0 = AQUA_TINT
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
        """Near-black navy #0f172a fading to dark ocean #162e42 (twist)."""
        steps = 50
        r0, g0, b0 = NAVY
        r1, g1, b1 = DARK_OCEAN
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
        self.set_text_color(*(TEAL_BRIGHT if dark else TEAL))
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

    def _kicker(self, y, text, color=CORAL):
        """Editorial section label: letter-spaced caps with flanking rules."""
        spaced = " ".join(text.upper())
        self.set_font("Helvetica", "B", 13)
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

    def _pull_quote(self, y, text, attribution, size=15, width_frac=0.70, dark=False):
        """Serif pull quote (verbatim paper text) with an oversized coral mark."""
        qw = WIDTH * width_frac
        qx = (WIDTH - qw) / 2

        self.set_xy(qx - 6, y - 7)
        self.set_font("Helvetica", "B", 46)
        self.set_text_color(*CORAL)
        self.cell(14, 14, '"')

        self.set_xy(qx, y)
        self.set_font("Times", "I", size)
        self.set_text_color(*(WHITE if dark else NAVY))
        self.multi_cell(qw, size * 0.52, text, align="C")
        end_y = self.get_y() + 4

        self.set_xy(0, end_y)
        self.set_font("Helvetica", "", 12)
        self.set_text_color(*(LIGHT_GRAY if dark else GRAY))
        self.cell(WIDTH, 6, attribution, align="C")
        return end_y + 8

    def draw_split_logo(self, y, size=18):
        """Split-color diff-diff logo with TEAL middle dash."""
        self.set_xy(0, y)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="R")
        self.set_text_color(*TEAL)
        self.cell(10, 10, "-", align="C")
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="L")

    # -----------------------------------------------------------------
    # Shadowed card helpers
    # -----------------------------------------------------------------
    def _shadow_rect(self, x, y, w, h):
        self.set_fill_color(*SHADOW)
        self.rect(x + 1.4, y + 1.4, w, h, "F")

    def _stat_card(self, x, y, w, h, headline, sub_lines, accent, headline_size=34):
        """One shadowed stat card: big number + small caption lines."""
        self._shadow_rect(x, y, w, h)
        self.set_fill_color(*WHITE)
        self.set_draw_color(220, 220, 220)
        self.set_line_width(0.5)
        self.rect(x, y, w, h, "DF")
        self.set_fill_color(*accent)
        self.rect(x, y, w, 3.2, "F")

        self.set_xy(x, y + 12)
        self.set_font("Helvetica", "B", headline_size)
        self.set_text_color(*accent)
        self.cell(w, 14, headline, align="C")

        ly = y + 32
        for line, emphasize in sub_lines:
            self.set_xy(x + 6, ly)
            self.set_font("Helvetica", "B" if emphasize else "", 13)
            self.set_text_color(*(NAVY if emphasize else GRAY))
            self.cell(w - 12, 7, line, align="C")
            ly += 9.5

    # -----------------------------------------------------------------
    # Figure helpers (matplotlib -> PNG -> fpdf image)
    # -----------------------------------------------------------------
    def _save_fig(self, fig, dpi=200, transparent=False, facecolor="white"):
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=transparent,
            facecolor=None if transparent else facecolor,
        )
        plt.close(fig)
        with PILImage.open(path) as img:
            pw, ph = img.size
        self._temp_files.append(path)
        return path, pw, ph

    def _place_image_centered(self, path, pw, ph, y, max_w=200):
        aspect = ph / pw
        display_w = min(max_w, WIDTH * 0.8)
        display_h = display_w * aspect
        self.image(path, (WIDTH - display_w) / 2, y, display_w)
        return display_h

    # -----------------------------------------------------------------
    # Cover motif -- counterfactual (ocean) vs treated (teal) spend
    # densities from the tutorial's locked DGP; the lower tail lifts.
    # Illustrative art (smoothed population densities), not a data claim.
    # -----------------------------------------------------------------
    def _render_cover_densities(self):
        from scipy import stats as sps

        rng = np.random.default_rng(27)
        u = _U_LO + (_U_HI - _U_LO) * rng.beta(_BETA_A, _BETA_B, 200_000)
        y_cf = np.exp(_ALPHA + _GAMMA * (_MU_LOG + _SIGMA_LOG * sps.norm.ppf(u)))
        lift = _LIFT_MAX / (1.0 + np.exp((u - _LIFT_MID) / _LIFT_SCALE))
        y_tr = y_cf * (1.0 + lift)

        grid = np.linspace(np.log(5), np.log(120), 300)
        kde_cf = sps.gaussian_kde(np.log(y_cf), bw_method=0.45)(grid)
        kde_tr = sps.gaussian_kde(np.log(y_tr), bw_method=0.45)(grid)

        fig, ax = plt.subplots(figsize=(10, 3.2))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        # Amber wash under the region where the lift lives (lower half of
        # the treated distribution -- up to its median).
        med_tr = np.log(np.median(y_tr))
        ax.axvspan(grid[0], med_tr, color=AMBER_HEX, alpha=0.07, linewidth=0)

        ax.plot(grid, kde_cf, color=OCEAN_HEX, linewidth=3.2, alpha=0.55)
        ax.fill_between(grid, 0, kde_cf, color=OCEAN_HEX, alpha=0.08, linewidth=0)
        ax.plot(grid, kde_tr, color=TEAL_HEX, linewidth=3.4, alpha=0.85)
        ax.fill_between(grid, 0, kde_tr, color=TEAL_HEX, alpha=0.10, linewidth=0)

        ax.text(
            grid[30],
            kde_cf.max() * 0.92,
            "without the program",
            fontsize=13,
            color=OCEAN_HEX,
            alpha=0.9,
        )
        ax.text(
            med_tr,
            kde_tr.max() * 1.04,
            "with it",
            fontsize=13,
            color=TEAL_HEX,
            fontweight="bold",
        )

        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(0, max(kde_cf.max(), kde_tr.max()) * 1.18)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=0.2)
        return self._save_fig(fig, transparent=True)

    # -----------------------------------------------------------------
    # QTE profile figure (slide 5) -- pointwise CIs.
    # -----------------------------------------------------------------
    def _render_qte_profile(self):
        taus = np.array([r[0] for r in QTE_ROWS])
        qte = np.array([r[1] for r in QTE_ROWS])
        lo = np.array([r[2] for r in QTE_ROWS])
        hi = np.array([r[3] for r in QTE_ROWS])

        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.axvspan(0.03, 0.50, color=AMBER_HEX, alpha=0.10, linewidth=0)
        ax.fill_between(
            taus, lo, hi, color=SEAFOAM_HEX, alpha=0.75, linewidth=0, label="pointwise 95% CI"
        )
        ax.plot(
            taus,
            qte,
            "o-",
            color=TEAL_HEX,
            linewidth=2.6,
            markersize=6,
            label="CiC quantile treatment effect",
        )
        ax.axhline(0, color=GRAY_HEX, linewidth=1)
        ax.axhline(
            DID_ATT,
            color=CORAL_HEX,
            linewidth=1.4,
            linestyle=":",
            label=f"mean DiD (${DID_ATT:.2f})",
        )

        for tau, val in ((0.05, 4.91), (0.25, 5.44), (0.50, 3.86)):
            ax.annotate(
                f"+${val:.2f}",
                xy=(tau, val),
                xytext=(tau - 0.012, val + 2.1),
                fontsize=12.5,
                fontweight="bold",
                color=NAVY_HEX,
            )

        ax.set_xlabel("spend quantile (tau)", fontsize=13, color=NAVY_HEX)
        ax.set_ylabel("effect ($ / month)", fontsize=13, color=NAVY_HEX)
        ax.set_xlim(0.02, 0.98)
        ax.set_ylim(-5.8, 11.2)
        ax.tick_params(labelsize=11, colors=GRAY_HEX)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(LIGHT_GRAY_HEX)
        ax.legend(loc="upper right", fontsize=11, frameon=False)
        fig.tight_layout(pad=0.3)
        return self._save_fig(fig, facecolor="white")

    # -----------------------------------------------------------------
    # Uniform-bands figure (slide 6) -- sup-t bands + exclusion strip.
    # Bands = qte +/- SUP_T_CRIT * se (the results object's uniform_bands()
    # construction; crit value from the committed output).
    # -----------------------------------------------------------------
    def _render_uniform_bands(self):
        taus = np.array([r[0] for r in QTE_ROWS])
        qte = np.array([r[1] for r in QTE_ROWS])
        ses = np.array(QTE_SES)
        blo = qte - SUP_T_CRIT * ses
        bhi = qte + SUP_T_CRIT * ses
        excluded = (blo > 0) | (bhi < 0)

        fig, (ax, strip) = plt.subplots(
            2, 1, figsize=(10, 4.9), height_ratios=[5, 0.7], sharex=True
        )
        ax.axvspan(0.03, 0.50, color=AMBER_HEX, alpha=0.10, linewidth=0)
        ax.plot(taus, blo, color=OCEAN_HEX, linewidth=1.8)
        ax.plot(taus, bhi, color=OCEAN_HEX, linewidth=1.8, label="sup-t uniform band (fixed 95%)")
        ax.fill_between(taus, blo, bhi, color=OCEAN_HEX, alpha=0.06, linewidth=0)
        ax.plot(taus, qte, "o-", color=TEAL_HEX, linewidth=2.4, markersize=5.5, label="CiC QTE")
        ax.axhline(0, color=GRAY_HEX, linewidth=1)
        ax.set_ylim(-10.5, 15.0)
        ax.tick_params(labelsize=11, colors=GRAY_HEX)
        ax.set_ylabel("effect ($ / month)", fontsize=13, color=NAVY_HEX)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        from matplotlib.lines import Line2D

        extra = [
            Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markersize=9,
                color=TEAL_HEX,
                label="band excludes zero",
            ),
            Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markersize=9,
                color="#d8e2ea",
                label="band includes zero",
            ),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles + extra,
            labels + [e.get_label() for e in extra],
            loc="upper right",
            fontsize=10.5,
            frameon=False,
        )

        strip.scatter(
            taus[excluded],
            np.zeros(excluded.sum()),
            marker="s",
            s=110,
            color=TEAL_HEX,
            label="band excludes zero",
        )
        strip.scatter(
            taus[~excluded],
            np.zeros((~excluded).sum()),
            marker="s",
            s=110,
            color="#d8e2ea",
            label="band includes zero",
        )
        strip.set_ylim(-1, 1)
        strip.set_yticks([])
        strip.set_xlabel("spend quantile (tau)", fontsize=13, color=NAVY_HEX)
        strip.tick_params(labelsize=11, colors=GRAY_HEX)
        for s in strip.spines.values():
            s.set_visible(False)
        fig.tight_layout(pad=0.3)
        return self._save_fig(fig, facecolor="white")

    # -----------------------------------------------------------------
    # Code-block helper (dark panel with token-colored lines)
    # -----------------------------------------------------------------
    def _add_code_block(self, x, y, w, token_lines, font_size=12, line_height=10):
        n_lines = len(token_lines)
        total_h = n_lines * line_height + 24

        self._shadow_rect(x, y, w, total_h)
        self.set_fill_color(*NAVY)
        self.rect(x, y, w, total_h, "F")

        self.set_font("Courier", "", font_size)
        char_w = self.get_string_width("M")

        pad_x = 15
        pad_y = 12
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
    # Math-slide figure: the Theorem 3.1 composition, annotated step by
    # step, plus the QTE definition line.
    # -----------------------------------------------------------------
    def _render_math_composition(self):
        fig = plt.figure(figsize=(10, 4.4))
        fig.patch.set_alpha(0)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        eq_y = 0.80
        fs = 31
        full = r"$F^{N}_{11}(y) \;=\; F_{10}(F_{00}^{-1}(F_{01}(y)))$"
        ax.text(0.5, eq_y, full, fontsize=fs, ha="center", va="center", color=OCEAN_HEX)

        # Measure sub-expression anchors so each arrow lands on the piece it
        # names (hand-tuned coordinates drifted - arrow 3 pointed at the "=").
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        def _width(s):
            t = ax.text(0, -5, s, fontsize=fs, alpha=0)
            w = t.get_window_extent(renderer).width
            t.remove()
            return w

        w_full = _width(full)
        x_left_disp = ax.transAxes.transform((0.5, eq_y))[0] - w_full / 2.0
        inv = ax.transAxes.inverted()

        def _anchor(prefix, comp):
            xa = x_left_disp + _width(prefix) + _width(comp) / 2.0
            return float(inv.transform((xa, 0.0))[0])

        x_f10 = _anchor(r"$F^{N}_{11}(y) \;=\; $", r"$F_{10}$")
        x_f00 = _anchor(r"$F^{N}_{11}(y) \;=\; F_{10}($", r"$F_{00}^{-1}$")
        x_f01 = _anchor(r"$F^{N}_{11}(y) \;=\; F_{10}(F_{00}^{-1}($", r"$F_{01}(y)$")
        tip_y = eq_y - 0.065

        ax.annotate(
            "1. the rank of y among\ncontrol-post outcomes",
            xy=(x_f01, tip_y),
            xytext=(0.83, 0.44),
            fontsize=13.5,
            color=CORAL_HEX,
            fontweight="bold",
            ha="center",
            arrowprops=dict(arrowstyle="->", color=CORAL_HEX, lw=1.5, shrinkA=2, shrinkB=4),
        )
        ax.annotate(
            "2. the same rank's outcome\nin control-pre",
            xy=(x_f00, tip_y),
            xytext=(0.5, 0.40),
            fontsize=13.5,
            color=AMBER_HEX,
            fontweight="bold",
            ha="center",
            arrowprops=dict(arrowstyle="->", color=AMBER_HEX, lw=1.5, shrinkA=2, shrinkB=4),
        )
        ax.annotate(
            "3. how many treated-pre\ncustomers sit below it",
            xy=(x_f10, tip_y),
            xytext=(0.17, 0.44),
            fontsize=13.5,
            color=TEAL_HEX,
            fontweight="bold",
            ha="center",
            arrowprops=dict(arrowstyle="->", color=TEAL_HEX, lw=1.5, shrinkA=2, shrinkB=4),
        )
        ax.text(
            0.5,
            0.14,
            r"$\mathrm{QTE}(\tau) \;=\; F_{11}^{-1}(\tau) \,-\, F^{N,-1}_{11}(\tau)$",
            fontsize=24,
            ha="center",
            va="center",
            color=NAVY_HEX,
        )
        return self._save_fig(fig, dpi=250, transparent=True)

    # -----------------------------------------------------------------
    # Slides
    # -----------------------------------------------------------------
    def slide_01_cover(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(1)

        self.draw_split_logo(32, size=40)

        self.centered_text(80, "The effect your DiD", size=50)
        self.centered_text(112, "can't see.", size=50, color=TEAL)

        motif_path, _pw, _ph = self._render_cover_densities()
        motif_w = 222
        self.image(motif_path, (WIDTH - motif_w) / 2, 140, motif_w)

        self.centered_text(
            222,
            "Real programs move parts of the distribution.",
            size=21,
            color=NAVY,
        )
        self.centered_text(240, "The mean averages them away.", size=21, color=NAVY)

        self.set_xy(0, HEIGHT - 74)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*TEAL)
        self.cell(WIDTH, 8, "ChangesInChanges: distributional DiD. Now in diff-diff.", align="C")
        self.set_xy(0, HEIGHT - 60)
        self.set_font("Helvetica", "I", 12)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 8, "Athey & Imbens (2006), Econometrica 74(2), 431-497.", align="C")
        self.add_footer()

    def slide_02_villain(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(2)

        self._kicker(34, "The Problem")
        self.centered_text(52, "A real effect,", size=38)
        self.centered_text(80, "averaged into nothing.", size=38, color=CORAL)

        self.set_xy(35, 112)
        self.set_font("Helvetica", "", 14.5)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 70,
            8,
            "A loyalty program lifts low spenders by up to 65% and top spenders"
            " not at all. (Simulated customer-spend 2x2, so the truth is known"
            " by construction.) Run the standard playbook:",
            align="C",
        )

        card_w = 100
        card_h = 66
        gap = 14
        left_x = (WIDTH - 2 * card_w - gap) / 2
        cy = 158
        self._stat_card(
            left_x,
            cy,
            card_w,
            card_h,
            f"${DID_ATT:.2f}",
            [
                ("mean DiD, in dollars", True),
                (
                    f"p = {DID_P:.2f}   CI [-${abs(DID_CI[0]):.2f}, ${DID_CI[1]:.2f}]",
                    False,
                ),
                ("verdict: kill the program", False),
            ],
            CORAL,
        )
        self._stat_card(
            left_x + card_w + gap,
            cy,
            card_w,
            card_h,
            f"${TRUE_MEAN_EFFECT:.2f}",
            [
                ("the TRUE mean effect", True),
                ("known by construction", False),
                ("", False),
            ],
            OCEAN,
        )

        self.centered_text(246, "The effect lives where", size=17, color=NAVY)
        self.centered_text(262, "the mean isn't looking.", size=17, color=NAVY)
        self.centered_text(282, "And it gets worse.", size=15, bold=False, italic=True, color=GRAY)
        self.add_footer()

    def slide_03_twist(self):
        self.add_page()
        self.dark_gradient_background()
        self._draw_vertical_sidebar(3, dark=True)

        self._kicker(38, "The Twist")

        self.centered_text(74, "Same data. One change:", size=34, color=WHITE)
        self.centered_text(102, "log(spend).", size=34, color=SEAFOAM)

        self.centered_text(146, f"+{LOG_DID_PCT:.1f}%", size=64, color=TEAL_BRIGHT)
        self.centered_text(184, f"p = {LOG_DID_P:.3f}", size=16, bold=False, color=LIGHT_GRAY)

        self.centered_text(212, "Your scale choice just became", size=22, color=WHITE)
        self.centered_text(230, "your conclusion.", size=22, color=WHITE)

        self.set_xy(30, 252)
        self.set_font("Helvetica", "I", 12)
        self.set_text_color(*LIGHT_GRAY)
        self.multi_cell(
            WIDTH - 60,
            6.5,
            f"And neither verdict is right (truth: ${TRUE_MEAN_EFFECT:.2f}). The"
            " simulated market trend is a power-law transform of spend - additive"
            " on neither scale across groups with different customer mixes.",
            align="C",
        )

        self.centered_text(
            285, "Choosing a scale IS choosing the assumption.", size=19, color=AMBER
        )
        self.add_footer(dark=True)

    def slide_04_paper(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(4)

        self._kicker(34, "The Idea")
        self.centered_text(52, "Athey & Imbens (2006):", size=36)
        self.centered_text(80, "compare ranks, not scales.", size=36, color=TEAL)

        # VERBATIM from the abstract (Econometrica 74(2), p. 431).
        self._pull_quote(
            116,
            "The assumptions of the proposed model are invariant" " to the scaling of the outcome.",
            "- Athey & Imbens (2006), Econometrica 74(2), p. 431",
            size=16,
        )

        self.set_xy(32, 168)
        self.set_font("Helvetica", "", 15)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 64,
            8.2,
            "Their Changes-in-Changes model swaps parallel trends in means for a"
            " rank story: a treated customer's counterfactual is the outcome move"
            " of a control customer at the same rank. No scale enters the"
            " assumptions - and none enters the answer.",
            align="C",
        )

        # The receipt: dollars fit vs exp(logs fit), identical.
        chip_y = 224
        self.set_xy(0, chip_y)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 8, "The receipt (median counterfactual quantile):", align="C")

        pair_y = chip_y + 14
        chip_w = 92
        gap = 10
        left_x = (WIDTH - 2 * chip_w - gap) / 2
        for i, (label, color) in enumerate(
            (("fit in dollars", OCEAN), ("exp( fit in logs )", TEAL))
        ):
            cx = left_x + i * (chip_w + gap)
            self._shadow_rect(cx, pair_y, chip_w, 26)
            self.set_fill_color(*WHITE)
            self.set_draw_color(220, 220, 220)
            self.rect(cx, pair_y, chip_w, 26, "DF")
            self.set_xy(cx, pair_y + 3)
            self.set_font("Helvetica", "B", 17)
            self.set_text_color(*color)
            self.cell(chip_w, 10, f"${RECEIPT_ROWS[1][1]:,.4f}", align="C")
            self.set_xy(cx, pair_y + 15)
            self.set_font("Helvetica", "", 11)
            self.set_text_color(*GRAY)
            self.cell(chip_w, 6, label, align="C")

        self.set_xy(30, pair_y + 34)
        self.set_font("Helvetica", "I", 12)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 60,
            6.2,
            f"Identical across all 19 quantiles - max relative difference"
            f" {MAX_REL_DIFF_LABEL} on this run (unconditional fits; means and"
            " ATTs stay scale-specific for every estimator).",
            align="C",
        )
        self.add_footer()

    def slide_05_math(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(5)

        self._kicker(34, "The Math")
        self.centered_text(52, "One composition builds", size=36)
        self.centered_text(80, "the whole counterfactual.", size=36, color=TEAL)

        eq_path, epw, eph = self._render_math_composition()
        eq_h = self._place_image_centered(eq_path, epw, eph, 110, max_w=210)

        base_y = 110 + eq_h + 10
        self.set_xy(30, base_y)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 60,
            7.6,
            "Read inside-out: rank a treated outcome among control-post, walk"
            " that rank back to control-pre, and count the treated-pre customers"
            " below it. The result is the treated group's entire no-program"
            " outcome distribution (Theorem 3.1) - quantile treatment effects at"
            " every tau, and the ATT as its mean. Nonparametrically identified;"
            " panel or repeated cross-sections.",
            align="C",
        )
        self.add_footer()

    def slide_06_payoff(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(6)

        self._kicker(32, "The Payoff")
        self.centered_text(50, "Same program.", size=38)
        self.centered_text(78, "Now visible.", size=38, color=TEAL)

        plot_path, ppw, pph = self._render_qte_profile()
        plot_h = self._place_image_centered(plot_path, ppw, pph, 104, max_w=212)

        base_y = 104 + plot_h + 8
        self.set_xy(30, base_y)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 60,
            7.6,
            "Gains of about +$5/month across the bottom half of the spend"
            " distribution, fading to nothing distinguishable from zero above"
            f" the 70th percentile. And the ATT - ${CIC_ATT:.2f}"
            f" (p = {CIC_ATT_P:.3f}) - lands on the ${TRUE_MEAN_EFFECT:.2f}"
            " truth the mean missed.",
            align="C",
        )
        self.add_footer()

    def slide_07_bands(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(7)

        self._kicker(32, "One Claim")
        self.centered_text(50, '"The bottom half moved."', size=34)
        self.centered_text(78, "One test, nineteen quantiles.", size=34, color=TEAL)

        plot_path, ppw, pph = self._render_uniform_bands()
        plot_h = self._place_image_centered(plot_path, ppw, pph, 102, max_w=208)

        base_y = 102 + plot_h + 7
        self.set_xy(28, base_y)
        self.set_font("Helvetica", "", 13.5)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 56,
            7.2,
            "Nineteen pointwise intervals over-reject when read together. Sup-t"
            " uniform bands (fixed 95%, a qte-parity convention) exclude zero for"
            " every quantile from 0.05 through 0.50 - and none above. The"
            f" pointwise blip at tau = {BLIP_TAU:.2f} (p = {BLIP_P:.3f})"
            " evaporates under the bands: exactly the trap they exist to"
            " prevent. Above the median the bands are silent, not exonerating.",
            align="C",
        )
        self.add_footer()

    def slide_08_code(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(8)

        self._kicker(34, "The Code")
        self.centered_text(52, "Three lines to the", size=40)
        self.centered_text(80, "whole distribution.", size=40, color=TEAL)

        margin = 30
        code_y = 108
        token_lines = [
            [
                ("from", SLATE_CODE),
                (" diff_diff ", WHITE),
                ("import", SLATE_CODE),
                (" ChangesInChanges", WHITE),
            ],
            [],
            [
                ("results", WHITE),
                (" = ", SLATE_CODE),
                ("ChangesInChanges", AMBER_CODE),
                ("(", WHITE),
                ("n_bootstrap", WHITE),
                ("=", SLATE_CODE),
                ("999", AMBER_CODE),
                (", ", SLATE_CODE),
                ("seed", WHITE),
                ("=", SLATE_CODE),
                ("27", AMBER_CODE),
                (").fit(", WHITE),
            ],
            [
                ("    df, ", WHITE),
                ("outcome", WHITE),
                ("=", SLATE_CODE),
                ("'spend'", AMBER_CODE),
                (", ", SLATE_CODE),
                ("treatment", WHITE),
                ("=", SLATE_CODE),
                ("'treated'", AMBER_CODE),
                (", ", SLATE_CODE),
                ("time", WHITE),
                ("=", SLATE_CODE),
                ("'post'", AMBER_CODE),
                (",", SLATE_CODE),
            ],
            [(")", WHITE)],
            [],
            [
                ("results.att", WHITE),
                ("                # $3.05 (p = 0.004)", LIGHT_GRAY),
            ],
            [
                ("results.quantile_effects", WHITE),
                ("   # QTE at 19 quantiles", LIGHT_GRAY),
            ],
            [
                ("results.uniform_bands()", WHITE),
                ("    # sup-t joint bands", LIGHT_GRAY),
            ],
            [],
            [("# customer mix differs?  fit(..., covariates=['tenure'])", LIGHT_GRAY)],
            [("# same units both periods?  ChangesInChanges(panel=True)", LIGHT_GRAY)],
        ]
        code_h = self._add_code_block(
            margin, code_y, WIDTH - margin * 2, token_lines, font_size=12, line_height=10
        )

        self.centered_text(
            code_y + code_h + 12,
            "Same sklearn-style API as every diff-diff estimator.",
            size=14,
            bold=False,
            color=GRAY,
        )
        self.add_footer()

    def slide_09_covariates(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(9)

        self._kicker(34, "Composition")
        self.centered_text(52, "When the mix differs,", size=38)
        self.centered_text(80, "condition.", size=38, color=TEAL)

        self.set_xy(32, 110)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 64,
            7.6,
            "If the treated group's mix differs on something that also drives"
            " the trend, the unconditional comparison is confounded. Here:"
            " treated customers skew longer-tenure, tenure drives engagement"
            f" growth, and the true effect is {COV_TRUTH:.1f} points - for"
            " everyone.",
            align="C",
        )

        card_w = 104
        card_h = 70
        gap = 12
        left_x = (WIDTH - 2 * card_w - gap) / 2
        cy = 152
        self._stat_card(
            left_x,
            cy,
            card_w,
            card_h,
            f"{COV_UNC[0]:.2f}",
            [
                ("unconditional CiC", True),
                (f"CI [{COV_UNC[1]:.2f}, {COV_UNC[2]:.2f}]", False),
                ("excludes the truth:", False),
                ("confidently wrong", False),
            ],
            CORAL,
            headline_size=30,
        )
        self._stat_card(
            left_x + card_w + gap,
            cy,
            card_w,
            card_h,
            f"{COV_COND[0]:.2f}",
            [
                ("covariates=['tenure']", True),
                (f"CI [{COV_COND[1]:.2f}, {COV_COND[2]:.2f}]", False),
                (f"covers the truth ({COV_TRUTH:.1f})", False),
                ("", False),
            ],
            TEAL,
            headline_size=30,
        )

        self.set_xy(30, 244)
        self.set_font("Helvetica", "", 13.5)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 60,
            7.2,
            "The conditional fit compares within covariate values: per-cell"
            f" linear quantile regressions on a {QR_GRID_TAUS}-tau grid,"
            " following R qte's xformla conventions verbatim - with a"
            " conditional support diagnostic standing guard.",
            align="C",
        )
        self.add_footer()

    def slide_10_production(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(10)

        self._kicker(30, "Production-Ready")
        self.centered_text(46, "Built for real pipelines.", size=34)

        features = [
            (
                "Panel + repeated cross-sections",
                "panel=True enforces balanced units + unit-block bootstrap",
            ),
            (
                "Bootstrap + sup-t uniform bands",
                "seeded, reproducible; fixed-95% bands (qte parity)",
            ),
            ("Covariates", "quantile-regression conditioning via covariates=[...]"),
            (
                "Loud guardrails",
                "support, interior range + ties warnings; NaN, never a guess",
            ),
            (
                "practitioner_next_steps()",
                "distributional next-step guidance, built in",
            ),
            (
                "sklearn-style API",
                "fit() / summary() / to_dataframe(), like every estimator",
            ),
        ]
        col_w = 104
        row_h = 40
        gap_x = 12
        gap_y = 7
        start_x = (WIDTH - 2 * col_w - gap_x) / 2
        start_y = 78
        for i, (title, desc) in enumerate(features):
            cx = start_x + (i % 2) * (col_w + gap_x)
            cy = start_y + (i // 2) * (row_h + gap_y)
            self._shadow_rect(cx, cy, col_w, row_h)
            self.set_fill_color(*WHITE)
            self.set_draw_color(220, 220, 220)
            self.set_line_width(0.5)
            self.rect(cx, cy, col_w, row_h, "DF")
            self.set_fill_color(*TEAL)
            self.rect(cx, cy, 4, row_h, "F")
            self.set_xy(cx + 10, cy + 7)
            self.set_font("Helvetica", "B", 13.5)
            self.set_text_color(*NAVY)
            self.cell(col_w - 16, 7, title)
            self.set_xy(cx + 10, cy + 19)
            self.set_font("Helvetica", "", 11.5)
            self.set_text_color(*GRAY)
            self.multi_cell(col_w - 16, 5.6, desc, align="L")

        vy = start_y + 3 * (row_h + gap_y) + 8
        self.set_fill_color(*OCEAN)
        self.rect(34, vy, WIDTH - 68, 54, "F")
        self.set_xy(0, vy + 6)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*SEAFOAM)
        self.cell(WIDTH, 8, "Validated against R qte 1.3.1", align="C")
        self.set_xy(34 + 8, vy + 18)
        self.set_font("Helvetica", "", 12.5)
        self.set_text_color(*WHITE)
        self.multi_cell(
            WIDTH - 68 - 16,
            6.4,
            "Unconditional point estimates match the qte golden files to 1e-10;"
            " the type-1 quantile arithmetic is pinned bit-exactly (atol=0)."
            " Covariate branch: conventions exact given R inputs; the LP solver"
            " is proven optimal at every tau.",
            align="C",
        )
        self.add_footer()

    def slide_11_cta(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(11)

        self.draw_split_logo(52, size=44)
        self.centered_text(92, "Ask your 2x2 who moved -", size=30)
        self.centered_text(114, "not just how much on average.", size=30, color=TEAL)

        chip_w = 150
        chip_x = (WIDTH - chip_w) / 2
        chip_y = 152
        self._shadow_rect(chip_x, chip_y, chip_w, 24)
        self.set_fill_color(*NAVY)
        self.rect(chip_x, chip_y, chip_w, 24, "F")
        self.set_xy(chip_x, chip_y + 6)
        self.set_font("Courier", "B", 17)
        self.set_text_color(*AMBER_CODE)
        self.cell(chip_w, 12, "pip install diff-diff", align="C")

        self.set_xy(35, 196)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 70,
            7.8,
            "Want the full walkthrough? Tutorial 27 reproduces this example" " end to end:",
            align="C",
        )
        self.centered_text(
            216,
            "diff-diff.readthedocs.io/en/latest/tutorials/27_cic_distributional_effects.html",
            size=12.5,
            color=TEAL_DARK,
        )

        self.centered_text(244, "github.com/igerber/diff-diff", size=16, color=OCEAN)

        self.set_xy(0, 280)
        self.set_font("Helvetica", "I", 12)
        self.set_text_color(*GRAY)
        self.cell(
            WIDTH,
            6,
            "ChangesInChanges - Athey & Imbens (2006). Now in diff-diff.",
            align="C",
        )
        self.add_footer()


def main():
    pdf = CiCCarouselPDF()
    try:
        pdf.slide_01_cover()
        pdf.slide_02_villain()
        pdf.slide_03_twist()
        pdf.slide_04_paper()
        pdf.slide_05_math()
        pdf.slide_06_payoff()
        pdf.slide_07_bands()
        pdf.slide_08_code()
        pdf.slide_09_covariates()
        pdf.slide_10_production()
        pdf.slide_11_cta()

        output_path = Path(__file__).parent / "diff-diff-cic-carousel.pdf"
        pdf.output(str(output_path))
        print(f"PDF saved to: {output_path}")
    finally:
        pdf.cleanup()


if __name__ == "__main__":
    main()
