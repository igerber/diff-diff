#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for the MMM calibration interop launch.

Mirrors the architecture of ``generate_cic_carousel.py`` (magazine sidebar
with progress tick, light gradient background, split-color logo, footer
wordmark, one dark slide, soft card shadows, phone type floors: body >= 13pt,
headlines 34-50) with the "Signal" palette: deep violet primary (calibrated
posterior / diff-diff accent), coral for the uncalibrated model, amber
reserved EXCLUSIVELY for ground truth in every chart, on a lavender-to-white
gradient. The single dark slide is the HANDSHAKE CODE slide (slide 3) - the
deck is ease-first (user decision, 2026-08-19): the drama is how little code
there is, not a villain arc.

Narrative spine (ease-first; Meridian / Tutorial 30 is the running example,
PyMC-Marketing / Tutorial 29 gets the secondary beat):

1.  Cover           -- "Your geo test is already an MMM calibration input."
                       Posterior-density motif + amber truth line
2.  The setup       -- launch in 8 of 12 markets; the MMM has a calibration
                       hook waiting; connecting them by hand = three chances
                       to silently mis-calibrate
3.  Handshake (DARK)-- fit -> aggregate('total') -> to_meridian_roi_prior.
                       Three lines, no hand-derived scaling
4.  Paste and run   -- prior.to_code() writes the Meridian code; the tutorial
                       executes the generated snippet exactly as printed
5.  The payoff      -- default 3.29 [3.02, 3.57] (truth 2.5 outside the 90%
                       interval) -> calibrated 2.54 [2.47, 2.62]
6.  Why the gap     -- a national model never saw the geo contrast; the
                       demand ramp landed on the launch window (schematic)
7.  Rigor included  -- staggered waves via CS; launch-from-zero total IS
                       roi_m's estimand; loud setup checks; cross-check
8.  PyMC too        -- spend boost -> lift test row: 3.52 -> 2.21 vs truth 2.0
9.  Capabilities    -- one-line totals (4 estimators), any-estimate explicit
                       route, both MMM dialects, pooling, guardrails, zero
                       new dependencies (capabilities-over-validation: user
                       decision, 2026-08-20 - readers care what it gets
                       them, not how we test it)
10. CTA             -- pip install; tutorials 29 + 30 mentioned ONCE, here,
                       as the learn-more teaser

Claim discipline (verified against the committed executed notebooks
``docs/tutorials/30_mmm_calibration_meridian.ipynb`` and
``docs/tutorials/29_mmm_calibration_pymc.ipynb``):

- EVERY number on the deck is a committed, seed-locked tutorial value. The
  synchronization chain: library <-> notebook is pinned by the t29/t30 drift
  tests; notebook <-> deck is pinned by ``tests/test_mmm_carousel_claims.py``
  (parses this module's constants and locates each on the committed notebook
  surface). Truth provenance is CONCENTRATED, not repeated (user decision,
  2026-08-20 - the receipt band and per-slide DGP framing were cut): the
  payoff chart's truth line is labeled "(simulated)", slide 8 carries a bare
  "truth: 2.0", and the CTA line "A simulated market, so the truth is known"
  is the full statement. Never strip the remaining three - a deck that shows
  a "true ROI" with no simulation label implies real-market ground truth.
- The slide-5 payoff chart draws ONLY committed summary numbers (posterior
  means + 90% interval endpoints as an interval plot) - no invented
  distribution shapes. The COVER density motif is illustrative art (curves
  drawn from the committed mean/interval summaries with widths uniformly
  inflated for legibility, relative widths preserved), not a data claim -
  same convention as the CiC deck's cover motif.
- The slide-6 panel is labeled a schematic ON the slide; it illustrates the
  aggregation argument and plots no tutorial data.
- Slide 4 shows the generated ``to_code()`` snippet ABRIDGED and says so
  on-slide; every displayed snippet line is verbatim from the committed
  notebook output (claims-test pinned), and the "executes the generated
  snippet exactly as printed" claim is backed by the notebook's exec cell.
- "interval 6x narrower" (slide 8) derives from committed widths
  3.59 -> 0.59; the claims test recomputes the ratio and requires >= 6.
- Slide-9 capability claims are source-pinned by the claims test: the four
  named ``aggregate('total')`` adopters must each list ``"total"`` in their
  results class's ``_AGGREGATE_SUPPORTED``, and the "Zero new dependencies"
  claim is backed by an import scan of ``diff_diff/mmm.py`` (no framework
  imports at module level - the templates embed them as text only).
- Framework versions appear once, as build facts ("built against
  pymc-marketing 1.0 + google-meridian 1.8"); no forward-compat claims.
- Guardrail claims are SCOPED, never absolute (round-1 local review): the
  exporters validate what is machine-checkable (signs, double scaling,
  window/channel membership in the mask builder) while estimand, outcome
  scale, population, and window alignment stay caller-owned (REGISTRY.md
  MMM section). The deck says "the easy mistakes fail loudly" and "you own
  the design, it owns the math"; the claims test BANS the retired absolutes
  ("no silent mis-calibration", "never silently", "anything with an
  estimate + SE", "any effect + SE", "any estimate exports"). The
  round-2 resolution (user decision, 2026-08-20; DEFERRED.md decision
  record): the slide-9 card reads "Bring your own estimator" (an API
  capability, no universal), and slide 8 carries NO on-slide linearity
  qualifier - the staggered-boost compression's linear-channel scoping is
  tutorial 29's job, which the CTA points to.
- NO competitive claims ("only frequentist bridge", competitor names): not
  click-through verifiable (user decision, 2026-08-19).
- Tutorials are referenced ONCE, on the CTA slide (CiC-deck precedent).

Run with::

    python carousel/generate_mmm_carousel.py

Produces ``carousel/diff-diff-mmm-carousel.pdf``. Generation requires
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

TOTAL_SLIDES = 10

# -------------------------------------------------------------------------
# "Signal" palette
# -------------------------------------------------------------------------
VIOLET = (124, 58, 237)  # #7c3aed  primary accent (calibrated / diff-diff)
VIOLET_DARK = (109, 40, 217)  # #6d28d9
VIOLET_BRIGHT = (167, 139, 250)  # #a78bfa  accents on the dark slide
LILAC = (221, 214, 254)  # #ddd6fe  light fills / dark-slide text pop
CORAL = (225, 29, 72)  # #e11d48  the uncalibrated model + kickers
AMBER = (245, 158, 11)  # #f59e0b  GROUND TRUTH ONLY (chart grammar)
LAVENDER_TINT = (245, 243, 255)  # #f5f3ff  gradient start
SHADOW = (203, 213, 225)  # #cbd5e1  soft card shadow

# Text + structural (shared with the other decks for legibility)
NAVY = (15, 23, 42)  # #0f172a  primary text; dark-slide gradient start
GRAY = (100, 116, 139)  # #64748b
LIGHT_GRAY = (148, 163, 184)  # #94a3b8
WHITE = (255, 255, 255)
DARK_VIOLET = (46, 16, 101)  # #2e1065  dark-slide gradient end
PANEL_NAVY = (30, 41, 59)  # #1e293b  code panel on the dark slide
AMBER_CODE = (252, 211, 77)  # #fcd34d  code string/number literals
SLATE_CODE = (148, 163, 184)  # #94a3b8

VIOLET_HEX = "#7c3aed"
VIOLET_BRIGHT_HEX = "#a78bfa"
LILAC_HEX = "#ddd6fe"
CORAL_HEX = "#e11d48"
AMBER_HEX = "#f59e0b"
NAVY_HEX = "#0f172a"
GRAY_HEX = "#64748b"
LIGHT_GRAY_HEX = "#94a3b8"

# -------------------------------------------------------------------------
# Seed-locked numbers from the committed executed tutorials.
#
# Tutorial 30 (Meridian spine): docs/tutorials/30_mmm_calibration_meridian.ipynb
# (seed 30301, 12 geos x 104 weeks, launch in 8 geos, waves 30/44/58).
# Tutorial 29 (PyMC beat): docs/tutorials/29_mmm_calibration_pymc.ipynb
# (seed 2026, 15 geos x 104 weeks, boost in 9 geos).
#
# Every constant below is located on the committed notebook surface by
# tests/test_mmm_carousel_claims.py - never edit one without re-running it.
# -------------------------------------------------------------------------
MER_DEFAULT = (3.29, 3.02, 3.57)  # default prior: ROI mean, 90% interval
MER_CAL = (2.54, 2.47, 2.62)  # calibrated prior: ROI mean, 90% interval
MER_TRUTH = 2.50  # simulated true ROI (known by construction)
MER_DID = (2.49, 0.05)  # the DiD measurement the prior encodes
MER_ERR = (0.79, 0.04)  # posterior mean abs error, default -> calibrated
MER_WIDTH = (0.55, 0.15)  # 90% interval width, default -> calibrated

CS_TOTAL_LABEL = "184,281"  # CS aggregate('total') incremental sales
SPEND_LABEL = "74,100"  # total experiment spend
ROI_MEAS = (2.487, 0.048)  # experiment ROI +/- sd fed to the prior
PRIOR_MU, PRIOR_SIGMA = 0.9109, 0.0195  # lognormal prior parameters
MASK_WEEKS = 74  # search weeks inside the calibration window
N_GEOS_MER, N_LAUNCHED = 12, 8
WAVE_WEEKS = (30, 44, 58)
N_WEEKS = 104

PM_PLAIN = (3.52, 1.78, 5.36)  # without lift test: ROI mean, 90% interval
PM_CAL = (2.21, 1.91, 2.50)  # with lift test: ROI mean, 90% interval
PM_TRUTH = 2.00
PM_ERR = (1.52, 0.21)  # posterior mean abs error, plain -> calibrated
PM_WIDTH = (3.59, 0.59)  # 90% interval width, plain -> calibrated
N_GEOS_PM, N_BOOSTED = 15, 9
# Slide-8 call arguments (tutorial 29's lift row): x = national weekly
# baseline spend (SEARCH_BASE x N_GEOS), delta_x = total weekly boost
# (BOOST x boosted geos), scale = the boosted-geo count. The claims test
# re-derives all three from the notebook's DGP constants.
PM_X, PM_DELTA_X, PM_SCALE = 1500, 450, 9

# Lines shown on slide 4, VERBATIM from the committed to_code() output cell
# (the claims test locates each in the notebook output). The mask-building
# lines are elided on-slide; the slide says "abridged".
SNIPPET_LINES = [
    "mu = [0.9108575376065811, 0.2]",
    "sigma = [0.019461139350044197, 0.9]",
    'roi_prior = tfp.distributions.LogNormal(mu, sigma, name="roi_m")',
    "prior = prior_distribution.PriorDistribution(roi_m=roi_prior)",
    "model_spec = spec.ModelSpec(",
    "    prior=prior,",
    '    media_prior_type="roi",',
    "    roi_calibration_period=roi_calibration_period,",
    ")",
]


class MMMCarouselPDF(FPDF):
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
    # Magazine vertical sidebar -- VIOLET bar, CORAL progress tick.
    # -----------------------------------------------------------------
    def _draw_vertical_sidebar(self, slide_number, total=TOTAL_SLIDES, dark=False):
        bar_x = 14
        bar_y_top = 45
        bar_y_bottom = 275
        self.set_draw_color(*(VIOLET_BRIGHT if dark else VIOLET))
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
        """Lavender #f5f3ff fading to white."""
        steps = 50
        r0, g0, b0 = LAVENDER_TINT
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
        """Near-black navy #0f172a fading to deep violet #2e1065."""
        steps = 50
        r0, g0, b0 = NAVY
        r1, g1, b1 = DARK_VIOLET
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
        self.set_text_color(*(VIOLET_BRIGHT if dark else VIOLET))
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

    def draw_split_logo(self, y, size=18):
        """Split-color diff-diff logo with VIOLET middle dash."""
        self.set_xy(0, y)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="R")
        self.set_text_color(*VIOLET)
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

    def _chip_card(self, x, y, w, h, title, desc, accent):
        """Small feature card: accent left bar + title + description."""
        self._shadow_rect(x, y, w, h)
        self.set_fill_color(*WHITE)
        self.set_draw_color(220, 220, 220)
        self.set_line_width(0.5)
        self.rect(x, y, w, h, "DF")
        self.set_fill_color(*accent)
        self.rect(x, y, 4, h, "F")
        self.set_xy(x + 10, y + 7)
        self.set_font("Helvetica", "B", 13.5)
        self.set_text_color(*NAVY)
        self.cell(w - 16, 7, title)
        self.set_xy(x + 10, y + 19)
        self.set_font("Helvetica", "", 11.5)
        self.set_text_color(*GRAY)
        self.multi_cell(w - 16, 5.6, desc, align="L")

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
    # Cover motif -- the two ROI posteriors + the amber truth line.
    # Illustrative art from the COMMITTED summaries (means + 90% interval
    # half-widths as normal curves, widths uniformly inflated 2.5x for
    # legibility; relative widths preserved) - not a data claim.
    # -----------------------------------------------------------------
    def _render_cover_posteriors(self):
        z90 = 1.645
        sd_def = (MER_DEFAULT[2] - MER_DEFAULT[1]) / (2 * z90) * 2.5
        sd_cal = (MER_CAL[2] - MER_CAL[1]) / (2 * z90) * 2.5
        grid = np.linspace(1.7, 4.3, 600)

        def _norm(mu, sd):
            return np.exp(-0.5 * ((grid - mu) / sd) ** 2) / sd

        d_def = _norm(MER_DEFAULT[0], sd_def)
        d_cal = _norm(MER_CAL[0], sd_cal)
        peak = d_cal.max()
        d_def, d_cal = d_def / peak, d_cal / peak

        fig, ax = plt.subplots(figsize=(10, 3.1))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        ax.axvline(MER_TRUTH, color=AMBER_HEX, linewidth=3.0, linestyle=(0, (5, 2)))
        ax.text(
            MER_TRUTH - 0.05,
            1.06,
            "the true ROI (simulated)",
            fontsize=13.5,
            color=AMBER_HEX,
            fontweight="bold",
            ha="right",
        )
        ax.text(
            grid[-1],
            -0.06,
            "illustrative",
            fontsize=11,
            color=LIGHT_GRAY_HEX,
            style="italic",
            ha="right",
            va="top",
        )

        ax.plot(grid, d_def, color=CORAL_HEX, linewidth=3.2, alpha=0.75)
        ax.fill_between(grid, 0, d_def, color=CORAL_HEX, alpha=0.10, linewidth=0)
        ax.plot(grid, d_cal, color=VIOLET_HEX, linewidth=3.4)
        ax.fill_between(grid, 0, d_cal, color=VIOLET_HEX, alpha=0.12, linewidth=0)

        ax.text(
            MER_DEFAULT[0] + 0.16,
            d_def.max() * 1.28,
            "MMM alone",
            fontsize=13.5,
            color=CORAL_HEX,
            fontweight="bold",
        )
        ax.text(
            MER_CAL[0] + 0.09,
            0.86,
            "with your experiment",
            fontsize=13.5,
            color=VIOLET_HEX,
            fontweight="bold",
        )

        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(0, 1.22)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=0.2)
        return self._save_fig(fig, transparent=True)

    # -----------------------------------------------------------------
    # Payoff figure (slide 5) -- committed means + 90% intervals ONLY,
    # drawn as an interval plot. No invented distribution shapes.
    # -----------------------------------------------------------------
    def _render_payoff_intervals(self):
        fig, ax = plt.subplots(figsize=(10, 4.1))

        rows = [
            (1.0, MER_DEFAULT, CORAL_HEX, "default prior"),
            (0.0, MER_CAL, VIOLET_HEX, "with the experiment prior"),
        ]
        ax.axvline(
            MER_TRUTH,
            color=AMBER_HEX,
            linewidth=3.0,
            linestyle=(0, (5, 2)),
            zorder=1,
            label=f"true ROI = {MER_TRUTH:.1f} (simulated)",
        )
        for yv, (mean, lo, hi), color, label in rows:
            ax.plot([lo, hi], [yv, yv], color=color, linewidth=7, solid_capstyle="butt", zorder=2)
            for xend in (lo, hi):
                ax.plot([xend, xend], [yv - 0.06, yv + 0.06], color=color, linewidth=2.4)
            ax.plot([mean], [yv], "o", color=color, markersize=13, zorder=3)
            ax.annotate(
                f"{mean:.2f}",
                xy=(mean, yv),
                xytext=(mean - 0.02, yv + 0.16),
                fontsize=16,
                fontweight="bold",
                color=color,
                ha="center",
            )
            ax.text(
                lo - 0.045,
                yv,
                label,
                fontsize=13.5,
                color=color,
                fontweight="bold",
                ha="right",
                va="center",
            )
            ax.text(
                hi + 0.045,
                yv,
                f"[{lo:.2f}, {hi:.2f}]",
                fontsize=12.5,
                color=GRAY_HEX,
                va="center",
            )

        ax.text(
            MER_DEFAULT[0],
            0.68,
            "the truth isn't even in the 90% interval",
            fontsize=13,
            color=CORAL_HEX,
            fontweight="bold",
            ha="center",
        )

        ax.set_xlim(1.28, 4.05)
        ax.set_ylim(-0.55, 1.55)
        ax.set_yticks([])
        ax.set_xlabel(
            "search-channel ROI posterior (mean + 90% interval)", fontsize=13, color=NAVY_HEX
        )
        ax.tick_params(labelsize=12, colors=GRAY_HEX)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(LIGHT_GRAY_HEX)
        ax.legend(loc="upper left", fontsize=12, frameon=False)
        fig.tight_layout(pad=0.3)
        return self._save_fig(fig, facecolor="white")

    # -----------------------------------------------------------------
    # Aggregation schematic (slide 6) -- labeled a schematic on-slide;
    # illustrates the argument, plots no tutorial data.
    # -----------------------------------------------------------------
    def _render_aggregation_schematic(self):
        rng = np.random.default_rng(30301)
        t = np.arange(N_WEEKS)
        ramp = np.clip((t - 26) / 52.0, 0, None)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.8))
        for ax in (ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

        # Left: the geo contrast the experiment sees.
        waves = [30, 30, 30, 44, 44, 44, 58, 58]
        for i, w in enumerate(waves):
            base = i * 1.15
            y = base + 0.55 * (t >= w)
            ax1.plot(t, y, color=VIOLET_HEX, linewidth=1.9, alpha=0.85)
            ax1.axvline(w, color=VIOLET_HEX, alpha=0.0)
        for j in range(4):
            base = -(j + 1) * 1.15
            ax1.plot(t, np.full_like(t, base, dtype=float), color=LIGHT_GRAY_HEX, linewidth=1.9)
        ax1.text(
            2, 9.6, "8 launched markets (3 waves)", fontsize=13, color=VIOLET_HEX, fontweight="bold"
        )
        ax1.text(2, -5.6, "4 holdouts", fontsize=13, color=GRAY_HEX, fontweight="bold")
        ax1.set_title("what the experiment sees", fontsize=14, color=NAVY_HEX, fontweight="bold")
        ax1.set_ylim(-6.6, 10.8)

        # Right: the single national series the MMM sees.
        launched_frac = np.mean([(t >= w) for w in waves], axis=0)
        national = 1.0 + 2.4 * ramp + 0.55 * launched_frac + rng.normal(0, 0.16, N_WEEKS)
        ax2.plot(t, national, color=CORAL_HEX, linewidth=2.6)
        ax2.axvspan(30, N_WEEKS - 1, color=LILAC_HEX, alpha=0.45, linewidth=0)
        ax2.text(35, 0.62, "launch window", fontsize=12.5, color=VIOLET_HEX)
        ax2.annotate(
            "demand ramp + launch,\none indistinguishable line",
            xy=(78, float(national[78])),
            xytext=(18, 4.25),
            fontsize=13,
            color=CORAL_HEX,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=CORAL_HEX, lw=1.5),
        )
        ax2.set_title("what a national MMM sees", fontsize=14, color=NAVY_HEX, fontweight="bold")
        ax2.set_ylim(0.3, 5.0)

        fig.text(
            0.99, 0.01, "schematic", fontsize=11, color=LIGHT_GRAY_HEX, ha="right", style="italic"
        )
        fig.tight_layout(pad=0.6)
        return self._save_fig(fig, facecolor="white")

    # -----------------------------------------------------------------
    # Code-block helper (dark panel with token-colored lines)
    # -----------------------------------------------------------------
    def _add_code_block(self, x, y, w, token_lines, font_size=12, line_height=10, fill=NAVY):
        n_lines = len(token_lines)
        total_h = n_lines * line_height + 24

        self._shadow_rect(x, y, w, total_h)
        self.set_fill_color(*fill)
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
    # Slides
    # -----------------------------------------------------------------
    def slide_01_cover(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(1)

        self.draw_split_logo(30, size=38)

        self.centered_text(72, "Your geo test is already", size=45)
        self.centered_text(102, "an MMM calibration input.", size=45, color=VIOLET)

        motif_path, _pw, _ph = self._render_cover_posteriors()
        motif_w = 216
        self.image(motif_path, (WIDTH - motif_w) / 2, 134, motif_w)

        self.centered_text(
            216, "Turn a DiD experiment into your MMM's prior -", size=20, color=NAVY
        )
        self.centered_text(233, "in three lines of code.", size=20, color=NAVY)

        self.set_xy(0, HEIGHT - 70)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*VIOLET)
        self.cell(
            WIDTH,
            8,
            "Speaks Google Meridian + PyMC-Marketing natively.",
            align="C",
        )
        self.add_footer()

    def slide_02_setup(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(2)

        self._kicker(34, "The Setup")
        self.centered_text(52, "You ran the experiment.", size=38)
        self.centered_text(80, "Your MMM is still guessing.", size=38, color=CORAL)

        self.set_xy(33, 112)
        self.set_font("Helvetica", "", 14.5)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 66,
            8,
            f"A search channel launched in {N_LAUNCHED} of {N_GEOS_MER} markets, in three"
            f" staggered waves. The {N_GEOS_MER - N_LAUNCHED} holdouts never got it. Your MMM"
            " has a calibration hook built for exactly this experiment - Meridian takes"
            " ROI priors, PyMC-Marketing takes lift tests. Connecting them by hand means"
            " deriving, yourself:",
            align="C",
        )

        card_w = 66
        card_h = 42
        gap = 8
        left_x = (WIDTH - 3 * card_w - 2 * gap) / 2
        cy = 164
        for i, (title, desc) in enumerate(
            (
                ("a total", "incremental outcome,\nin outcome units"),
                ("a lognormal", "mu / sigma from the\nestimate and its SE"),
                ("a mask window", "which weeks the\nprior may speak for"),
            )
        ):
            cx = left_x + i * (card_w + gap)
            self._shadow_rect(cx, cy, card_w, card_h)
            self.set_fill_color(*WHITE)
            self.set_draw_color(220, 220, 220)
            self.set_line_width(0.5)
            self.rect(cx, cy, card_w, card_h, "DF")
            self.set_fill_color(*CORAL)
            self.rect(cx, cy, card_w, 3.0, "F")
            self.set_xy(cx, cy + 8)
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(*NAVY)
            self.cell(card_w, 8, title, align="C")
            self.set_xy(cx + 4, cy + 20)
            self.set_font("Helvetica", "", 11.5)
            self.set_text_color(*GRAY)
            self.multi_cell(card_w - 8, 5.6, desc, align="C")

        self.centered_text(226, "Get any one wrong and the model is", size=17, color=NAVY)
        self.centered_text(242, "silently mis-calibrated.", size=17, color=NAVY)
        self.centered_text(
            268, "Here's the whole bridge.", size=15, bold=False, italic=True, color=GRAY
        )
        self.add_footer()

    def slide_03_handshake(self):
        self.add_page()
        self.dark_gradient_background()
        self._draw_vertical_sidebar(3, dark=True)

        self._kicker(36, "The Handshake")
        self.centered_text(56, "Three lines.", size=52, color=WHITE)

        margin = 26
        code_y = 100
        token_lines = [
            [
                ("res", WHITE),
                (" = ", SLATE_CODE),
                ("CallawaySantAnna", AMBER_CODE),
                ("().fit(panel, ", WHITE),
                ("outcome", WHITE),
                ("=", SLATE_CODE),
                ("'sales'", AMBER_CODE),
                (", ", SLATE_CODE),
                ("unit", WHITE),
                ("=", SLATE_CODE),
                ("'geo'", AMBER_CODE),
                (",", SLATE_CODE),
            ],
            [
                ("                              ", WHITE),
                ("time", WHITE),
                ("=", SLATE_CODE),
                ("'week'", AMBER_CODE),
                (", ", SLATE_CODE),
                ("first_treat", WHITE),
                ("=", SLATE_CODE),
                ("'first_treat'", AMBER_CODE),
                (")", WHITE),
            ],
            [],
            [
                ("total", WHITE),
                (" = ", SLATE_CODE),
                ("res.aggregate(", WHITE),
                ("'total'", AMBER_CODE),
                (")", WHITE),
                ("      # incremental sales: 184,281", LIGHT_GRAY),
            ],
            [],
            [
                ("prior", WHITE),
                (" = ", SLATE_CODE),
                ("to_meridian_roi_prior", AMBER_CODE),
                ("(", WHITE),
                ("aggregation_result", WHITE),
                ("=", SLATE_CODE),
                ("total", WHITE),
                (",", SLATE_CODE),
            ],
            [
                ("                              ", WHITE),
                ("spend", WHITE),
                ("=", SLATE_CODE),
                ("total_spend", WHITE),
                (")", WHITE),
                ("   # 74,100", LIGHT_GRAY),
            ],
            [],
            [
                ("# experiment ROI: 2.487 +/- 0.048", LIGHT_GRAY),
            ],
            [
                ("# -> LogNormal(mu=0.9109, sigma=0.0195), ready for Meridian", LIGHT_GRAY),
            ],
        ]
        code_h = self._add_code_block(
            margin,
            code_y,
            WIDTH - margin * 2,
            token_lines,
            font_size=11.5,
            line_height=10,
            fill=PANEL_NAVY,
        )

        y = code_y + code_h + 14
        for line, color in (
            ("Staggered waves?  Handled by the estimator.", LILAC),
            ("Outcome units?  The total already speaks them.", LILAC),
            ("Lognormal math?  Done, with the SE propagated.", LILAC),
        ):
            self.centered_text(y, line, size=15, bold=False, color=color)
            y += 13

        self.centered_text(y + 6, "No hand-derived scaling anywhere.", size=18, color=VIOLET_BRIGHT)
        self.add_footer(dark=True)

    def slide_04_to_code(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(4)

        self._kicker(32, "Paste and Run")
        self.centered_text(50, "It writes the", size=40)
        self.centered_text(78, "Meridian code for you.", size=40, color=VIOLET)

        # The displayed call is the tutorial's ACTUAL invocation - to_code()
        # deliberately fails closed without channel + time scope, so a bare
        # prior.to_code() would raise (CI review R1).
        self.set_xy(0, 98)
        self.set_font("Courier", "B", 12.5)
        self.set_text_color(*NAVY)
        self.cell(
            WIDTH, 7, 'prior.to_code(channel="search", media_channels=["search", "tv"],', align="C"
        )
        self.set_xy(0, 106)
        self.cell(WIDTH, 7, "roi_calibration_period=mask)", align="C")
        self.set_xy(0, 116)
        self.set_font("Helvetica", "I", 12)
        self.set_text_color(*GRAY)
        self.cell(
            WIDTH,
            8,
            "generated output (abridged - the full snippet builds the mask too):",
            align="C",
        )

        margin = 32
        code_y = 130
        token_lines = []
        for line in SNIPPET_LINES:
            stripped = line.strip()
            if stripped.startswith(("mu =", "sigma =")):
                name, rest = line.split(" = ", 1)
                token_lines.append([(name, WHITE), (" = ", SLATE_CODE), (rest, AMBER_CODE)])
            else:
                token_lines.append([(line, WHITE)])
        code_h = self._add_code_block(
            margin, code_y, WIDTH - margin * 2, token_lines, font_size=11.5, line_height=9.5
        )

        y = code_y + code_h + 12
        for text in (
            "Your experiment's prior on 'search' - the other channels keep Meridian's own defaults.",
            f"For 'search', the calibration mask covers the {MASK_WEEKS}-week experiment window"
            " and nothing else.",
            "Paste it into your Meridian pipeline - it runs exactly as generated.",
        ):
            self.set_xy(30, y)
            self.set_font("Helvetica", "", 13.5)
            self.set_text_color(*NAVY)
            self.multi_cell(WIDTH - 60, 7, text, align="C")
            y = self.get_y() + 3
        self.add_footer()

    def slide_05_payoff(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(5)

        self._kicker(32, "The Payoff")
        self.centered_text(50, "One prior. The posterior", size=38)
        self.centered_text(78, "snaps to the truth.", size=38, color=VIOLET)

        plot_path, ppw, pph = self._render_payoff_intervals()
        plot_h = self._place_image_centered(plot_path, ppw, pph, 104, max_w=214)

        base_y = 104 + plot_h + 8
        self.set_xy(28, base_y)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 56,
            7.6,
            f"With Meridian's default prior, the model settles on {MER_DEFAULT[0]:.2f} -"
            f" confidently, and wrong. Feed it the experiment and the posterior lands on"
            f" {MER_CAL[0]:.2f} against a true ROI of {MER_TRUTH:.1f}. Posterior error"
            f" {MER_ERR[0]:.2f} -> {MER_ERR[1]:.2f}; interval width {MER_WIDTH[0]:.2f} ->"
            f" {MER_WIDTH[1]:.2f}. The DiD measured {MER_DID[0]:.2f} +/- {MER_DID[1]:.2f}.",
            align="C",
        )
        self.add_footer()

    def slide_06_why(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(6)

        self._kicker(34, "Why It Was Wrong")
        self.centered_text(52, "The national model", size=38)
        self.centered_text(80, "never saw the contrast.", size=38, color=CORAL)

        plot_path, ppw, pph = self._render_aggregation_schematic()
        plot_h = self._place_image_centered(plot_path, ppw, pph, 108, max_w=216)

        base_y = 108 + plot_h + 10
        self.set_xy(30, base_y)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 60,
            7.6,
            "Aggregation destroys the holdout comparison, and a demand ramp lands right"
            " on the launch window - so the model credits the launch. The experiment is"
            " the one thing in the data that moved spend independent of demand. That is"
            " exactly why both MMMs ship calibration hooks.",
            align="C",
        )
        self.add_footer()

    def slide_07_rigor(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(7)

        self._kicker(34, "Rigor Included")
        self.centered_text(52, "The estimand matches,", size=38)
        self.centered_text(80, "so you don't have to.", size=38, color=VIOLET)

        cards = [
            (
                "Staggered launches",
                "Callaway-Sant'Anna cohorts - not a naive pre/post average across waves",
            ),
            (
                "Launch-from-zero = roi_m",
                "the experiment total IS Meridian's full-spend-vs-none ROI estimand",
            ),
            (
                "It checks your setup",
                "malformed windows, wrong signs, double-scaled totals - the easy mistakes fail loudly",
            ),
            (
                "Cross-checked estimate",
                "swap in ImputationDiD and the total agrees - a one-line robustness check",
            ),
        ]
        col_w = 104
        row_h = 44
        gap_x = 12
        gap_y = 12
        start_x = (WIDTH - 2 * col_w - gap_x) / 2
        start_y = 122
        for i, (title, desc) in enumerate(cards):
            cx = start_x + (i % 2) * (col_w + gap_x)
            cy = start_y + (i // 2) * (row_h + gap_y)
            self._chip_card(cx, cy, col_w, row_h, title, desc, VIOLET)

        self.centered_text(
            start_y + 2 * (row_h + gap_y) + 16,
            "You think about the experiment. It thinks about the export.",
            size=17,
            color=NAVY,
        )
        self.add_footer()

    def slide_08_pymc(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(8)

        self._kicker(34, "Also Speaks PyMC")
        self.centered_text(52, "Spend boost instead?", size=38)
        self.centered_text(80, "That's a lift test.", size=38, color=VIOLET)

        self.set_xy(32, 108)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*GRAY)
        self.multi_cell(
            WIDTH - 64,
            7.6,
            f"The other classic geo experiment - a budget boost in {N_BOOSTED} of"
            f" {N_GEOS_PM} markets, with a spend history that chased demand - exports"
            " as a PyMC-Marketing lift-test row:",
            align="C",
        )

        margin = 42
        code_y = 134
        token_lines = [
            [
                ("df", WHITE),
                (" = ", SLATE_CODE),
                ("to_pymc_marketing_lift_test", AMBER_CODE),
                ("(", WHITE),
                ("channel", WHITE),
                ("=", SLATE_CODE),
                ("'search'", AMBER_CODE),
                (", ", SLATE_CODE),
                ("x", WHITE),
                ("=", SLATE_CODE),
                (str(PM_X), AMBER_CODE),
                (",", SLATE_CODE),
            ],
            [
                ("        ", WHITE),
                ("delta_x", WHITE),
                ("=", SLATE_CODE),
                (str(PM_DELTA_X), AMBER_CODE),
                (", ", SLATE_CODE),
                ("aggregation_result", WHITE),
                ("=", SLATE_CODE),
                ("simple", WHITE),
                (", ", SLATE_CODE),
                ("scale", WHITE),
                ("=", SLATE_CODE),
                (str(PM_SCALE), AMBER_CODE),
                (")", WHITE),
            ],
            [
                ("mmm.add_lift_test_measurements(df)", WHITE),
                ("   # scale = 9 boosted geos", LIGHT_GRAY),
            ],
        ]
        self._add_code_block(
            margin, code_y, WIDTH - margin * 2, token_lines, font_size=11, line_height=9.5
        )

        card_w = 104
        card_h = 66
        gap = 12
        left_x = (WIDTH - 2 * card_w - gap) / 2
        cy = 196
        self._stat_card(
            left_x,
            cy,
            card_w,
            card_h,
            f"{PM_PLAIN[0]:.2f}",
            [
                ("without the lift test", True),
                (f"90% interval [{PM_PLAIN[1]:.2f}, {PM_PLAIN[2]:.2f}]", False),
                ("wide AND centered high", False),
            ],
            CORAL,
            headline_size=32,
        )
        self._stat_card(
            left_x + card_w + gap,
            cy,
            card_w,
            card_h,
            f"{PM_CAL[0]:.2f}",
            [
                ("with it", True),
                (f"90% interval [{PM_CAL[1]:.2f}, {PM_CAL[2]:.2f}]", False),
                (f"truth: {PM_TRUTH:.1f}", False),
            ],
            VIOLET,
            headline_size=32,
        )

        self.set_xy(30, 272)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 60,
            7.4,
            f"Posterior error {PM_ERR[0]:.2f} -> {PM_ERR[1]:.2f}; the 90% interval is"
            f" 6x narrower ({PM_WIDTH[0]:.2f} -> {PM_WIDTH[1]:.2f}). One exported"
            " DataFrame did that.",
            align="C",
        )
        self.add_footer()

    def slide_09_production(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(9)

        self._kicker(30, "What You Get")
        self.centered_text(46, "Built for how you measure.", size=34)

        features = [
            (
                "One-line totals",
                "aggregate('total') on Callaway-Sant'Anna, EfficientDiD, ImputationDiD, TwoStageDiD",
            ),
            (
                "Bring your own estimator",
                "synthetic-control geo tests, on/off pulses, dose designs - the explicit route takes your effect + SE",
            ),
            (
                "Both MMM dialects",
                "Meridian ROI priors + calibration masks; PyMC-Marketing lift tests, geo dims included",
            ),
            (
                "Multiple experiments",
                "spend-weighted pooling into one prior, with optional SE widening",
            ),
            (
                "Loud guardrails",
                "wrong-sign and double-scaling mistakes raise - you own the design, it owns the math",
            ),
            (
                "sklearn-style API",
                "fit() / summary() / to_dataframe(), like every diff-diff estimator",
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
            self._chip_card(cx, cy, col_w, row_h, title, desc, VIOLET)

        vy = start_y + 3 * (row_h + gap_y) + 8
        self.set_fill_color(*DARK_VIOLET)
        self.rect(34, vy, WIDTH - 68, 54, "F")
        self.set_xy(0, vy + 6)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*LILAC)
        self.cell(WIDTH, 8, "Zero new dependencies", align="C")
        self.set_xy(34 + 8, vy + 18)
        self.set_font("Helvetica", "", 12.5)
        self.set_text_color(*WHITE)
        self.multi_cell(
            WIDTH - 68 - 16,
            6.4,
            "The exporters emit plain DataFrames, prior parameters, and generated code"
            " - your DiD analysis never has to share an environment with Meridian or"
            " PyMC. Built against pymc-marketing 1.0 + google-meridian 1.8.",
            align="C",
        )
        self.add_footer()

    def slide_10_cta(self):
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(10)

        self.draw_split_logo(52, size=44)
        self.centered_text(92, "Stop choosing between", size=30)
        self.centered_text(114, "the experiment and the model.", size=30, color=VIOLET)

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

        self.set_xy(35, 194)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*NAVY)
        self.multi_cell(
            WIDTH - 70,
            7.8,
            "A simulated market, so the truth is known - tutorials 29 and 30 reproduce"
            " every number here end to end:",
            align="C",
        )
        self.centered_text(
            216,
            "diff-diff.readthedocs.io/en/latest/tutorials/30_mmm_calibration_meridian.html",
            size=12,
            color=VIOLET_DARK,
        )
        self.centered_text(
            228,
            "diff-diff.readthedocs.io/en/latest/tutorials/29_mmm_calibration_pymc.html",
            size=12,
            color=VIOLET_DARK,
        )

        self.centered_text(252, "github.com/igerber/diff-diff", size=16, color=NAVY)

        self.set_xy(0, 280)
        self.set_font("Helvetica", "I", 12)
        self.set_text_color(*GRAY)
        self.cell(
            WIDTH,
            6,
            "DiD experiments -> MMM calibration. Now in diff-diff.",
            align="C",
        )
        self.add_footer()


def main():
    pdf = MMMCarouselPDF()
    try:
        pdf.slide_01_cover()
        pdf.slide_02_setup()
        pdf.slide_03_handshake()
        pdf.slide_04_to_code()
        pdf.slide_05_payoff()
        pdf.slide_06_why()
        pdf.slide_07_rigor()
        pdf.slide_08_pymc()
        pdf.slide_09_production()
        pdf.slide_10_cta()

        output_path = Path(__file__).parent / "diff-diff-mmm-carousel.pdf"
        pdf.output(str(output_path))
        print(f"PDF saved to: {output_path}")
    finally:
        pdf.cleanup()


if __name__ == "__main__":
    main()
