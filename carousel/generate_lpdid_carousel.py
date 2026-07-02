#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for LPDiD (Dube, Girardi, Jorda & Taylor 2025) launch.

Mirrors the architecture of ``generate_spillover_carousel.py`` (magazine
sidebar with progress tick, light gradient background, split-color logo,
footer wordmark) with the "Horizon" palette: violet primary / sky-cyan
clean-control accent / amber impact accent on a lavender -> white gradient.
Adds editorial flair: amber section kickers, serif pull-quotes lifted
verbatim from the paper, soft card shadows, a cover impulse-response motif,
and a single dark "twist" slide at the narrative pivot. Type sizes are
floored for phone-width LinkedIn rendering (body >= 13pt, headlines 38-52).

Narrative spine (the paper's own novelty argument, JAE 2025 pp. 741-742):

1.  Cover        -- macro built LPs to read dynamic effects of shocks;
                    your event study is one
2.  Problem      -- negative weighting: previously treated units are
                    implicitly used as controls (pull quote + schematic)
3.  Twist (DARK) -- naive LP inherits the same bias (Eqs. 6-7); the
                    comparisons are the fix, not the estimator
4.  Insight      -- the "underappreciated feature": LP makes restricting
                    comparisons easy; LP + clean controls = LP-DiD
5.  Mechanism    -- "unclean" observations leave the control group
                    (clean-control panel schematic)
6.  The math     -- non-negative weights (Eqs. 9-10) + "subsumes":
                    recent estimators as specific subcases (Sec. 3.7)
7.  Output       -- native event study; pre/post estimated symmetrically
                    (avoids Roth 2024 interpretation difficulties)
8.  Code         -- sklearn-like fit(); reweight=True == Callaway-Sant'Anna
9.  Production   -- 2x3 feature grid (incl. non-absorbing + survey designs)
10. Validated    -- authors' tooling, fixest, svyglm, tested equivalences
11. CTA          -- pip install + GitHub

Claim discipline (verified against docs/methodology/papers/dube-2025-review.md
and the JAE 2025 PDF, pp. 741-742):
- All pull quotes are VERBATIM from the paper (abstract + Section 1-2 intro);
  "..." marks elisions. No invented or paraphrased-as-quoted text.
- The "zoo" named on the slide-6 equivalence map is EXACTLY the four
  estimators the paper proves LP-DiD nests: Callaway-Sant'Anna, Cengiz-style
  stacking, classic 2x2 DiD, and (single-cohort) BJS. Sun-Abraham has no
  documented LP-DiD equivalence and is deliberately absent.
- CS and Cengiz-stacked equivalences are EXACT (paper Section 3.7).
- The stacking equivalence target is the Cengiz et al. (2019) LITERATURE design.
  Never call it "Stacked DiD": this library's `StackedDiD` estimator implements
  Wing, Freedman & Hollingsworth (2024) with Q-weights, a different (corrected)
  scheme that LP-DiD does NOT nest (see the CE-4 note in tests/test_lpdid.py).
- BJS equality holds only for PMD k=t-1 single-cohort (fn. 10-11); phrased
  as "BJS-style / exact single-cohort" everywhere. Never a bare "==".
- Naive-LP-bias twist cites the paper's own statement (Sec. 2 intro, Eqs. 6-7):
  a naive LP implementation incurs negative weighting analogous to TWFE.
- The Roth (2024) symmetric-pretrends point is the authors' claim (Sec. 1),
  attributed to the paper, not asserted independently.
- NO speed claims: the paper's Table 2 speedups are Stata-implementation
  artifacts; diff-diff's own vectorized CS is faster than LPDiD (verified
  empirically 2026-07-01), so speed is omitted from the narrative.

Run with::

    python carousel/generate_lpdid_carousel.py

Produces ``carousel/diff-diff-lpdid-carousel.pdf``. Generation requires
``fpdf2``, ``Pillow``, and ``matplotlib`` (carousel-only dependencies, not
part of the library's install extras; matplotlib ships in dev/docs extras).
"""

import os
import re
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from fpdf import FPDF  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

# Computer Modern for math
plt.rcParams["mathtext.fontset"] = "cm"

# Page dimensions (4:5 portrait -- same as spillover/HAD)
WIDTH = 270  # mm
HEIGHT = 337.5  # mm

# Version label for the footer wordmark -- derived from pyproject.toml so the
# carousel can never drift from the release it advertises. Parsed with a
# regex (not tomllib, which is Python 3.11+; the project supports >=3.9).
_PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"
_m = re.search(r'^version\s*=\s*"([^"]+)"', _PYPROJECT.read_text(encoding="utf-8"), re.MULTILINE)
if _m is None:
    raise RuntimeError(f"could not parse project version from {_PYPROJECT}")
VERSION_LABEL = "v" + _m.group(1)

TOTAL_SLIDES = 11

# -------------------------------------------------------------------------
# "Horizon" palette
# -------------------------------------------------------------------------
# Primary palette (RGB)
VIOLET = (109, 40, 217)  # #6d28d9  primary accent
VIOLET_DARK = (91, 33, 182)  # #5b21b6
VIOLET_LIGHT = (196, 181, 253)  # #c4b5fd  treated-cell tint
VIOLET_BRIGHT = (167, 139, 250)  # #a78bfa  accents on the dark slide
CYAN = (8, 145, 178)  # #0891b2  clean-control / forward accent
CYAN_LIGHT = (165, 243, 252)  # #a5f3fc
AMBER = (245, 158, 11)  # #f59e0b  impact / event-time pop
LAVENDER = (245, 243, 255)  # #f5f3ff  gradient start
SHADOW = (203, 213, 225)  # #cbd5e1  soft card shadow

# Text + structural (shared with the spillover deck for legibility)
NAVY = (15, 23, 42)  # #0f172a  primary text
GRAY = (100, 116, 139)  # #64748b  secondary text
LIGHT_GRAY = (148, 163, 184)  # #94a3b8  fine print
WHITE = (255, 255, 255)
DARK_SLATE = (30, 41, 59)  # #1e293b  code block bg / dark-slide gradient end
AMBER_CODE = (252, 211, 77)  # #fcd34d  code string literals
SLATE_CODE = (148, 163, 184)  # #94a3b8  code keyword tone

# Hex equivalents for matplotlib
VIOLET_HEX = "#6d28d9"
VIOLET_DARK_HEX = "#5b21b6"
VIOLET_LIGHT_HEX = "#c4b5fd"
CYAN_HEX = "#0891b2"
AMBER_HEX = "#f59e0b"
NAVY_HEX = "#0f172a"
GRAY_HEX = "#64748b"
LIGHT_GRAY_HEX = "#94a3b8"


class LPDiDCarouselPDF(FPDF):
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
    # Magazine vertical sidebar -- drawn on every slide. The tick
    # advances from near-top (slide 1) to near-bottom (slide 11). Bar is
    # VIOLET (VIOLET_BRIGHT on the dark slide); tick is AMBER.
    # -----------------------------------------------------------------

    def _draw_vertical_sidebar(self, slide_number, total=TOTAL_SLIDES, dark=False):
        bar_x = 14  # mm from left edge
        bar_y_top = 45
        bar_y_bottom = 275
        self.set_draw_color(*(VIOLET_BRIGHT if dark else VIOLET))
        self.set_line_width(0.6)
        self.line(bar_x, bar_y_top, bar_x, bar_y_bottom)

        if total > 1:
            ratio = (slide_number - 1) / (total - 1)
        else:
            ratio = 0.0
        tick_y = bar_y_top + ratio * (bar_y_bottom - bar_y_top)
        self.set_draw_color(*AMBER)
        self.set_line_width(1.2)
        self.line(bar_x - 4, tick_y, bar_x + 7, tick_y)

    # -----------------------------------------------------------------
    # Backgrounds + footer
    # -----------------------------------------------------------------

    def light_gradient_background(self):
        """Lavender #f5f3ff fading to white. Interpolates all 3 RGB channels."""
        steps = 50
        r0, g0, b0 = LAVENDER
        r1, g1, b1 = 255, 255, 255
        for i in range(steps):
            ratio = i / steps
            r = int(r0 + (r1 - r0) * ratio)
            g = int(g0 + (g1 - g0) * ratio)
            b = int(b0 + (b1 - b0) * ratio)
            self.set_fill_color(r, g, b)
            y = i * HEIGHT / steps
            self.rect(0, y, WIDTH, HEIGHT / steps + 1, "F")

    def dark_gradient_background(self):
        """Navy #0f172a fading to dark slate #1e293b (the twist slide)."""
        steps = 50
        r0, g0, b0 = NAVY
        r1, g1, b1 = DARK_SLATE
        for i in range(steps):
            ratio = i / steps
            r = int(r0 + (r1 - r0) * ratio)
            g = int(g0 + (g1 - g0) * ratio)
            b = int(b0 + (b1 - b0) * ratio)
            self.set_fill_color(r, g, b)
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
        style = ""
        if bold:
            style += "B"
        if italic:
            style += "I"
        self.set_font("Helvetica", style, size)
        self.set_text_color(*color)
        self.cell(WIDTH, size * 0.5, text, align="C")

    def _kicker(self, y, text, color=AMBER):
        """Editorial section label: letter-spaced small caps with flanking rules."""
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

    def _pull_quote(self, y, text, attribution, size=15, width_frac=0.76, dark=False):
        """Serif pull quote (verbatim paper text) with an oversized amber mark.

        Returns the y coordinate after the attribution line.
        """
        qw = WIDTH * width_frac
        qx = (WIDTH - qw) / 2

        # Oversized opening quotation mark
        self.set_xy(qx - 13, y - 7)
        self.set_font("Helvetica", "B", 46)
        self.set_text_color(*AMBER)
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

    def _card_stack(
        self, items, start_y, box_h=42, accent=VIOLET, margin=30, title_size=16, desc_size=13
    ):
        """Stacked cards: soft shadow, accent left bar, title + one-line desc."""
        box_w = WIDTH - margin * 2
        gap = 6
        bar_w = 5
        for i, (title, desc) in enumerate(items):
            by = start_y + i * (box_h + gap)
            self._shadow_rect(margin, by, box_w, box_h)
            self.set_fill_color(*WHITE)
            self.set_draw_color(220, 220, 220)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, box_h, "DF")
            self.set_fill_color(*accent)
            self.rect(margin, by, bar_w, box_h, "F")

            self.set_xy(margin + bar_w + 12, by + 8)
            self.set_font("Helvetica", "B", title_size)
            self.set_text_color(*NAVY)
            self.cell(box_w - bar_w - 24, 10, title)

            self.set_xy(margin + bar_w + 12, by + 26)
            self.set_font("Helvetica", "", desc_size)
            self.set_text_color(*GRAY)
            self.cell(box_w - bar_w - 24, 10, desc)
        return start_y + len(items) * (box_h + gap)

    # -----------------------------------------------------------------
    # Equation rendering (matplotlib mathtext -> PNG -> fpdf image)
    # -----------------------------------------------------------------

    def _place_equation_centered(self, path, pw, ph, y, max_w=200):
        aspect = ph / pw
        display_w = min(max_w, WIDTH * 0.78)
        display_h = display_w * aspect
        eq_x = (WIDTH - display_w) / 2
        self.image(path, eq_x, y, display_w)
        return display_h

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

    # -----------------------------------------------------------------
    # Cover motif -- a soft impulse-response curve: flat, then a shock
    # (amber dash), then the dynamic response rising. Sits behind the
    # punchline at low alpha.
    # -----------------------------------------------------------------

    def _render_cover_irf(self):
        fig, ax = plt.subplots(figsize=(10, 3.4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        x_pre = np.linspace(-3.5, 0, 60)
        x_post = np.linspace(0, 8, 160)
        y_pre = np.zeros_like(x_pre)
        y_post = 2.0 * (1 - np.exp(-x_post / 2.2))

        # Shock dash anchored AT the curve baseline (y=0) -- extending it
        # below the flat pre-period line reads as a rendering mistake.
        ax.plot(
            [0, 0],
            [0, 2.3],
            color=AMBER_HEX,
            linewidth=2.4,
            linestyle=(0, (5, 4)),
            alpha=0.6,
            solid_capstyle="butt",
        )
        ax.plot(x_pre, y_pre, color=VIOLET_HEX, linewidth=3.2, alpha=0.45)
        ax.plot(x_post, y_post, color=VIOLET_HEX, linewidth=3.2, alpha=0.45)
        ax.fill_between(x_post, 0, y_post, color=VIOLET_HEX, alpha=0.09, linewidth=0)
        # "shock" hugs its dashed line (low, so it stays clear of the punchline)
        ax.text(
            -0.35,
            0.52,
            "shock",
            fontsize=14,
            color=AMBER_HEX,
            fontweight="bold",
            alpha=0.85,
            ha="right",
        )

        ax.set_xlim(-3.5, 8)
        # Almost no padding below y=0: the curve baseline IS the motif's
        # bottom edge, so page text can sit flush above it.
        ax.set_ylim(-0.1, 2.4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=0.2)
        return self._save_fig(fig, transparent=True)

    # -----------------------------------------------------------------
    # Slide-2 schematic -- contamination: an early-treated unit is still
    # responding to its own treatment when TWFE drafts it as a "control"
    # for a later cohort.
    # -----------------------------------------------------------------

    def _render_contamination(self):
        fig, ax = plt.subplots(figsize=(10, 4.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        x = np.linspace(1, 12, 240)
        early_entry, late_entry = 3.0, 8.0
        y_early = np.where(x < early_entry, 0.0, 2.2 * (1 - np.exp(-(x - early_entry) / 2.6)))
        y_never = np.zeros_like(x)

        ax.plot(x, y_never, color=LIGHT_GRAY_HEX, linewidth=2.6, label="never treated")
        ax.plot(x, y_early, color=VIOLET_HEX, linewidth=3.4, label="treated at t=3")

        ax.axvline(late_entry, color=AMBER_HEX, linewidth=2.2, linestyle=(0, (5, 4)), zorder=3)
        ax.text(
            late_entry + 0.15,
            -0.42,
            "new cohort\ntreated at t=8",
            fontsize=13,
            color=AMBER_HEX,
            fontweight="bold",
            va="top",
        )

        # The contaminated stretch: still responding when drafted as control
        mask = x >= late_entry
        ax.fill_between(
            x[mask], y_never[mask], y_early[mask], color=VIOLET_HEX, alpha=0.12, linewidth=0
        )
        ax.annotate(
            "still responding to its OWN treatment...",
            xy=(9.8, float(2.2 * (1 - np.exp(-(9.8 - early_entry) / 2.6)))),
            xytext=(4.6, 2.62),
            fontsize=14,
            color=VIOLET_HEX,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=VIOLET_HEX, lw=1.6),
        )
        ax.text(
            9.95,
            1.02,
            '...yet TWFE drafts it\nas a "control"',
            fontsize=14,
            color=NAVY_HEX,
            fontweight="bold",
            ha="center",
        )

        ax.set_xlim(1, 12)
        ax.set_ylim(-1.15, 3.0)
        ax.set_xlabel("time", fontsize=13, color=GRAY_HEX)
        ax.set_xticks(range(1, 13))
        ax.set_yticks([])
        ax.tick_params(colors=GRAY_HEX, labelsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(LIGHT_GRAY_HEX)
        ax.legend(loc="upper left", fontsize=12, frameon=False, labelcolor=NAVY_HEX)

        fig.tight_layout(pad=0.3)
        return self._save_fig(fig)

    # -----------------------------------------------------------------
    # Slide-6 weight equation with annotation arrow pointing at the
    # omega >= 0 condition (weights over TREATED cohorts only, g != 0).
    # -----------------------------------------------------------------

    def _render_weights_equation(self):
        fig = plt.figure(figsize=(10, 3.4))
        fig.patch.set_alpha(0)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.5,
            0.82,
            r"$y_{i,t+h} - y_{i,t-1} \;=\; \beta_h\,\Delta D_{it} \;+\; \delta_t^h \;+\; e_{it}^h$",
            fontsize=27,
            ha="center",
            va="center",
            color=NAVY_HEX,
        )
        ax.text(
            0.5,
            0.52,
            r"$E(\beta_h) \;=\; \sum_{g \neq 0}\,\omega_{g,h}\,\tau_h^g\,,"
            r"\qquad \omega_{g,h} \,\geq\, 0$",
            fontsize=27,
            ha="center",
            va="center",
            color=NAVY_HEX,
        )
        ax.annotate(
            "always non-negative -- no forbidden comparisons",
            xy=(0.735, 0.42),
            xytext=(0.60, 0.10),
            fontsize=15,
            color=AMBER_HEX,
            fontweight="bold",
            ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="->", color=AMBER_HEX, lw=1.6, shrinkA=2, shrinkB=4),
        )
        return self._save_fig(fig, dpi=250, transparent=True)

    # -----------------------------------------------------------------
    # Slide-5 mechanism schematic -- panel grid (units x time) with the
    # horizon-h=+2 regression's sample highlighted; the already-treated
    # ("unclean") cohort is greyed out of the control group.
    # -----------------------------------------------------------------

    def _render_clean_control_grid(self):
        fig, ax = plt.subplots(figsize=(10, 6.6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        n_t = 12
        groups = [
            ("Cohort A\n(enters t=3)", 3, range(8, 12)),
            ("Cohort B\n(enters t=6)", 6, range(4, 8)),
            ("Never\ntreated", None, range(0, 4)),
        ]
        t_entry = 6
        h = 2

        cell = 0.86
        for _, entry, rows in groups:
            for r in rows:
                for t in range(1, n_t + 1):
                    treated = entry is not None and t >= entry
                    face = VIOLET_LIGHT_HEX if treated else "#eef2f7"
                    rect = patches.Rectangle(
                        (t - cell / 2, r - cell / 2),
                        cell,
                        cell,
                        facecolor=face,
                        edgecolor="white",
                        linewidth=0.8,
                        zorder=2,
                    )
                    ax.add_patch(rect)

        # Grey-out overlay: cohort A is "unclean" at t=6 (still potentially
        # responding to its own entry) -> excluded from the control group.
        overlay = patches.Rectangle(
            (1 - cell / 2 - 0.07, 8 - cell / 2 - 0.07),
            n_t - 1 + cell + 0.14,
            4 - 1 + cell + 0.14,
            facecolor="white",
            alpha=0.62,
            edgecolor="none",
            zorder=3,
        )
        ax.add_patch(overlay)
        ax.text(
            (1 + n_t) / 2,
            9.5,
            '"unclean" at t=6  ->  EXCLUDED',
            ha="center",
            va="center",
            fontsize=15,
            color=GRAY_HEX,
            fontweight="bold",
            zorder=4,
        )

        for r in range(4, 8):
            ax.add_patch(
                patches.Rectangle(
                    (t_entry - cell / 2, r - cell / 2),
                    cell,
                    cell,
                    facecolor="none",
                    edgecolor=VIOLET_HEX,
                    linewidth=2.6,
                    zorder=5,
                )
            )
        for r in range(0, 4):
            ax.add_patch(
                patches.Rectangle(
                    (t_entry - cell / 2, r - cell / 2),
                    cell,
                    cell,
                    facecolor="none",
                    edgecolor=CYAN_HEX,
                    linewidth=2.6,
                    zorder=5,
                )
            )

        # Long-difference bracket above the grid: y(t+2) - y(t-1)
        brace_y = 12.1
        ax.annotate(
            "",
            xy=(t_entry + h, brace_y),
            xytext=(t_entry - 1, brace_y),
            arrowprops=dict(arrowstyle="->", color=AMBER_HEX, lw=2.2),
            zorder=6,
        )
        ax.text(
            t_entry + (h - 1) / 2,
            brace_y + 0.35,
            r"long difference:  $y_{t+2} - y_{t-1}$",
            ha="center",
            va="bottom",
            fontsize=15,
            color=AMBER_HEX,
            fontweight="bold",
            zorder=6,
        )

        for label, _, rows in groups:
            mid = (min(rows) + max(rows)) / 2
            ax.text(
                -0.4,
                mid,
                label,
                ha="right",
                va="center",
                fontsize=13,
                color=NAVY_HEX,
                fontweight="bold",
            )

        legend_items = [
            (VIOLET_LIGHT_HEX, None, "treated cell", 1.0, -1.5),
            ("#eef2f7", None, "untreated cell", 7.0, -1.5),
            ("none", VIOLET_HEX, "newly treated (enters regression)", 1.0, -2.8),
            ("none", CYAN_HEX, "clean control (enters regression)", 7.0, -2.8),
        ]
        for face, edge, txt, lx, ly in legend_items:
            sq = patches.Rectangle(
                (lx, ly - 0.28),
                0.55,
                0.55,
                facecolor=face if face != "none" else "white",
                edgecolor=edge if edge else "white",
                linewidth=2.2 if edge else 0.8,
                zorder=5,
            )
            ax.add_patch(sq)
            ax.text(lx + 0.75, ly, txt, ha="left", va="center", fontsize=12, color=NAVY_HEX)

        ax.set_xlim(-3.2, n_t + 0.8)
        ax.set_ylim(-3.7, 13.6)
        ax.set_xticks(range(1, n_t + 1))
        ax.set_xticklabels([f"{t}" for t in range(1, n_t + 1)], fontsize=11, color=GRAY_HEX)
        ax.set_yticks([])
        ax.set_xlabel("time", fontsize=13, color=GRAY_HEX)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout(pad=0.4)
        return self._save_fig(fig)

    # -----------------------------------------------------------------
    # Slide-7 event-study plot -- symmetric pre/post estimation, flat
    # pre-trends, dynamic post path, violet CI band. Values illustrative.
    # -----------------------------------------------------------------

    def _render_event_study(self):
        fig, ax = plt.subplots(figsize=(10, 5.4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        pre_h = np.array([-5, -4, -3, -2])
        pre_b = np.array([0.03, -0.04, 0.05, -0.02])
        pre_ci = np.array([0.16, 0.15, 0.14, 0.13])
        post_h = np.arange(0, 9)
        post_b = np.array([1.50, 1.72, 1.90, 2.05, 2.18, 2.29, 2.38, 2.45, 2.50])
        post_ci = np.array([0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.22, 0.24])

        hs = np.concatenate([pre_h, [-1], post_h])
        bs = np.concatenate([pre_b, [0.0], post_b])
        cis = np.concatenate([pre_ci, [0.0], post_ci])

        ax.axhline(0, color=LIGHT_GRAY_HEX, linewidth=0.9, zorder=1)
        ax.axvline(-0.5, color=AMBER_HEX, linewidth=1.8, linestyle=(0, (5, 4)), zorder=2)
        ax.text(
            -0.66,
            2.72,
            "treatment",
            ha="right",
            va="top",
            fontsize=13,
            color=AMBER_HEX,
            fontweight="bold",
        )

        ax.fill_between(hs, bs - cis, bs + cis, color=VIOLET_HEX, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(hs, bs, color=VIOLET_HEX, linewidth=2.4, zorder=4)
        ax.scatter(hs, bs, s=62, color=VIOLET_HEX, edgecolors="white", linewidths=1.1, zorder=5)
        ax.scatter(
            [-1],
            [0],
            s=76,
            facecolors="white",
            edgecolors=VIOLET_HEX,
            linewidths=2.0,
            zorder=6,
        )
        ax.annotate(
            "h = -1 reference",
            xy=(-1, 0),
            xytext=(-3.6, 0.9),
            fontsize=12,
            color=GRAY_HEX,
            arrowprops=dict(arrowstyle="->", color=GRAY_HEX, lw=1.0),
            zorder=6,
        )
        ax.text(
            -3.5,
            -0.62,
            "placebos: the same\nregressions, run at h < 0",
            ha="center",
            fontsize=12.5,
            color=CYAN_HEX,
            fontweight="bold",
        )
        ax.text(
            5.4,
            1.10,
            "dynamic ATT, one\ncoefficient per horizon",
            ha="center",
            fontsize=12.5,
            color=VIOLET_HEX,
            fontweight="bold",
        )

        ax.set_xlabel("event time h (periods since treatment)", fontsize=13, color=NAVY_HEX)
        ax.set_ylabel(r"$\beta_h$", fontsize=14, color=NAVY_HEX)
        ax.set_xticks(list(range(-5, 9)))
        ax.tick_params(colors=GRAY_HEX, labelsize=11)
        ax.set_ylim(-1.0, 3.0)
        for spine in ax.spines.values():
            spine.set_color(LIGHT_GRAY_HEX)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout(pad=0.4)
        return self._save_fig(fig)

    # -----------------------------------------------------------------
    # Code block (dark-slate bg with token highlighting)
    # -----------------------------------------------------------------

    def _add_code_block(self, x, y, w, token_lines, font_size=12, line_height=10):
        n_lines = len(token_lines)
        total_h = n_lines * line_height + 24

        self._shadow_rect(x, y, w, total_h)
        self.set_fill_color(*DARK_SLATE)
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

    # =================================================================
    # SLIDES
    # =================================================================

    def slide_01_cover(self):
        """Slide 1: macro built LPs for shocks; your event study is one."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(1)

        self.draw_split_logo(34, size=40)

        self.centered_text(88, "Macro built local projections to read", size=30)
        self.centered_text(112, "the dynamic effects of shocks.", size=30)

        # IRF motif behind the punchline
        irf_path, _ipw, _iph = self._render_cover_irf()
        motif_w = 222
        self.image(irf_path, (WIDTH - motif_w) / 2, 148, motif_w)

        # Punchline sits fully ABOVE the motif's baseline (its bottom edge)
        self.centered_text(156, "Your event study", size=52, color=VIOLET)
        self.centered_text(188, "is one.", size=52, color=VIOLET)

        self.set_xy(0, HEIGHT - 78)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*VIOLET)
        self.cell(WIDTH, 8, "LP-DiD: Local Projections meet DiD. Now in diff-diff.", align="C")
        self.set_xy(0, HEIGHT - 64)
        self.set_font("Helvetica", "I", 12)
        self.set_text_color(*GRAY)
        self.cell(
            WIDTH,
            8,
            "Dube, Girardi, Jordà & Taylor (2025), J. of Applied Econometrics 40(5).",
            align="C",
        )
        self.set_xy(0, HEIGHT - 51)
        self.set_font("Helvetica", "", 11.5)
        self.set_text_color(*LIGHT_GRAY)
        self.cell(
            WIDTH, 8, "Co-author Òscar Jordà introduced local projections in 2005.", align="C"
        )

        self.add_footer()

    def slide_02_problem(self):
        """Slide 2: negative weighting -- contamination schematic + pull quote."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(2)

        self._kicker(34, "The Problem")
        self.centered_text(52, "Already-treated units sneak", size=37)
        self.centered_text(80, "into your control group.", size=37, color=VIOLET)

        self._pull_quote(
            112,
            "Previously treated units, which might still be experiencing lagged"
            " ... treatment effects, are implicitly used as controls for newly"
            " treated ones.",
            "- Dube, Girardi, Jordà & Taylor (2025)",
            size=15,
        )

        plot_path, ppw, pph = self._render_contamination()
        plot_w = WIDTH * 0.78
        plot_h = plot_w * (pph / ppw)
        self.image(plot_path, (WIDTH - plot_w) / 2, 162, plot_w)

        fact_y = 162 + plot_h + 8
        self.centered_text(
            fact_y,
            "The TWFE estimate could even lie outside the range",
            size=15,
            bold=True,
            color=NAVY,
        )
        self.centered_text(
            fact_y + 14,
            "of group-specific treatment effects.",
            size=15,
            bold=True,
            color=NAVY,
        )

        self.add_footer()

    def slide_03_twist(self):
        """Slide 3 (DARK): naive LP inherits the same bias -- the pivot."""
        self.add_page()
        self.dark_gradient_background()
        self._draw_vertical_sidebar(3, dark=True)

        self._kicker(38, "The Catch")

        self.centered_text(78, "So just point macro's tool", size=38, color=WHITE)
        self.centered_text(106, "at your rollout?", size=38, color=WHITE)

        self.centered_text(152, "Not so fast.", size=58, color=VIOLET_BRIGHT)

        self.centered_text(
            206,
            "A naive LP implementation inherits the same",
            size=17,
            bold=False,
            color=WHITE,
        )
        self.centered_text(
            222,
            "negative-weighting bias as TWFE.",
            size=17,
            bold=False,
            color=WHITE,
        )
        self.centered_text(
            240,
            "(Dube, Girardi, Jordà & Taylor 2025, Eqs. 6-7)",
            size=12,
            bold=False,
            italic=True,
            color=LIGHT_GRAY,
        )

        self.centered_text(266, "The estimator isn't the fix.", size=21, color=AMBER)
        self.centered_text(284, "The comparisons are.", size=21, color=AMBER)

        self.add_footer(dark=True)

    def slide_04_insight(self):
        """Slide 4: the 'underappreciated feature' -- LP + clean controls."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(4)

        self._kicker(34, "The Insight")
        self.centered_text(52, "The underappreciated feature.", size=38)

        self._pull_quote(
            88,
            "An underappreciated feature of the LP framework is that it is"
            " straightforward to limit the set of permissible comparisons"
            " based on a desired criterion, such as past treatment history.",
            "- Dube, Girardi, Jordà & Taylor (2025)",
            size=15,
        )

        # Fusion diagram: LP + clean controls = LP-DiD
        card_w = 100
        card_h = 56
        y_cards = 168
        x1 = 26
        x2 = WIDTH - 26 - card_w

        for cx, title, l1, l2 in [
            (
                x1,
                "Local projections",
                "Jordà (2005). Built to estimate",
                "dynamic average effects.",
            ),
            (x2, "Clean controls", "Cengiz et al. (2019) spirit.", '"Unclean" units stay out.'),
        ]:
            self._shadow_rect(cx, y_cards, card_w, card_h)
            self.set_fill_color(*WHITE)
            self.set_draw_color(*VIOLET)
            self.set_line_width(0.7)
            self.rect(cx, y_cards, card_w, card_h, "DF")
            self.set_xy(cx + 8, y_cards + 9)
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(*NAVY)
            self.cell(card_w - 16, 9, title)
            for j, line in enumerate((l1, l2)):
                self.set_xy(cx + 8, y_cards + 27 + j * 12)
                self.set_font("Helvetica", "", 12.5)
                self.set_text_color(*GRAY)
                self.cell(card_w - 16, 9, line)

        # The "+" between cards
        self.set_xy(0, y_cards + card_h / 2 - 9)
        self.set_font("Helvetica", "B", 40)
        self.set_text_color(*VIOLET)
        self.cell(WIDTH, 18, "+", align="C")

        # "=" then the result card
        self.set_xy(0, y_cards + card_h + 8)
        self.set_font("Helvetica", "B", 30)
        self.set_text_color(*NAVY)
        self.cell(WIDTH, 14, "=", align="C")

        res_y = y_cards + card_h + 26
        res_w = WIDTH - 2 * 26
        res_h = 46
        self._shadow_rect(26, res_y, res_w, res_h)
        self.set_fill_color(*VIOLET)
        self.rect(26, res_y, res_w, res_h, "F")
        self.set_xy(26, res_y + 8)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*WHITE)
        self.cell(res_w, 12, "LP-DiD", align="C")
        self.set_xy(26, res_y + 26)
        self.set_font("Helvetica", "", 13.5)
        self.set_text_color(255, 255, 255)
        self.cell(
            res_w, 9, "Dynamic treatment effects with provably non-negative weights.", align="C"
        )

        self.add_footer()

    def slide_05_mechanism(self):
        """Slide 5: 'unclean' observations leave the control group."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(5)

        self._kicker(34, "How It Works")
        self.centered_text(52, '"Unclean" observations', size=37)
        self.centered_text(80, "leave the control group.", size=37, color=VIOLET)

        plot_path, ppw, pph = self._render_clean_control_grid()
        plot_w = WIDTH * 0.72
        plot_h = plot_w * (pph / ppw)
        plot_y = 106
        self.image(plot_path, (WIDTH - plot_w) / 2, plot_y, plot_w)

        cap_y = plot_y + plot_h + 8
        self.centered_text(
            cap_y,
            "Each horizon is one regression you could run by hand.",
            size=17,
            bold=True,
            color=VIOLET,
        )
        self.centered_text(
            cap_y + 16,
            "In the absorbing (default) path, already-treated units never serve as controls.",
            size=13.5,
            bold=False,
            italic=True,
            color=GRAY,
        )

        self.add_footer()

    def slide_06_math(self):
        """Slide 6: non-negative weights + 'subsumes' equivalence map."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(6)

        self._kicker(34, "The Math")
        self.centered_text(52, "Non-negative weights,", size=36)
        self.centered_text(80, "guaranteed.", size=36, color=VIOLET)

        eq_path, epw, eph = self._render_weights_equation()
        eq_h = self._place_equation_centered(eq_path, epw, eph, 102, max_w=205)

        intro_y = 102 + eq_h + 10
        self.centered_text(
            intro_y,
            'The paper\'s abstract says it "subsumes many of the recent solutions". Specifically:',
            size=13.5,
            bold=False,
            italic=True,
            color=GRAY,
        )

        rows = [
            ("reweight=True", "==  Callaway-Sant'Anna", "exact  (Sec. 3.7)"),
            ("default (variance-weighted)", "==  Cengiz et al. stacked", "exact  (Sec. 3.7)"),
            ("h=0, single cohort", "==  classic 2x2 DiD", "exact  (Sec. 2.2)"),
            ("pmd='max'", "->  BJS imputation", "exact single-cohort  (fn. 10-11)"),
        ]

        margin = 32
        box_w = WIDTH - margin * 2
        row_h = 25
        gap = 5
        start_y = intro_y + 14
        col1_w = 86

        for i, (setting, target, source) in enumerate(rows):
            by = start_y + i * (row_h + gap)
            self._shadow_rect(margin, by, box_w, row_h)
            self.set_fill_color(*WHITE)
            self.set_draw_color(220, 220, 220)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, row_h, "DF")
            self.set_fill_color(*CYAN)
            self.rect(margin, by, 5, row_h, "F")

            # Line 1: setting -> target; line 2: source citation (right-
            # aligned on its own line so it can never collide with line 1).
            self.set_xy(margin + 13, by + 4)
            self.set_font("Courier", "B", 13)
            self.set_text_color(*VIOLET_DARK)
            self.cell(col1_w, 10, setting)

            self.set_xy(margin + 13 + col1_w, by + 4)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*NAVY)
            self.cell(box_w - col1_w - 26, 10, target)

            self.set_xy(margin + 13, by + 14)
            self.set_font("Helvetica", "I", 11)
            self.set_text_color(*GRAY)
            self.cell(box_w - 26, 8, source, align="R")

        self.add_footer()

    def slide_07_output(self):
        """Slide 7: native event study, symmetric pre/post estimation."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(7)

        self._kicker(34, "What You Get")
        self.centered_text(52, "Pre-trends, estimated", size=37)
        self.centered_text(80, "symmetrically.", size=37, color=VIOLET)

        plot_path, ppw, pph = self._render_event_study()
        plot_w = WIDTH * 0.84
        plot_h = plot_w * (pph / ppw)
        plot_y = 106
        self.image(plot_path, (WIDTH - plot_w) / 2, plot_y, plot_w)

        cap_y = plot_y + plot_h + 10
        self.centered_text(
            cap_y,
            "The event study is native - not a post-hoc aggregation.",
            size=17,
            bold=True,
            color=VIOLET,
        )
        self.centered_text(
            cap_y + 17,
            "Pre and post coefficients are estimated symmetrically - the paper notes this",
            size=13,
            bold=False,
            color=GRAY,
        )
        self.centered_text(
            cap_y + 30,
            "avoids the pretrend-interpretation difficulties raised in Roth (2024).",
            size=13,
            bold=False,
            color=GRAY,
        )

        self.add_footer()

    def slide_08_code(self):
        """Slide 8: code example -- fit() + the reweight dial."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(8)

        self._kicker(34, "The Code")
        self.centered_text(52, "It's just a regression.", size=42)
        self.centered_text(
            88,
            "Same sklearn-like API as every diff-diff estimator.",
            size=14,
            bold=False,
            color=GRAY,
        )

        margin = 24
        code_y = 106

        token_lines = [
            [
                ("from", SLATE_CODE),
                (" diff_diff ", WHITE),
                ("import", SLATE_CODE),
                (" LPDiD", WHITE),
            ],
            [],
            [
                ("result", WHITE),
                (" = ", SLATE_CODE),
                ("LPDiD", AMBER_CODE),
                ("(", WHITE),
            ],
            [
                ("    ", WHITE),
                ("pre_window", WHITE),
                ("=", SLATE_CODE),
                ("5", AMBER_CODE),
                (", ", SLATE_CODE),
                ("post_window", WHITE),
                ("=", SLATE_CODE),
                ("10", AMBER_CODE),
                (",", SLATE_CODE),
            ],
            [
                (").", WHITE),
                ("fit(", WHITE),
            ],
            [
                ("    ", WHITE),
                ("data,", WHITE),
            ],
            [
                ("    ", WHITE),
                ("outcome", WHITE),
                ("=", SLATE_CODE),
                ("'revenue'", AMBER_CODE),
                (", ", SLATE_CODE),
                ("unit", WHITE),
                ("=", SLATE_CODE),
                ("'store'", AMBER_CODE),
                (",", SLATE_CODE),
            ],
            [
                ("    ", WHITE),
                ("time", WHITE),
                ("=", SLATE_CODE),
                ("'week'", AMBER_CODE),
                (", ", SLATE_CODE),
                ("treatment", WHITE),
                ("=", SLATE_CODE),
                ("'treated'", AMBER_CODE),
                (",", SLATE_CODE),
            ],
            [(")", WHITE)],
            [],
            [
                ("print(", WHITE),
                ("result.att", WHITE),
                (")", WHITE),
                ("        # pooled post ATT = 2.31", LIGHT_GRAY),
            ],
            [
                ("result.event_study", WHITE),
                ("       # one beta per horizon", LIGHT_GRAY),
            ],
            [],
            [
                ("# flip one switch:", LIGHT_GRAY),
            ],
            [
                ("#   reweight=True   ->  Callaway-Sant'Anna", LIGHT_GRAY),
            ],
            [
                ("#   non_absorbing=  ->  on/off treatments (entry effects)", LIGHT_GRAY),
            ],
            [
                ("#   survey_design=  ->  pweights + strata + PSUs (default path)", LIGHT_GRAY),
            ],
        ]

        code_h = self._add_code_block(
            margin,
            code_y,
            WIDTH - margin * 2,
            token_lines,
            font_size=12,
            line_height=10,
        )

        sub_y = code_y + code_h + 10
        self.centered_text(
            sub_y,
            "Callaway-Sant'Anna is literally a keyword argument.",
            size=14,
            bold=False,
            color=GRAY,
        )

        self.add_footer()

    def slide_09_production_ready(self):
        """Slide 9: production-ready feature grid (2x3)."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(9)

        self._kicker(34, "In Production")
        self.centered_text(52, "Production-ready.", size=46, color=CYAN)

        margin = 26
        grid_gap = 9
        card_w = (WIDTH - margin * 2 - grid_gap) / 2
        card_h = 56
        start_y = 96

        features = [
            ("Cluster-Robust SEs", "Unit-level by default,\nmatches Stata lpdid"),
            ("Covariates, Two Ways", "RA path (BJS-style) or\ndirect with guardrails"),
            ("PMD Baselines", "Premean differencing +\npooled pre/post ATTs"),
            ("Non-Absorbing Entry Effects", "First-entry & effect-\nstabilization (Eq. 12/13)"),
            ("Survey Designs", "pweights, strata, PSU, FPC;\nBinder TSL (default path)"),
            ("Composition Control", "no_composition fixes the\npost-window sample"),
        ]

        for idx, (title, desc) in enumerate(features):
            row = idx // 2
            col = idx % 2
            cx = margin + col * (card_w + grid_gap)
            cy = start_y + row * (card_h + grid_gap)

            self._shadow_rect(cx, cy, card_w, card_h)
            self.set_fill_color(*WHITE)
            self.set_draw_color(*VIOLET)
            self.set_line_width(0.6)
            self.rect(cx, cy, card_w, card_h, "DF")

            self.set_xy(cx + 10, cy + 8)
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(*CYAN)
            self.cell(card_w - 20, 10, title)

            for j, line in enumerate(desc.split("\n")):
                self.set_xy(cx + 10, cy + 25 + j * 12)
                self.set_font("Helvetica", "", 12.5)
                self.set_text_color(*GRAY)
                self.cell(card_w - 20, 10, line)

        comp_y = start_y + 3 * (card_h + grid_gap) + 6
        self.set_xy(0, comp_y)
        self.set_font("Helvetica", "I", 12)
        self.set_text_color(*LIGHT_GRAY)
        self.cell(
            WIDTH,
            8,
            "Composable where validated - survey_design runs the default path only.",
            align="C",
        )

        self.add_footer()

    def slide_10_validated(self):
        """Slide 10: validation story -- authors' tooling + tested equivalences."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(10)

        self._kicker(34, "Validated")
        self.centered_text(52, "Validated.", size=44, color=VIOLET)
        self.centered_text(
            92,
            "Against the authors' own tooling - and against our own shelf.",
            size=14,
            bold=False,
            italic=True,
            color=GRAY,
        )

        items = [
            (
                "Authors' R recipes (danielegirardi/lpdid)",
                "Event-study and pooled estimands match to ~1e-12.",
            ),
            (
                "Independent fixest reconstruction",
                "Non-absorbing Eq. 12/13, variance-weighted: point + SE to ~1e-13.",
            ),
            (
                "survey::svyglm, end to end",
                "Default-path survey: per-horizon point, SE, and df all pinned.",
            ),
            (
                "Equivalences tested, not cited",
                "reweight == our CallawaySantAnna; PMD == our ImputationDiD (1 cohort).",
            ),
        ]
        self._card_stack(items, start_y=114, box_h=44)

        self.add_footer()

    def slide_11_cta(self):
        """Slide 11: CTA -- pip install + GitHub."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(11)

        self.centered_text(58, "Now in diff-diff.", size=24, bold=False, italic=True, color=GRAY)
        self.centered_text(88, "LP-DiD.", size=54, color=VIOLET)

        badge_w = 232
        badge_h = 44
        badge_x = (WIDTH - badge_w) / 2
        badge_y = 156
        self._shadow_rect(badge_x, badge_y, badge_w, badge_h)
        self.set_fill_color(*VIOLET)
        self.rect(badge_x, badge_y, badge_w, badge_h, "F")

        self.set_xy(badge_x, badge_y + 13)
        self.set_font("Courier", "B", 17)
        self.set_text_color(*WHITE)
        self.cell(badge_w, 16, "$ pip install --upgrade diff-diff", align="C")

        self.centered_text(222, "github.com/igerber/diff-diff", size=19, color=VIOLET)

        self.draw_split_logo(256, size=28)

        self.centered_text(
            284, "Difference-in-Differences for Python", size=15, bold=False, color=GRAY
        )

        self.add_footer()


def main():
    pdf = LPDiDCarouselPDF()
    try:
        pdf.slide_01_cover()
        pdf.slide_02_problem()
        pdf.slide_03_twist()
        pdf.slide_04_insight()
        pdf.slide_05_mechanism()
        pdf.slide_06_math()
        pdf.slide_07_output()
        pdf.slide_08_code()
        pdf.slide_09_production_ready()
        pdf.slide_10_validated()
        pdf.slide_11_cta()

        output_path = Path(__file__).parent / "diff-diff-lpdid-carousel.pdf"
        pdf.output(str(output_path))
        print(f"PDF saved to: {output_path}")
    finally:
        pdf.cleanup()


if __name__ == "__main__":
    main()
