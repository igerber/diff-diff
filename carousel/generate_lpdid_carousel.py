#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for LPDiD (Dube, Girardi, Jorda & Taylor 2025) launch.

Mirrors the architecture of ``generate_spillover_carousel.py`` (magazine
sidebar with progress tick, light gradient background, split-color logo,
footer wordmark) but introduces the "Horizon" palette: violet primary /
sky-cyan clean-control accent / amber impact accent on a lavender -> white
gradient. The temporal-forward identity signals LP-DiD's horizon-by-horizon
local-projection nature.

Narrative spine ("one regression to rule them all"):

1.  Cover        -- the estimator zoo, then the punchline (single LOTR nod)
2.  Shelf        -- fragmentation problem: four fixes, four frameworks
3.  Real world   -- four scenarios of tool-choice anxiety
4.  Introducing  -- "It's just a regression" (3 capability cards)
5.  Mechanism    -- clean-control panel schematic at one horizon
6.  The math     -- LP-DiD equation + the proven equivalence map
7.  Output       -- native event-study plot (pre-trends = negative horizons)
8.  Code         -- sklearn-like fit(); reweight=True == Callaway-Sant'Anna
9.  Production   -- 2x3 feature grid (incl. non-absorbing + survey designs)
10. Validated    -- authors' tooling, fixest, svyglm, tested equivalences
11. CTA          -- pip install + GitHub

Claim discipline (verified against docs/methodology/papers/dube-2025-review.md):
- The "zoo" named on the cover / shelf / equivalence map is EXACTLY the four
  estimators the paper proves LP-DiD nests: Callaway-Sant'Anna, Cengiz-style
  stacking, classic 2x2 DiD, and (single-cohort) BJS. Sun-Abraham has no
  documented LP-DiD equivalence and is deliberately absent from the
  unification arc.
- CS and Cengiz-stacked equivalences are EXACT (paper Section 3.7).
- BJS equality holds only for PMD k=t-1 single-cohort (fn. 10-11); phrased
  as "BJS-style / exact single-cohort" everywhere. Never a bare "==".
- NO speed claims: the paper's Table 2 speedups are Stata-implementation
  artifacts; diff-diff's own vectorized CS is faster than LPDiD (verified
  empirically 2026-07-01), so speed is omitted from the narrative.

Run with::

    python carousel/generate_lpdid_carousel.py

Produces ``carousel/diff-diff-lpdid-carousel.pdf``. Generation requires
``fpdf2``, ``Pillow``, and ``matplotlib`` (carousel-only dependencies, not
part of the library's install or dev extras).
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
CYAN = (8, 145, 178)  # #0891b2  clean-control / forward accent
CYAN_LIGHT = (165, 243, 252)  # #a5f3fc
AMBER = (245, 158, 11)  # #f59e0b  impact / event-time pop
LAVENDER = (245, 243, 255)  # #f5f3ff  gradient start

# Text + structural (shared with the spillover deck for legibility)
NAVY = (15, 23, 42)  # #0f172a  primary text
GRAY = (100, 116, 139)  # #64748b  secondary text
LIGHT_GRAY = (148, 163, 184)  # #94a3b8  fine print
WHITE = (255, 255, 255)
DARK_SLATE = (30, 41, 59)  # #1e293b  code block bg
AMBER_CODE = (252, 211, 77)  # #fcd34d  code string literals
SLATE_CODE = (148, 163, 184)  # #94a3b8  code keyword tone

# Hex equivalents for matplotlib
VIOLET_HEX = "#6d28d9"
VIOLET_DARK_HEX = "#5b21b6"
VIOLET_LIGHT_HEX = "#c4b5fd"
CYAN_HEX = "#0891b2"
CYAN_LIGHT_HEX = "#a5f3fc"
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
    # VIOLET; tick is AMBER so the accent reads as a deliberate progress
    # marker.
    # -----------------------------------------------------------------

    def _draw_vertical_sidebar(self, slide_number, total=TOTAL_SLIDES):
        bar_x = 14  # mm from left edge
        bar_y_top = 45
        bar_y_bottom = 275
        self.set_draw_color(*VIOLET)
        self.set_line_width(0.6)
        self.line(bar_x, bar_y_top, bar_x, bar_y_bottom)

        if total > 1:
            ratio = (slide_number - 1) / (total - 1)
        else:
            ratio = 0.0
        tick_y = bar_y_top + ratio * (bar_y_bottom - bar_y_top)
        self.set_draw_color(*AMBER)
        self.set_line_width(0.9)
        self.line(bar_x - 4, tick_y, bar_x + 7, tick_y)

    # -----------------------------------------------------------------
    # Background + footer
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

    def add_footer(self):
        """Centered split-color ``diff-diff vX.Y.Z`` wordmark."""
        self.set_font("Helvetica", "B", 12)
        dd_text = "diff-diff "
        v_text = VERSION_LABEL
        dd_w = self.get_string_width(dd_text)
        v_w = self.get_string_width(v_text)
        start_x = (WIDTH - dd_w - v_w) / 2

        self.set_xy(start_x, HEIGHT - 18)
        self.set_text_color(*GRAY)
        self.cell(dd_w, 10, dd_text)
        self.set_text_color(*VIOLET)
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
    # Equation rendering (matplotlib mathtext -> PNG -> fpdf image)
    # -----------------------------------------------------------------

    def _place_equation_centered(self, path, pw, ph, y, max_w=200):
        aspect = ph / pw
        display_w = min(max_w, WIDTH * 0.75)
        display_h = display_w * aspect
        eq_x = (WIDTH - display_w) / 2
        self.image(path, eq_x, y, display_w)
        return display_h

    # -----------------------------------------------------------------
    # Slide-6 weight equation with annotation arrow pointing at the
    # omega >= 0 condition specifically (the paper's central result).
    # -----------------------------------------------------------------

    def _render_weights_equation(self):
        fig = plt.figure(figsize=(10, 3.4))
        fig.patch.set_alpha(0)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # The per-horizon regression (paper Eq. 4 restricted by Eq. 8)
        ax.text(
            0.5,
            0.82,
            r"$y_{i,t+h} - y_{i,t-1} \;=\; \beta_h\,\Delta D_{it} \;+\; \delta_t^h \;+\; e_{it}^h$",
            fontsize=25,
            ha="center",
            va="center",
            color=NAVY_HEX,
        )

        # The estimand (paper Eqs. 9-10): non-negative weights over TREATED
        # cohorts only (g != 0; the never-treated cohort is the control).
        ax.text(
            0.5,
            0.52,
            r"$E(\beta_h) \;=\; \sum_{g \neq 0}\,\omega_{g,h}\,\tau_h^g\,,"
            r"\qquad \omega_{g,h} \,\geq\, 0$",
            fontsize=25,
            ha="center",
            va="center",
            color=NAVY_HEX,
        )

        # Arrow + label pointing UP at the omega >= 0 condition, which
        # sits right of center in the rendered line.
        ax.annotate(
            "always non-negative -- no forbidden comparisons",
            xy=(0.735, 0.42),
            xytext=(0.60, 0.10),
            fontsize=14,
            color=AMBER_HEX,
            fontweight="bold",
            ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="->", color=AMBER_HEX, lw=1.5, shrinkA=2, shrinkB=4),
        )

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.06, transparent=True)
        plt.close(fig)
        with PILImage.open(path) as img:
            pw, ph = img.size
        self._temp_files.append(path)
        return path, pw, ph

    # -----------------------------------------------------------------
    # Slide-5 mechanism schematic -- panel grid (units x time), colored
    # by treatment status, with the horizon-h=+2 regression's sample
    # highlighted: newly-treated entry cells (violet outline), clean
    # controls (cyan outline), already-treated cohort greyed out.
    # -----------------------------------------------------------------

    def _render_clean_control_grid(self):
        fig, ax = plt.subplots(figsize=(10, 6.6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        n_t = 12  # periods 1..12
        groups = [
            ("Cohort A\n(enters t=3)", 3, range(8, 12)),
            ("Cohort B\n(enters t=6)", 6, range(4, 8)),
            ("Never\ntreated", None, range(0, 4)),
        ]
        t_entry = 6  # the regression under focus: cohort B entry
        h = 2  # horizon +2

        cell = 0.86  # cell side (leaves gutters)
        for label, entry, rows in groups:
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

        # Grey-out overlay on cohort A (already treated at t_entry) --
        # excluded from the h=+2 regression at t=6.
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
            "already treated at t=6  ->  EXCLUDED",
            ha="center",
            va="center",
            fontsize=13,
            color=GRAY_HEX,
            fontweight="bold",
            zorder=4,
        )

        # Newly-treated entry cells (cohort B at t=6): bold violet outline
        for r in range(4, 8):
            rect = patches.Rectangle(
                (t_entry - cell / 2, r - cell / 2),
                cell,
                cell,
                facecolor="none",
                edgecolor=VIOLET_HEX,
                linewidth=2.4,
                zorder=5,
            )
            ax.add_patch(rect)

        # Clean controls (never-treated at t=6): bold cyan outline
        for r in range(0, 4):
            rect = patches.Rectangle(
                (t_entry - cell / 2, r - cell / 2),
                cell,
                cell,
                facecolor="none",
                edgecolor=CYAN_HEX,
                linewidth=2.4,
                zorder=5,
            )
            ax.add_patch(rect)

        # Long-difference bracket ABOVE the whole grid (kept clear of the
        # cohort-A EXCLUDED overlay): y(t+2) - y(t-1), spanning t=5..8.
        brace_y = 12.1
        ax.annotate(
            "",
            xy=(t_entry + h, brace_y),
            xytext=(t_entry - 1, brace_y),
            arrowprops=dict(arrowstyle="->", color=AMBER_HEX, lw=2.0),
            zorder=6,
        )
        ax.text(
            t_entry + (h - 1) / 2,
            brace_y + 0.35,
            r"long difference:  $y_{t+2} - y_{t-1}$",
            ha="center",
            va="bottom",
            fontsize=13,
            color=AMBER_HEX,
            fontweight="bold",
            zorder=6,
        )

        # Group labels on the left
        for label, _, rows in groups:
            mid = (min(rows) + max(rows)) / 2
            ax.text(
                -0.4,
                mid,
                label,
                ha="right",
                va="center",
                fontsize=11,
                color=NAVY_HEX,
                fontweight="bold",
            )

        # Legend -- 2x2 grid so nothing clips at the right axis limit
        legend_items = [
            (VIOLET_LIGHT_HEX, None, "treated cell", 1.0, -1.5),
            ("#eef2f7", None, "untreated cell", 7.0, -1.5),
            ("none", VIOLET_HEX, "newly treated (enters regression)", 1.0, -2.7),
            ("none", CYAN_HEX, "clean control (enters regression)", 7.0, -2.7),
        ]
        for face, edge, txt, lx, ly in legend_items:
            sq = patches.Rectangle(
                (lx, ly - 0.28),
                0.55,
                0.55,
                facecolor=face if face != "none" else "white",
                edgecolor=edge if edge else "white",
                linewidth=2.0 if edge else 0.8,
                zorder=5,
            )
            ax.add_patch(sq)
            ax.text(lx + 0.75, ly, txt, ha="left", va="center", fontsize=10.5, color=NAVY_HEX)

        ax.set_xlim(-3.2, n_t + 0.8)
        ax.set_ylim(-3.6, 13.6)
        ax.set_xticks(range(1, n_t + 1))
        ax.set_xticklabels([f"{t}" for t in range(1, n_t + 1)], fontsize=9, color=GRAY_HEX)
        ax.set_yticks([])
        ax.set_xlabel("time", fontsize=11, color=GRAY_HEX)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout(pad=0.4)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.12, facecolor="white")
        plt.close(fig)
        with PILImage.open(path) as img:
            pw, ph = img.size
        self._temp_files.append(path)
        return path, pw, ph

    # -----------------------------------------------------------------
    # Slide-7 event-study plot -- flat pre-trends, dynamic post path,
    # violet CI band, amber treatment line. Values are illustrative
    # (styled after the staggered DGP used in the library benchmarks).
    # -----------------------------------------------------------------

    def _render_event_study(self):
        fig, ax = plt.subplots(figsize=(10, 5.6))
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
        ax.axvline(-0.5, color=AMBER_HEX, linewidth=1.6, linestyle=(0, (5, 4)), zorder=2)
        ax.text(
            -0.62,
            2.72,
            "treatment",
            ha="right",
            va="top",
            fontsize=11,
            color=AMBER_HEX,
            fontweight="bold",
        )

        ax.fill_between(
            hs,
            bs - cis,
            bs + cis,
            color=VIOLET_HEX,
            alpha=0.13,
            linewidth=0,
            zorder=2,
        )
        ax.plot(hs, bs, color=VIOLET_HEX, linewidth=2.2, zorder=4)
        ax.scatter(
            hs,
            bs,
            s=52,
            color=VIOLET_HEX,
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        # Reference period h = -1: open marker, coefficient fixed at 0
        ax.scatter(
            [-1],
            [0],
            s=64,
            facecolors="white",
            edgecolors=VIOLET_HEX,
            linewidths=1.8,
            zorder=6,
        )
        ax.annotate(
            "h = -1 reference",
            xy=(-1, 0),
            xytext=(-3.4, 0.85),
            fontsize=10.5,
            color=GRAY_HEX,
            arrowprops=dict(arrowstyle="->", color=GRAY_HEX, lw=0.9),
            zorder=6,
        )
        ax.text(
            -3.5,
            -0.55,
            "pre-trends: flat",
            ha="center",
            fontsize=11,
            color=CYAN_HEX,
            fontweight="bold",
        )
        ax.text(
            5.4,
            1.15,
            "dynamic ATT, one\ncoefficient per horizon",
            ha="center",
            fontsize=11,
            color=VIOLET_HEX,
            fontweight="bold",
        )

        ax.set_xlabel("event time h (periods since treatment)", fontsize=12, color=NAVY_HEX)
        ax.set_ylabel(r"$\beta_h$", fontsize=13, color=NAVY_HEX)
        ax.set_xticks(list(range(-5, 9)))
        ax.tick_params(colors=GRAY_HEX, labelsize=9)
        ax.set_ylim(-0.8, 3.0)
        for spine in ax.spines.values():
            spine.set_color(LIGHT_GRAY_HEX)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout(pad=0.4)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.12, facecolor="white")
        plt.close(fig)
        with PILImage.open(path) as img:
            pw, ph = img.size
        self._temp_files.append(path)
        return path, pw, ph

    # -----------------------------------------------------------------
    # Code block (dark-slate bg with token highlighting)
    # -----------------------------------------------------------------

    def _add_code_block(self, x, y, w, token_lines, font_size=13, line_height=12):
        n_lines = len(token_lines)
        total_h = n_lines * line_height + 24

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

    # -----------------------------------------------------------------
    # Card row helper (title + one-line description, accent left bar)
    # -----------------------------------------------------------------

    def _card_stack(self, items, start_y, box_h=40, accent=VIOLET, margin=30):
        box_w = WIDTH - margin * 2
        gap = 5
        bar_w = 4
        for i, (title, desc) in enumerate(items):
            by = start_y + i * (box_h + gap)
            self.set_fill_color(*WHITE)
            self.set_draw_color(220, 220, 220)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, box_h, "DF")
            self.set_fill_color(*accent)
            self.rect(margin, by, bar_w, box_h, "F")

            self.set_xy(margin + bar_w + 12, by + 8)
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(*NAVY)
            self.cell(box_w - bar_w - 24, 10, title)

            self.set_xy(margin + bar_w + 12, by + 25)
            self.set_font("Helvetica", "", 12)
            self.set_text_color(*GRAY)
            self.cell(box_w - bar_w - 24, 10, desc)
        return start_y + len(items) * (box_h + gap)

    # =================================================================
    # SLIDES
    # =================================================================

    def slide_01_cover(self):
        """Slide 1: the zoo, then the punchline (the single LOTR nod)."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(1)

        self.draw_split_logo(38, size=42)

        # The zoo -- exactly the four estimators LP-DiD provably nests
        # (paper Secs. 2.2 / 3.7, fn. 10-11), so the punchline is cashable.
        # Qualified names carry the equivalence scope on the user-visible
        # surface: "Cengiz-style" (the paper's stacked target, not any
        # stacked variant) and "BJS-style" (exact only single-cohort PMD).
        self.centered_text(104, "Callaway-Sant'Anna. Cengiz-style stacking.", size=27)
        self.centered_text(128, "BJS-style imputation. The classic 2x2.", size=27)

        # Punchline
        self.centered_text(186, "One regression", size=50, color=VIOLET)
        self.centered_text(216, "to rule them all.", size=50, color=VIOLET)

        # Byline
        self.set_xy(0, HEIGHT - 72)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*VIOLET)
        self.cell(WIDTH, 8, "LP-DiD: Local Projections DiD.", align="C")
        self.set_xy(0, HEIGHT - 60)
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 8, "Now in diff-diff.", align="C")
        self.set_xy(0, HEIGHT - 48)
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 8, "Dube, Girardi, Jordà & Taylor (2025).", align="C")
        self.set_xy(0, HEIGHT - 37)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*LIGHT_GRAY)
        self.cell(WIDTH, 8, "Journal of Applied Econometrics 40(5)", align="C")

        self.add_footer()

    def slide_02_shelf(self):
        """Slide 2: fragmentation -- four fixes, four frameworks, one shelf."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(2)

        self.centered_text(40, "The staggered revolution", size=36)
        self.centered_text(72, "left you with a shelf.", size=42, color=VIOLET)

        # 2x2 grid of estimator "boxes" sitting on literal shelf lines
        margin = 32
        grid_gap = 10
        card_w = (WIDTH - margin * 2 - grid_gap) / 2
        card_h = 52
        start_y = 122

        # Same four estimators as the cover and the slide-6 equivalence map:
        # the shelf we show is exactly the shelf LP-DiD provably nests.
        # "BJS-Style" keeps the single-cohort equivalence scope visible.
        boxes = [
            ("Classic 2x2 DiD", "Where you started.\nOne clean comparison."),
            ("Callaway-Sant'Anna", "Group-time ATTs.\nIts own aggregation layer."),
            ("Stacked DiD (Cengiz et al.)", "Dataset duplication.\nIts own weighting scheme."),
            ("BJS-Style Imputation", "Borusyak-Jaravel-Spiess.\nIts own two-step machinery."),
        ]

        for idx, (title, desc) in enumerate(boxes):
            row = idx // 2
            col = idx % 2
            cx = margin + col * (card_w + grid_gap)
            cy = start_y + row * (card_h + 22)

            self.set_fill_color(*WHITE)
            self.set_draw_color(*VIOLET)
            self.set_line_width(0.6)
            self.rect(cx, cy, card_w, card_h, "DF")

            self.set_xy(cx + 10, cy + 8)
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(*NAVY)
            self.cell(card_w - 20, 10, title)

            for j, line in enumerate(desc.split("\n")):
                self.set_xy(cx + 10, cy + 24 + j * 11)
                self.set_font("Helvetica", "", 11)
                self.set_text_color(*GRAY)
                self.cell(card_w - 20, 10, line)

            # Shelf line under each row of boxes
            shelf_y = cy + card_h + 6
            self.set_draw_color(*NAVY)
            self.set_line_width(1.4)
            self.line(margin - 6, shelf_y, WIDTH - margin + 6, shelf_y)

        cap_y = start_y + 2 * (card_h + 22) + 12
        self.centered_text(
            cap_y,
            "Three fixes for TWFE's negative weights - plus the textbook case they outgrew.",
            size=14,
            bold=False,
            italic=True,
            color=GRAY,
        )
        self.centered_text(
            cap_y + 16,
            "Same causal question. Four mental models.",
            size=16,
            bold=True,
            color=NAVY,
        )

        self.add_footer()

    def slide_03_real_world(self):
        """Slide 3: four scenarios of tool-choice anxiety."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(3)

        self.centered_text(40, "Same question.", size=38)

        # Second line with "Four different machines." accented
        self.set_font("Helvetica", "B", 38)
        text_b = "Four different "
        text_c = "machines."
        w_b = self.get_string_width(text_b)
        w_c = self.get_string_width(text_c)
        start_x = (WIDTH - w_b - w_c) / 2
        self.set_xy(start_x, 70)
        self.set_text_color(*NAVY)
        self.cell(w_b, 20, text_b)
        self.set_text_color(*VIOLET)
        self.cell(w_c, 20, text_c)

        scenarios = [
            (
                "Staggered State Policy",
                "Minimum wage rolls out state by state. CS? Stacked? Which aggregation?",
            ),
            (
                "Phased Product Rollout",
                "The feature ships to markets in waves. Event study? Which estimator?",
            ),
            (
                "Programs That Switch Off",
                "The retention treatment ends. Absorbing-only estimators don't apply.",
            ),
            (
                "Survey-Weighted Panels",
                "CPS-style pweights, strata, PSUs. Which estimator takes a design?",
            ),
        ]
        self._card_stack(scenarios, start_y=130, accent=AMBER)

        self.add_footer()

    def slide_04_introducing(self):
        """Slide 4: LP-DiD intro -- 'It's just a regression.'"""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(4)

        self.centered_text(40, "LP-DiD.", size=44)
        self.centered_text(
            78, "It's just a regression.", size=22, bold=False, italic=True, color=VIOLET
        )

        self.centered_text(
            114, "A local projection per horizon: long difference,", size=14, bold=False, color=GRAY
        )
        self.centered_text(
            128, "time fixed effects, clean controls. Plain OLS.", size=14, bold=False, color=GRAY
        )

        items = [
            (
                "One long difference per horizon.",
                "OLS on clean controls. You can read every comparison it makes.",
            ),
            (
                "Weights provably non-negative.",
                "Clean-control baseline result (Eqs. 9-10). With covariates, prefer RA.",
            ),
            (
                "The estimand is a dial.",
                "Variance-weighted for precision, or equal-weighted - your call.",
            ),
        ]
        end_y = self._card_stack(items, start_y=164)

        self.set_xy(0, end_y + 8)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(*LIGHT_GRAY)
        self.cell(WIDTH, 8, "Local projections (Jordà 2005), pointed at DiD.", align="C")

        self.add_footer()

    def slide_05_mechanism(self):
        """Slide 5: clean-control panel schematic at horizon h=+2."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(5)

        self.centered_text(40, "Clean controls,", size=36)
        self.centered_text(64, "horizon by horizon.", size=36, color=VIOLET)

        plot_path, ppw, pph = self._render_clean_control_grid()
        plot_w = WIDTH * 0.80
        plot_aspect = pph / ppw
        plot_h = plot_w * plot_aspect
        plot_x = (WIDTH - plot_w) / 2
        plot_y = 104
        self.image(plot_path, plot_x, plot_y, plot_w)

        cap_y = plot_y + plot_h + 10
        self.centered_text(
            cap_y,
            "Each horizon is one regression you could run by hand.",
            size=16,
            bold=True,
            color=VIOLET,
        )
        self.centered_text(
            cap_y + 16,
            "Already-treated units never serve as controls.",
            size=13,
            bold=False,
            italic=True,
            color=GRAY,
        )

        self.add_footer()

    def slide_06_math(self):
        """Slide 6: the unification slide -- equation + equivalence map."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(6)

        self.centered_text(38, "The zoo was one regression", size=32)
        self.centered_text(62, "all along.", size=32, color=VIOLET)

        eq_path, epw, eph = self._render_weights_equation()
        eq_y = 88
        eq_h = self._place_equation_centered(eq_path, epw, eph, eq_y, max_w=225)

        # Equivalence map -- each row: LPDiD setting == estimator (source)
        rows = [
            ("reweight=True", "==  Callaway-Sant'Anna", "exact  (Sec. 3.7)"),
            ("default (variance-weighted)", "==  Cengiz et al. stacked", "exact  (Sec. 3.7)"),
            ("h=0, single cohort", "==  classic 2x2 DiD", "exact  (Sec. 2.2)"),
            ("pmd='max'", "->  BJS imputation", "exact single-cohort  (fn. 10-11)"),
        ]

        margin = 34
        box_w = WIDTH - margin * 2
        row_h = 24
        gap = 4
        start_y = eq_y + eq_h + 14
        col1_w = 82

        for i, (setting, target, source) in enumerate(rows):
            by = start_y + i * (row_h + gap)
            self.set_fill_color(*WHITE)
            self.set_draw_color(220, 220, 220)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, row_h, "DF")
            self.set_fill_color(*CYAN)
            self.rect(margin, by, 4, row_h, "F")

            self.set_xy(margin + 12, by + 7)
            self.set_font("Courier", "B", 12)
            self.set_text_color(*VIOLET_DARK)
            self.cell(col1_w, 10, setting)

            self.set_xy(margin + 12 + col1_w, by + 7)
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(*NAVY)
            self.cell(84, 10, target)

            self.set_xy(margin + 12 + col1_w + 84, by + 7)
            self.set_font("Helvetica", "I", 10)
            self.set_text_color(*GRAY)
            self.cell(box_w - col1_w - 84 - 24, 10, source, align="R")

        cap_y = start_y + len(rows) * (row_h + gap) + 8
        self.centered_text(
            cap_y,
            "Numerical equivalences proven in the paper - not analogies.",
            size=13,
            bold=False,
            italic=True,
            color=GRAY,
        )

        self.add_footer()

    def slide_07_output(self):
        """Slide 7: native event study."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(7)

        self.centered_text(40, "Pre-trends are just", size=36)
        self.centered_text(64, "negative horizons.", size=36, color=VIOLET)

        plot_path, ppw, pph = self._render_event_study()
        plot_w = WIDTH * 0.86
        plot_aspect = pph / ppw
        plot_h = plot_w * plot_aspect
        plot_x = (WIDTH - plot_w) / 2
        plot_y = 112
        self.image(plot_path, plot_x, plot_y, plot_w)

        cap_y = plot_y + plot_h + 12
        self.centered_text(
            cap_y,
            "The event study is native - not a post-hoc aggregation.",
            size=16,
            bold=True,
            color=VIOLET,
        )

        self.add_footer()

    def slide_08_code(self):
        """Slide 8: code example -- fit() + the reweight dial."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(8)

        self.centered_text(38, "The Code.", size=46)
        self.centered_text(
            78,
            "Same sklearn-like API as every diff-diff estimator.",
            size=14,
            bold=False,
            color=GRAY,
        )

        margin = 22
        code_y = 98

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
            font_size=11,
            line_height=9,
        )

        sub_y = code_y + code_h + 10
        self.centered_text(
            sub_y,
            "Callaway-Sant'Anna is literally a keyword argument.",
            size=13,
            bold=False,
            color=GRAY,
        )

        self.add_footer()

    def slide_09_production_ready(self):
        """Slide 9: production-ready feature grid (2x3)."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(9)

        self.centered_text(40, "Production-ready.", size=48, color=CYAN)

        margin = 26
        grid_gap = 8
        card_w = (WIDTH - margin * 2 - grid_gap) / 2
        card_h = 56
        start_y = 90

        features = [
            ("Cluster-Robust SEs", "Unit-level by default,\nmatches Stata lpdid"),
            ("Covariates, Two Ways", "RA path (BJS-style) or\ndirect with guardrails"),
            ("PMD Baselines", "Premean differencing +\npooled pre/post ATTs"),
            ("Non-Absorbing Treatment", "First-entry & effect-\nstabilization (Eq. 12/13)"),
            ("Survey Designs", "pweights, strata, PSU, FPC;\nBinder TSL (default path)"),
            ("Composition Control", "no_composition fixes the\npost-window sample"),
        ]

        for idx, (title, desc) in enumerate(features):
            row = idx // 2
            col = idx % 2
            cx = margin + col * (card_w + grid_gap)
            cy = start_y + row * (card_h + grid_gap)

            self.set_fill_color(*WHITE)
            self.set_draw_color(*VIOLET)
            self.set_line_width(0.6)
            self.rect(cx, cy, card_w, card_h, "DF")

            self.set_xy(cx + 10, cy + 8)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*CYAN)
            self.cell(card_w - 20, 10, title)

            for j, line in enumerate(desc.split("\n")):
                self.set_xy(cx + 10, cy + 24 + j * 12)
                self.set_font("Helvetica", "", 11)
                self.set_text_color(*GRAY)
                self.cell(card_w - 20, 10, line)

        comp_y = start_y + 3 * (card_h + grid_gap) + 4
        self.set_xy(0, comp_y)
        self.set_font("Helvetica", "I", 10)
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

        self.centered_text(40, "Validated.", size=42, color=VIOLET)
        self.centered_text(
            82,
            "Against the authors' own tooling - and against our own shelf.",
            size=13,
            bold=False,
            italic=True,
            color=GRAY,
        )

        items = [
            (
                "Authors' R recipes (danielegirardi/lpdid)",
                "Event-study and pooled estimands match to ~1e-13.",
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
        self._card_stack(items, start_y=105)

        self.add_footer()

    def slide_11_cta(self):
        """Slide 11: CTA -- pip install + GitHub."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(11)

        self.centered_text(58, "Now in diff-diff.", size=24, bold=False, italic=True, color=GRAY)
        self.centered_text(88, "LP-DiD.", size=52, color=VIOLET)

        badge_w = 230
        badge_h = 42
        badge_x = (WIDTH - badge_w) / 2
        badge_y = 158
        self.set_fill_color(*VIOLET)
        self.rect(badge_x, badge_y, badge_w, badge_h, "F")

        self.set_xy(badge_x, badge_y + 12)
        self.set_font("Courier", "B", 16)
        self.set_text_color(*WHITE)
        self.cell(badge_w, 16, "$ pip install --upgrade diff-diff", align="C")

        self.centered_text(222, "github.com/igerber/diff-diff", size=18, color=VIOLET)

        self.draw_split_logo(258, size=28)

        self.centered_text(
            284, "Difference-in-Differences for Python", size=14, bold=False, color=GRAY
        )

        self.add_footer()


def main():
    pdf = LPDiDCarouselPDF()
    try:
        pdf.slide_01_cover()
        pdf.slide_02_shelf()
        pdf.slide_03_real_world()
        pdf.slide_04_introducing()
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
