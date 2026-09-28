#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for SpilloverDiD (Butts 2021) launch.

Mirrors the architecture of ``generate_had_carousel.py`` but introduces a
fresh earth-tone palette (slate-blue / warm terra / muted gold / light cream)
that signals the spatial / geographic identity of the spillover estimator
while preserving the diff-diff family marks (split-color logo, footer
wordmark, gradient background).

Notable deviations from HAD:

- Magazine sidebar renders on every slide (HAD defines the helper but never
  calls it). The tick advances slide-by-slide from top -> bottom.
- Footer reads ``diff-diff v3.4.1`` (lifted to ``VERSION_LABEL`` constant).
- Gradient background interpolates all three RGB channels for cream -> white.
- HAD's connector-graphic helper is dropped (replaced by the sidebar).

Run with::

    python carousel/generate_spillover_carousel.py

Produces ``carousel/diff-diff-spillover-carousel.pdf``.
"""

import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from fpdf import FPDF  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

# Computer Modern for math
plt.rcParams["mathtext.fontset"] = "cm"

# Page dimensions (4:5 portrait — same as HAD)
WIDTH = 270  # mm
HEIGHT = 337.5  # mm

# Version label — single source of truth for the footer wordmark
VERSION_LABEL = "v3.4.1"

# -------------------------------------------------------------------------
# Earth-tone palette
# -------------------------------------------------------------------------
# Primary palette (RGB)
SLATE_BLUE = (71, 85, 105)  # #475569  primary accent
SLATE_BLUE_DARK = (51, 65, 85)  # #334155
SLATE_BLUE_LIGHT = (148, 163, 184)  # #94a3b8
TERRA = (194, 65, 12)  # #c2410c  spillover / "leak" accent
GOLD = (217, 119, 6)  # #d97706  secondary accent
CREAM = (254, 243, 199)  # #fef3c7  light cream highlight / gradient start
TERRA_LIGHT = (251, 146, 60)  # #fb923c  ring-1 control tint
GOLD_LIGHT = (253, 186, 116)  # #fdba74  ring-2 control tint

# Text + structural (kept from HAD for legibility)
NAVY = (15, 23, 42)  # #0f172a  primary text
GRAY = (100, 116, 139)  # #64748b  secondary text
LIGHT_GRAY = (148, 163, 184)  # #94a3b8  fine print
WHITE = (255, 255, 255)
DARK_SLATE = (30, 41, 59)  # #1e293b  code block bg
CODE_BG = (15, 23, 42)  # #0f172a  deeper code bg
AMBER_CODE = (252, 211, 77)  # #fcd34d  code string literals
SLATE_CODE = (148, 163, 184)  # #94a3b8  code keyword tone

# Hex equivalents for matplotlib
SLATE_BLUE_HEX = "#475569"
SLATE_BLUE_DARK_HEX = "#334155"
SLATE_BLUE_LIGHT_HEX = "#94a3b8"
TERRA_HEX = "#c2410c"
TERRA_LIGHT_HEX = "#fb923c"
GOLD_HEX = "#d97706"
GOLD_LIGHT_HEX = "#fdba74"
CREAM_HEX = "#fef3c7"
NAVY_HEX = "#0f172a"
GRAY_HEX = "#64748b"
LIGHT_GRAY_HEX = "#94a3b8"


class SpilloverCarouselPDF(FPDF):
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
    # Magazine vertical sidebar — drawn on every slide. The tick
    # advances from near-top (slide 1) to near-bottom (slide 11),
    # signalling progress through the deck. Bar is SLATE_BLUE; tick is
    # TERRA so the accent reads as a deliberate progress marker rather
    # than a stray line.
    # -----------------------------------------------------------------

    def _draw_vertical_sidebar(self, slide_number, total=11):
        """Draw the magazine vertical accent on the left edge with TERRA tick."""
        bar_x = 14  # mm from left edge
        bar_y_top = 45  # below the top margin
        bar_y_bottom = 275  # above the wordmark area
        self.set_draw_color(*SLATE_BLUE)
        self.set_line_width(0.6)
        self.line(bar_x, bar_y_top, bar_x, bar_y_bottom)

        # Progress tick — TERRA accent so it reads against the slate bar.
        if total > 1:
            ratio = (slide_number - 1) / (total - 1)
        else:
            ratio = 0.0
        tick_y = bar_y_top + ratio * (bar_y_bottom - bar_y_top)
        self.set_draw_color(*TERRA)
        self.set_line_width(0.9)
        self.line(bar_x - 4, tick_y, bar_x + 7, tick_y)

    # -----------------------------------------------------------------
    # Background + footer
    # -----------------------------------------------------------------

    def light_gradient_background(self):
        """Cream #fef3c7 fading to white. Interpolates all 3 RGB channels."""
        steps = 50
        r0, g0, b0 = 254, 243, 199  # cream
        r1, g1, b1 = 255, 255, 255  # white
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
        self.set_text_color(*SLATE_BLUE)
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
        """Split-color diff-diff logo with SLATE_BLUE middle dash."""
        self.set_xy(0, y)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="R")
        self.set_text_color(*SLATE_BLUE)
        self.cell(10, 10, "-", align="C")
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="L")

    # -----------------------------------------------------------------
    # Equation rendering (matplotlib mathtext -> PNG -> fpdf image)
    # -----------------------------------------------------------------

    def _render_equations(self, latex_lines, fontsize=26, color=NAVY_HEX):
        n = len(latex_lines)
        fig_h = max(0.7, 0.55 * n + 0.15)
        fig = plt.figure(figsize=(10, fig_h))
        for i, line in enumerate(latex_lines):
            y_frac = 1.0 - (2 * i + 1) / (2 * n)
            fig.text(0.5, y_frac, line, fontsize=fontsize, ha="center", va="center", color=color)
        fig.patch.set_alpha(0)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.06, transparent=True)
        plt.close(fig)
        with PILImage.open(path) as img:
            pw, ph = img.size
        self._temp_files.append(path)
        return path, pw, ph

    def _place_equation_centered(self, path, pw, ph, y, max_w=200):
        aspect = ph / pw
        display_w = min(max_w, WIDTH * 0.75)
        display_h = display_w * aspect
        eq_x = (WIDTH - display_w) / 2
        self.image(path, eq_x, y, display_w)
        return display_h

    # -----------------------------------------------------------------
    # Slide-6 bias equation with annotation arrow pointing at the
    # tau_spill(0) term specifically (NOT centered ambiguously under the
    # whole equation, which would visually land closer to tau_total).
    # -----------------------------------------------------------------

    def _render_bias_equation(self):
        fig = plt.figure(figsize=(10, 3.2))
        fig.patch.set_alpha(0)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Equation centered horizontally. Hairspace-aware spacing for legibility.
        ax.text(
            0.5,
            0.72,
            r"$\beta_{\mathrm{DiD}}\;\approx\;\tau_{\mathrm{total}}\;-\;\tau_{\mathrm{spill}}(0)$",
            fontsize=30,
            ha="center",
            va="center",
            color=NAVY_HEX,
        )

        # Arrow + label pointing UP at tau_spill(0), which sits to the
        # right of center in the rendered equation. xy is tail-of-arrow
        # at the bottom edge of the term; xytext is the label position.
        ax.annotate(
            "hidden bias term",
            xy=(0.668, 0.62),
            xytext=(0.668, 0.18),
            fontsize=14,
            color=TERRA_HEX,
            fontweight="bold",
            ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="->", color=TERRA_HEX, lw=1.5, shrinkA=2, shrinkB=4),
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
    # Slide-2 visual — stylized map: treated zone bleeding into the
    # spillover ring, with a dashed d_bar boundary. Clean-control region
    # beyond d_bar shown as faint slate dots.
    # -----------------------------------------------------------------

    def _render_bias_bleed_schematic(self):
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        n = 240
        x = np.linspace(-1.05, 1.05, n)
        y = np.linspace(-1.05, 1.05, n)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X ** 2 + Y ** 2)
        d_bar = 0.6
        # cos^2 profile: exposure = 1 at r=0, smoothly hits 0 at r=d_bar
        intensity = np.where(
            r < d_bar,
            np.cos(np.pi * r / (2 * d_bar)) ** 2,
            0.0,
        )

        # Custom colormap: transparent at intensity 0 -> deep terra at 1
        cmap = LinearSegmentedColormap.from_list(
            "bleed",
            [(1.0, 1.0, 1.0, 0.0), (1.0, 0.92, 0.78, 0.6), TERRA_LIGHT_HEX, TERRA_HEX],
            N=200,
        )
        ax.imshow(
            intensity,
            extent=(-1.05, 1.05, -1.05, 1.05),
            origin="lower",
            cmap=cmap,
            vmin=0,
            vmax=1.0,
            interpolation="bilinear",
            zorder=1,
        )

        # Clean-control dots beyond d_bar
        rng = np.random.default_rng(7)
        n_far = 70
        far_pts = []
        while len(far_pts) < n_far:
            px = rng.uniform(-1.0, 1.0)
            py = rng.uniform(-1.0, 1.0)
            if np.sqrt(px ** 2 + py ** 2) > d_bar + 0.06:
                far_pts.append((px, py))
        far_pts = np.array(far_pts)
        ax.scatter(
            far_pts[:, 0],
            far_pts[:, 1],
            s=18,
            color=LIGHT_GRAY_HEX,
            alpha=0.55,
            edgecolors="none",
            zorder=2,
        )

        # Treated centroid marker
        ax.scatter(
            [0],
            [0],
            s=320,
            marker="s",
            color=TERRA_HEX,
            edgecolors=NAVY_HEX,
            linewidths=1.4,
            zorder=5,
        )
        ax.text(
            0.0,
            -0.10,
            "treated",
            ha="center",
            va="top",
            fontsize=11,
            color=NAVY_HEX,
            fontweight="bold",
            zorder=6,
        )

        # d_bar boundary
        circle = patches.Circle(
            (0, 0),
            d_bar,
            fill=False,
            linestyle=(0, (6, 4)),
            edgecolor=SLATE_BLUE_HEX,
            linewidth=1.6,
            zorder=4,
        )
        ax.add_patch(circle)
        ax.annotate(
            r"$d_{\mathrm{bar}}$",
            xy=(d_bar * 0.71, d_bar * 0.71),
            xytext=(0.78, 0.78),
            fontsize=12,
            color=SLATE_BLUE_HEX,
            arrowprops=dict(arrowstyle="->", color=SLATE_BLUE_HEX, lw=0.9),
            zorder=6,
        )

        # Annotation: spillover bleeds outward
        ax.annotate(
            "spillover bleeds outward",
            xy=(0.32, 0.12),
            xytext=(0.78, 0.05),
            fontsize=10,
            color=TERRA_HEX,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=TERRA_HEX, lw=0.9),
            zorder=6,
        )

        # Annotation: clean control
        ax.text(
            -0.85,
            -0.85,
            "clean control\n($d > d_{\\mathrm{bar}}$)",
            ha="left",
            va="bottom",
            fontsize=10,
            color=GRAY_HEX,
            zorder=6,
        )

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
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
    # Slide-5 anchor map — synthetic spatial scatter with several treated
    # cores, each surrounded by concentric ring boundaries at 50 / 100 /
    # 200 km. Control units colored by their ring assignment (distance to
    # nearest treated core). Clean controls beyond 200 km in light gray.
    # -----------------------------------------------------------------

    def _render_anchor_rings_map(self):
        fig, ax = plt.subplots(figsize=(10, 7.0))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Treated cores (km)
        treated = np.array(
            [
                (120, 110),
                (340, 150),
                (210, 360),
                (420, 410),
                (110, 410),
            ],
            dtype=float,
        )
        rings_km = [50.0, 100.0, 200.0]

        rng = np.random.default_rng(11)
        n_control = 110
        ctl_x = rng.uniform(0, 500, n_control)
        ctl_y = rng.uniform(0, 500, n_control)
        ctl = np.column_stack([ctl_x, ctl_y])

        # Distance to nearest treated, then ring assignment
        diffs = ctl[:, None, :] - treated[None, :, :]
        dists = np.sqrt(np.einsum("ijk,ijk->ij", diffs, diffs))
        nearest_d = dists.min(axis=1)

        ring1_mask = nearest_d < rings_km[0]
        ring2_mask = (nearest_d >= rings_km[0]) & (nearest_d < rings_km[1])
        ring3_mask = (nearest_d >= rings_km[1]) & (nearest_d < rings_km[2])
        far_mask = nearest_d >= rings_km[2]

        # Dashed ring boundaries around every treated core
        for tx, ty in treated:
            for r_km in rings_km:
                circle = patches.Circle(
                    (tx, ty),
                    r_km,
                    fill=False,
                    linestyle=(0, (3, 4)),
                    edgecolor=SLATE_BLUE_HEX,
                    linewidth=0.9,
                    alpha=0.55,
                    zorder=2,
                )
                ax.add_patch(circle)

        # Controls by ring membership
        ax.scatter(
            ctl[far_mask, 0],
            ctl[far_mask, 1],
            s=22,
            color=LIGHT_GRAY_HEX,
            alpha=0.75,
            edgecolors="none",
            zorder=3,
            label="d > 200 km (clean)",
        )
        ax.scatter(
            ctl[ring3_mask, 0],
            ctl[ring3_mask, 1],
            s=30,
            color=SLATE_BLUE_LIGHT_HEX,
            alpha=0.95,
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
            label="ring 3 (100-200 km)",
        )
        ax.scatter(
            ctl[ring2_mask, 0],
            ctl[ring2_mask, 1],
            s=34,
            color=GOLD_LIGHT_HEX,
            alpha=0.95,
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
            label="ring 2 (50-100 km)",
        )
        ax.scatter(
            ctl[ring1_mask, 0],
            ctl[ring1_mask, 1],
            s=38,
            color=TERRA_LIGHT_HEX,
            alpha=0.95,
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
            label="ring 1 (0-50 km)",
        )

        # Treated cores on top
        ax.scatter(
            treated[:, 0],
            treated[:, 1],
            s=160,
            marker="s",
            color=TERRA_HEX,
            edgecolors=NAVY_HEX,
            linewidths=1.4,
            zorder=5,
            label="treated unit",
        )

        ax.set_xlim(-25, 525)
        ax.set_ylim(-25, 525)
        ax.set_aspect("equal")
        ax.set_xlabel("km", fontsize=10, color=GRAY_HEX)
        ax.set_ylabel("km", fontsize=10, color=GRAY_HEX)
        ax.tick_params(colors=GRAY_HEX, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(LIGHT_GRAY_HEX)

        leg = ax.legend(
            loc="upper right",
            fontsize=8,
            frameon=True,
            framealpha=0.95,
            labelcolor=NAVY_HEX,
            handletextpad=0.4,
        )
        leg.get_frame().set_edgecolor(LIGHT_GRAY_HEX)
        leg.get_frame().set_linewidth(0.5)

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
    # Slide-7 visual — per-ring delta_j bar chart with TERRA decay line
    # overlay and CI whiskers.
    # -----------------------------------------------------------------

    def _render_ring_decay_bars(self):
        fig, ax = plt.subplots(figsize=(10, 5.4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ring_labels = ["[0, 50)", "[50, 100)", "[100, 200]"]
        delta = np.array([0.045, 0.018, 0.003])
        ci_half = np.array([0.012, 0.009, 0.005])

        x_pos = np.arange(len(ring_labels))
        bar_w = 0.55

        ax.bar(
            x_pos,
            delta,
            width=bar_w,
            color=SLATE_BLUE_HEX,
            edgecolor=SLATE_BLUE_DARK_HEX,
            linewidth=0.8,
            zorder=3,
        )

        ax.errorbar(
            x_pos,
            delta,
            yerr=ci_half,
            fmt="none",
            ecolor=GRAY_HEX,
            elinewidth=1.4,
            capsize=8,
            capthick=1.4,
            zorder=4,
        )

        ax.plot(
            x_pos,
            delta,
            color=TERRA_HEX,
            linewidth=2.6,
            marker="o",
            markersize=10,
            markerfacecolor=TERRA_HEX,
            markeredgecolor=NAVY_HEX,
            markeredgewidth=1.0,
            zorder=5,
            label="decay trend",
        )

        ax.axhline(0, color=LIGHT_GRAY_HEX, linewidth=0.8, zorder=1)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(ring_labels, fontsize=11, color=NAVY_HEX)
        ax.set_xlabel("Ring (km)", fontsize=12, color=NAVY_HEX)
        ax.set_ylabel(r"$\delta_j$ (spillover effect)", fontsize=12, color=NAVY_HEX)
        ax.tick_params(axis="y", colors=GRAY_HEX, labelsize=9)
        ax.tick_params(axis="x", colors=NAVY_HEX, labelsize=11)
        ax.set_ylim(-0.005, 0.075)
        for spine in ax.spines.values():
            spine.set_color(LIGHT_GRAY_HEX)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for xi, di in zip(x_pos, delta):
            ax.text(
                float(xi),
                float(di + max(ci_half) + 0.005),
                f"{di:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=NAVY_HEX,
                fontweight="bold",
            )

        ax.legend(loc="upper right", fontsize=10, frameon=False, labelcolor=NAVY_HEX)

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

    # =================================================================
    # SLIDES
    # =================================================================

    def slide_01_cover(self):
        """Slide 1: Manifesto hook."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(1, total=11)

        self.draw_split_logo(40, size=42)

        # Hero typography
        self.centered_text(118, "There's a clean line between", size=34)
        self.centered_text(150, "treated and control.", size=34)

        # Punchline
        self.centered_text(212, "Until there isn't.", size=54, color=SLATE_BLUE)

        # Byline
        self.set_xy(0, HEIGHT - 70)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*SLATE_BLUE)
        self.cell(WIDTH, 8, "Spillover-Aware DiD.", align="C")
        self.set_xy(0, HEIGHT - 58)
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 8, "Now in diff-diff.", align="C")
        self.set_xy(0, HEIGHT - 46)
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 8, "Butts (2021).", align="C")
        self.set_xy(0, HEIGHT - 35)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*LIGHT_GRAY)
        self.cell(WIDTH, 8, "arXiv:2105.03737", align="C")

        self.add_footer()

    def slide_02_problem(self):
        """Slide 2: Bias schematic — treated zone bleeding into control ring."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(2, total=11)

        self.centered_text(40, "Some treatment leaks", size=34)
        self.centered_text(72, "into your controls.", size=42, color=SLATE_BLUE)

        plot_path, ppw, pph = self._render_bias_bleed_schematic()
        plot_w = WIDTH * 0.66
        plot_aspect = pph / ppw
        plot_h = plot_w * plot_aspect
        plot_x = (WIDTH - plot_w) / 2
        plot_y = 130
        self.image(plot_path, plot_x, plot_y, plot_w)

        cap_y = plot_y + plot_h + 10
        self.centered_text(
            cap_y, "The 'control' isn't. (Partly.)", size=14, bold=False, italic=True, color=GRAY
        )

        self.add_footer()

    def slide_03_real_world(self):
        """Slide 3: 4 marketing-heavy real-world cards."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(3, total=11)

        # Headline line 1
        self.centered_text(40, "Sometimes treatment", size=38)

        # Headline line 2 with "leaks" split-colored TERRA
        self.set_font("Helvetica", "B", 38)
        text_b = "leaks "
        text_c = "across borders."
        w_b = self.get_string_width(text_b)
        w_c = self.get_string_width(text_c)
        total_w = w_b + w_c
        start_x = (WIDTH - total_w) / 2
        self.set_xy(start_x, 70)
        self.set_text_color(*TERRA)
        self.cell(w_b, 20, text_b)
        self.set_text_color(*NAVY)
        self.cell(w_c, 20, text_c)

        # 4 cards stacked, TERRA left bar
        margin = 30
        box_w = WIDTH - margin * 2
        box_h = 40
        gap = 5
        start_y = 130
        bar_w = 4

        scenarios = [
            (
                "Place-Based Economic Policy",
                "Enterprise zones lift local investment - and bleed into adjacent neighborhoods.",
            ),
            (
                "Out-of-Home Media Buys",
                "Billboards reach commuters who live outside the placement DMA.",
            ),
            (
                "Geo-Targeted Digital Campaigns",
                "Meta / Google geo-fences leak via household-IP and travel boundaries.",
            ),
            (
                "Retail Footprint Tests",
                "A new store lifts sales in adjacent trade areas, not just at the location.",
            ),
        ]

        for i, (title, desc) in enumerate(scenarios):
            by = start_y + i * (box_h + gap)
            self.set_fill_color(*WHITE)
            self.set_draw_color(220, 220, 220)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, box_h, "DF")
            self.set_fill_color(*TERRA)
            self.rect(margin, by, bar_w, box_h, "F")

            self.set_xy(margin + bar_w + 12, by + 8)
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(*NAVY)
            self.cell(box_w - bar_w - 24, 10, title)

            self.set_xy(margin + bar_w + 12, by + 25)
            self.set_font("Helvetica", "", 13)
            self.set_text_color(*GRAY)
            self.cell(box_w - bar_w - 24, 10, desc)

        self.add_footer()

    def slide_04_introducing(self):
        """Slide 4: SpilloverDiD intro — 3 capability cards."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(4, total=11)

        self.centered_text(40, "The SpilloverDiD estimator.", size=38)
        self.centered_text(
            82, "For partial-contamination DiD.", size=18, bold=False, italic=True, color=SLATE_BLUE
        )

        self.centered_text(
            118, "Recovers both direct effects on treated", size=14, bold=False, color=GRAY
        )
        self.centered_text(
            132, "and per-ring spillover on near-controls.", size=14, bold=False, color=GRAY
        )

        margin = 30
        box_w = WIDTH - margin * 2
        box_h = 40
        gap = 5
        start_y = 168
        bar_w = 4

        items = [
            (
                "Two effects, one regression.",
                "Direct tau_total on treated AND per-ring delta_j on near-controls.",
            ),
            (
                "Far-away controls anchor identification.",
                "Beyond d_bar, controls are uncontaminated.",
            ),
            (
                "Spatial-HAC inference, out of the box.",
                "Conley (1999) panel-block standard errors.",
            ),
        ]

        for i, (title, desc) in enumerate(items):
            by = start_y + i * (box_h + gap)
            self.set_fill_color(*WHITE)
            self.set_draw_color(220, 220, 220)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, box_h, "DF")
            self.set_fill_color(*SLATE_BLUE)
            self.rect(margin, by, bar_w, box_h, "F")

            self.set_xy(margin + bar_w + 12, by + 8)
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(*NAVY)
            self.cell(box_w - bar_w - 24, 10, title)

            self.set_xy(margin + bar_w + 12, by + 24)
            self.set_font("Helvetica", "", 12)
            self.set_text_color(*GRAY)
            self.cell(box_w - bar_w - 24, 10, desc)

        footnote_y = start_y + len(items) * (box_h + gap) + 8
        self.set_xy(0, footnote_y)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(*LIGHT_GRAY)
        self.cell(WIDTH, 8, "Built on two-stage Gardner (2022) DiD.", align="C")

        self.add_footer()

    def slide_05_anchor(self):
        """Slide 5: ANCHOR — map of treated units with concentric rings."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(5, total=11)

        self.centered_text(40, "Identify the spillover", size=36)
        self.centered_text(64, "ring by ring.", size=36, color=SLATE_BLUE)

        plot_path, ppw, pph = self._render_anchor_rings_map()
        plot_w = WIDTH * 0.72
        plot_aspect = pph / ppw
        plot_h = plot_w * plot_aspect
        plot_x = (WIDTH - plot_w) / 2
        plot_y = 108
        self.image(plot_path, plot_x, plot_y, plot_w)

        cap_y = plot_y + plot_h + 10
        self.centered_text(
            cap_y,
            "Each ring estimates its own spillover effect.",
            size=16,
            bold=True,
            color=SLATE_BLUE,
        )

        self.add_footer()

    def slide_06_estimand(self):
        """Slide 6: Butts Proposition 2.1 bias decomposition."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(6, total=11)

        self.centered_text(70, "The hidden bias.", size=42)
        self.centered_text(
            108, "(Butts 2021, Proposition 2.1)", size=14, bold=False, italic=True, color=GRAY
        )

        # Equation + annotation arrow rendered together so the arrow points
        # specifically at tau_spill(0) (not ambiguously at the equation center).
        eq_path, epw, eph = self._render_bias_equation()
        eq_y = 140
        eq_h = self._place_equation_centered(eq_path, epw, eph, eq_y, max_w=230)

        # Plain-English gloss below
        gloss_y = eq_y + eq_h + 12
        self.centered_text(
            gloss_y,
            "Standard DiD recovers tau_total only when spillover is zero.",
            size=14,
            bold=False,
            color=NAVY,
        )
        self.centered_text(
            gloss_y + 16,
            "SpilloverDiD identifies tau_total and delta_j separately.",
            size=14,
            bold=False,
            color=NAVY,
        )

        self.add_footer()

    def slide_07_output(self):
        """Slide 7: per-ring delta_j decay bar chart."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(7, total=11)

        self.centered_text(40, "Spillover decays", size=36)
        self.centered_text(64, "with distance.", size=36, color=SLATE_BLUE)

        plot_path, ppw, pph = self._render_ring_decay_bars()
        plot_w = WIDTH * 0.86
        plot_aspect = pph / ppw
        plot_h = plot_w * plot_aspect
        plot_x = (WIDTH - plot_w) / 2
        plot_y = 118
        self.image(plot_path, plot_x, plot_y, plot_w)

        cap_y = plot_y + plot_h + 14
        self.centered_text(
            cap_y,
            "Per-ring delta_j attenuates toward zero.",
            size=17,
            bold=True,
            color=SLATE_BLUE,
        )

        self.add_footer()

    def slide_08_code(self):
        """Slide 8: code example — basic call + per-ring output."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(8, total=11)

        self.centered_text(38, "The Code.", size=46)
        self.centered_text(
            78,
            "Same sklearn-like API as every diff-diff estimator.",
            size=14,
            bold=False,
            color=GRAY,
        )

        margin = 22
        code_y = 100

        token_lines = [
            [
                ("from", SLATE_CODE),
                (" diff_diff ", WHITE),
                ("import", SLATE_CODE),
                (" SpilloverDiD", WHITE),
            ],
            [],
            [
                ("result", WHITE),
                (" = ", SLATE_CODE),
                ("SpilloverDiD", AMBER_CODE),
                ("(", WHITE),
            ],
            [
                ("    ", WHITE),
                ("rings", WHITE),
                ("=[", SLATE_CODE),
                ("0, 50, 100, 200", AMBER_CODE),
                ("],", SLATE_CODE),
            ],
            [
                ("    ", WHITE),
                ("conley_coords", WHITE),
                ("=", SLATE_CODE),
                ("('lat', 'lon')", AMBER_CODE),
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
                ("'sales'", AMBER_CODE),
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
                ("'campaign'", AMBER_CODE),
                (",", SLATE_CODE),
            ],
            [(")", WHITE)],
            [],
            [
                ("print(", WHITE),
                ("result.att", WHITE),
                (")", WHITE),
                ("              # tau_total = 0.124", LIGHT_GRAY),
            ],
            [
                ("print(", WHITE),
                ("result.spillover_effects", WHITE),
                (")", WHITE),
                ("  # per-ring delta_j", LIGHT_GRAY),
            ],
            [
                ("#             coef    se     p_value", LIGHT_GRAY),
            ],
            [
                ("# [0, 50)     0.045  0.012  0.001", LIGHT_GRAY),
            ],
            [
                ("# [50, 100)   0.018  0.009  0.041", LIGHT_GRAY),
            ],
            [
                ("# [100, 200]  0.003  0.005  0.561", LIGHT_GRAY),
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
        self.centered_text(sub_y, "Two outputs. One fit() call.", size=13, bold=False, color=GRAY)

        self.add_footer()

    def slide_09_production_ready(self):
        """Slide 9: production-ready feature grid (2x3)."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(9, total=11)

        self.centered_text(40, "Production-ready.", size=48, color=GOLD)

        margin = 26
        grid_gap = 8
        card_w = (WIDTH - margin * 2 - grid_gap) / 2
        card_h = 56
        start_y = 90

        features = [
            ("Spatial-HAC SEs", "Conley (1999) panel-block,\nkdtree fast path"),
            ("Cluster-Robust", "HC1 / CR1 with\nGardner GMM correction"),
            ("Survey Design", "pweights, strata, PSU,\nFPC via Binder TSL"),
            ("Subpopulation Domains", "Full-design retention\nvia zero-pad scores"),
            ("Event-Study Mode", "Per-event-time x ring\ndecomposition"),
            ("Staggered Timing", "Gardner two-stage;\nnon-staggered too"),
        ]

        for idx, (title, desc) in enumerate(features):
            row = idx // 2
            col = idx % 2
            cx = margin + col * (card_w + grid_gap)
            cy = start_y + row * (card_h + grid_gap)

            self.set_fill_color(*WHITE)
            self.set_draw_color(*SLATE_BLUE)
            self.set_line_width(0.6)
            self.rect(cx, cy, card_w, card_h, "DF")

            self.set_xy(cx + 10, cy + 8)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*GOLD)
            self.cell(card_w - 20, 10, title)

            desc_lines = desc.split("\n")
            for j, line in enumerate(desc_lines):
                self.set_xy(cx + 10, cy + 24 + j * 12)
                self.set_font("Helvetica", "", 11)
                self.set_text_color(*GRAY)
                self.cell(card_w - 20, 10, line)

        comp_y = start_y + 3 * (card_h + grid_gap) + 4
        self.set_xy(0, comp_y)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(*LIGHT_GRAY)
        self.cell(WIDTH, 8, "All composable on the basic fit() call.", align="C")

        self.add_footer()

    def slide_10_validated(self):
        """Slide 10: documented synthesis + Monte Carlo recovery."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(10, total=11)

        self.centered_text(40, "Validated.", size=42, color=SLATE_BLUE)
        self.centered_text(
            82,
            "Documented synthesis + Monte Carlo recovery on synthetic DGPs.",
            size=13,
            bold=False,
            italic=True,
            color=GRAY,
        )

        margin = 30
        box_w = WIDTH - margin * 2
        box_h = 40
        gap = 5
        start_y = 105
        bar_w = 4

        items = [
            (
                "Butts Eq 5/6 + Proposition 2.1",
                "Time-varying ring-exposure synthesis (Butts Sec 5 + Gardner + Conley).",
            ),
            (
                "Gardner Two-Stage GMM (Section 4)",
                "Stage-1 FE absorption + stage-2 with first-stage uncertainty correction.",
            ),
            (
                "Conley (1999) Spatial-HAC",
                "Matched to R conleyreg at atol=1e-6 on parity fixtures.",
            ),
            (
                "Monte Carlo Recovery",
                "Recovers known tau_total + delta_j on synthetic spillover DGPs.",
            ),
        ]

        for i, (title, desc) in enumerate(items):
            by = start_y + i * (box_h + gap)
            self.set_fill_color(*WHITE)
            self.set_draw_color(*SLATE_BLUE)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, box_h, "DF")
            self.set_fill_color(*SLATE_BLUE)
            self.rect(margin, by, bar_w, box_h, "F")

            self.set_xy(margin + bar_w + 12, by + 8)
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(*NAVY)
            self.cell(box_w - bar_w - 24, 10, title)

            self.set_xy(margin + bar_w + 12, by + 26)
            self.set_font("Helvetica", "", 12)
            self.set_text_color(*GRAY)
            self.cell(box_w - bar_w - 24, 10, desc)

        self.add_footer()

    def slide_11_cta(self):
        """Slide 11: CTA — Now in diff-diff, pip install, GitHub."""
        self.add_page()
        self.light_gradient_background()
        self._draw_vertical_sidebar(11, total=11)

        self.centered_text(58, "Now in diff-diff.", size=24, bold=False, italic=True, color=GRAY)
        self.centered_text(88, "Spillover-Aware DiD.", size=46, color=SLATE_BLUE)

        badge_w = 230
        badge_h = 42
        badge_x = (WIDTH - badge_w) / 2
        badge_y = 158
        self.set_fill_color(*SLATE_BLUE)
        self.rect(badge_x, badge_y, badge_w, badge_h, "F")

        self.set_xy(badge_x, badge_y + 12)
        self.set_font("Courier", "B", 16)
        self.set_text_color(*WHITE)
        self.cell(badge_w, 16, "$ pip install --upgrade diff-diff", align="C")

        self.centered_text(222, "github.com/igerber/diff-diff", size=18, color=SLATE_BLUE)

        self.draw_split_logo(258, size=28)

        self.centered_text(
            284, "Difference-in-Differences for Python", size=14, bold=False, color=GRAY
        )

        self.add_footer()


def main():
    pdf = SpilloverCarouselPDF()
    try:
        pdf.slide_01_cover()
        pdf.slide_02_problem()
        pdf.slide_03_real_world()
        pdf.slide_04_introducing()
        pdf.slide_05_anchor()
        pdf.slide_06_estimand()
        pdf.slide_07_output()
        pdf.slide_08_code()
        pdf.slide_09_production_ready()
        pdf.slide_10_validated()
        pdf.slide_11_cta()

        output_path = Path(__file__).parent / "diff-diff-spillover-carousel.pdf"
        pdf.output(str(output_path))
        print(f"PDF saved to: {output_path}")
    finally:
        pdf.cleanup()


if __name__ == "__main__":
    main()
