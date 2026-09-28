#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for HAD (HeterogeneousAdoptionDiD) launch."""

import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from fpdf import FPDF  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

# Computer Modern for math
plt.rcParams["mathtext.fontset"] = "cm"

# Page dimensions (4:5 portrait)
WIDTH = 270  # mm
HEIGHT = 337.5  # mm

# Colors - Light theme with indigo accent
INDIGO = (79, 70, 229)  # #4f46e5  primary accent
INDIGO_LIGHT = (165, 180, 252)  # #a5b4fc
INDIGO_DARK = (55, 48, 163)  # #3730a3
INDIGO_TINT = (238, 242, 255)  # #eef2ff  callout bg
NAVY = (15, 23, 42)  # #0f172a  primary text
ROSE = (225, 29, 72)  # #e11d48  problem accent
GRAY = (100, 116, 139)  # #64748b  secondary text
LIGHT_GRAY = (148, 163, 184)  # #94a3b8  fine print
WHITE = (255, 255, 255)
DARK_SLATE = (30, 41, 59)  # #1e293b  code block bg
CODE_BG = (15, 23, 42)  # #0f172a  even darker for contrast
GREEN_CODE = (134, 239, 172)  # #86efac  code string literals
INDIGO_CODE = (165, 180, 252)  # code keyword tone

# Hex colors for matplotlib
INDIGO_HEX = "#4f46e5"
INDIGO_LIGHT_HEX = "#a5b4fc"
INDIGO_DARK_HEX = "#3730a3"
INDIGO_TINT_HEX = "#eef2ff"
NAVY_HEX = "#0f172a"
ROSE_HEX = "#e11d48"
GRAY_HEX = "#64748b"
LIGHT_GRAY_HEX = "#94a3b8"


class HADCarouselPDF(FPDF):
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

    # Vertical sidebar accent - thin indigo bar on the left edge of
    # every slide, with a single tick mark whose vertical position
    # indicates the slide's progress through the deck (slide 1 = tick
    # near top, slide 10 = tick near bottom). Magazine-spread feel:
    # deliberate, structural, unobtrusive.

    def _draw_vertical_sidebar(self, slide_number, total=10):
        """Draw the magazine-style vertical accent on the left edge."""
        bar_x = 14  # mm from left edge
        bar_y_top = 45  # below the top margin
        bar_y_bottom = 275  # above the wordmark area
        self.set_draw_color(*INDIGO)
        self.set_line_width(0.6)
        self.line(bar_x, bar_y_top, bar_x, bar_y_bottom)

        # Progress tick - position scales linearly with slide number
        if total > 1:
            ratio = (slide_number - 1) / (total - 1)
        else:
            ratio = 0.0
        tick_y = bar_y_top + ratio * (bar_y_bottom - bar_y_top)
        self.set_line_width(0.8)
        self.line(bar_x - 4, tick_y, bar_x + 7, tick_y)

    # Decorative connector graphic - retained for reference, replaced
    # by vertical sidebar across all slides.

    def _draw_connector_graphic(self, side="right", bottom_clear=70, color=None):
        """Draw decorative curves + dots in the bottom corners.

        Parameters
        ----------
        side : "right" | "left"
            Which side gets the sweeping curves; the dot cluster goes
            on the opposite side.
        bottom_clear : float (mm)
            Vertical distance from the slide bottom that the decoration
            should occupy. Use larger values when the slide has content
            extending closer to the corners; smaller when corners are clear.
        color : (r, g, b) | None
            Color for both curves and dots. Defaults to ``INDIGO_LIGHT``
            so the decoration reads as a watermark rather than competing
            with primary content.
        """
        import math

        if color is None:
            color = INDIGO_LIGHT

        # Curve center off-page so only a partial arc shows
        if side == "right":
            cx, cy = WIDTH + 20, HEIGHT - bottom_clear / 2
            start_angle, end_angle = math.pi * 0.5, math.pi * 1.0
            dot_xs = [35, 50, 30]
        else:
            cx, cy = -20, HEIGHT - bottom_clear / 2
            start_angle, end_angle = 0.0, math.pi * 0.5
            dot_xs = [WIDTH - 35, WIDTH - 50, WIDTH - 30]

        # Lower clip: keep curves above the wordmark area (footer text
        # sits at HEIGHT - 18; leave 6mm of breathing room above it).
        lower_clip_y = HEIGHT - 24
        upper_clip_y = HEIGHT - bottom_clear

        self.set_draw_color(*color)
        for i, radius in enumerate([60, 80, 100]):
            self.set_line_width(2.0 - i * 0.4)
            segments = 30
            for j in range(segments):
                t1 = start_angle + (end_angle - start_angle) * j / segments
                t2 = start_angle + (end_angle - start_angle) * (j + 1) / segments
                x1 = cx + radius * math.cos(t1)
                y1 = cy + radius * math.sin(t1)
                x2 = cx + radius * math.cos(t2)
                y2 = cy + radius * math.sin(t2)
                # Clip both above the bottom_clear line (so curves do not
                # rise into content) and below the lower_clip_y line
                # (so curves do not collide with the wordmark).
                if y1 < upper_clip_y or y2 < upper_clip_y:
                    continue
                if y1 > lower_clip_y or y2 > lower_clip_y:
                    continue
                self.line(x1, y1, x2, y2)

        # Dot cluster - 3 dots of decreasing radius, also using the
        # lighter watermark color
        self.set_fill_color(*color)
        dot_ys = [HEIGHT - 50, HEIGHT - 38, HEIGHT - 30]
        for i, (dx, dy) in enumerate(zip(dot_xs, dot_ys)):
            dr = 2.5 - i * 0.4
            self.ellipse(dx - dr, dy - dr, dr * 2, dr * 2, "F")

    # Background and footer

    def light_gradient_background(self):
        """Light gradient background, top #e1f0ff fading to white."""
        steps = 50
        for i in range(steps):
            ratio = i / steps
            r = int(225 + (255 - 225) * ratio)
            g = int(240 + (255 - 240) * ratio)
            b = 255
            self.set_fill_color(r, g, b)
            y = i * HEIGHT / steps
            self.rect(0, y, WIDTH, HEIGHT / steps + 1, "F")

    def add_footer(self):
        # No horizontal rule (matches original carousel) so that
        # decorative curves can pass through the bottom region cleanly.
        self.set_font("Helvetica", "B", 12)
        dd_text = "diff-diff "
        v_text = "v3.3.1"
        dd_w = self.get_string_width(dd_text)
        v_w = self.get_string_width(v_text)
        start_x = (WIDTH - dd_w - v_w) / 2

        self.set_xy(start_x, HEIGHT - 18)
        self.set_text_color(*GRAY)
        self.cell(dd_w, 10, dd_text)
        self.set_text_color(*INDIGO)
        self.cell(v_w, 10, v_text)

    # Text helpers

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
        """Split-color diff-diff logo with indigo middle dash."""
        self.set_xy(0, y)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="R")
        self.set_text_color(*INDIGO)
        self.cell(10, 10, "-", align="C")
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="L")

    # Equation rendering

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

    # Slide-2 timeline visual: left = standard DiD, right = universal rollout
    # with varying dose intensity

    def _render_problem_timelines(self):
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(11, 4.5), gridspec_kw={"wspace": 0.30}
        )
        fig.patch.set_facecolor("white")

        n_periods = 10
        periods = np.arange(n_periods)
        bar_height = 0.6
        untreated_color = LIGHT_GRAY_HEX
        treated_left = INDIGO_HEX

        # Left: standard DiD - mix of treated cohorts and a never-treated row
        ax_left.set_facecolor("white")
        ax_left.set_title("Standard DiD", fontsize=13, fontweight="bold", color=NAVY_HEX, pad=10)
        left_units = [
            ("Unit A", 4),
            ("Unit B", 5),
            ("Unit C", 6),
            ("Unit D", None),  # never-treated (control), placed at bottom
        ]
        for row, (label, onset) in enumerate(left_units):
            y = len(left_units) - 1 - row
            for t in periods:
                color = treated_left if (onset is not None and t >= onset) else untreated_color
                ax_left.barh(
                    y, 0.9, left=t, height=bar_height, color=color, edgecolor="white", linewidth=0.5
                )
            ax_left.text(-0.5, y, label, ha="right", va="center", fontsize=10, color=NAVY_HEX)

        # Highlight the never-treated row as the "control"
        ax_left.text(
            n_periods + 0.2,
            0,
            "<- control",
            ha="left",
            va="center",
            fontsize=10,
            color=INDIGO_DARK_HEX,
            fontweight="bold",
        )

        ax_left.set_xlim(-0.2, n_periods + 2.5)
        ax_left.set_ylim(-0.8, len(left_units) - 0.3)
        ax_left.set_xlabel("Time", fontsize=10, color=GRAY_HEX)
        ax_left.set_yticks([])
        ax_left.tick_params(axis="x", colors=GRAY_HEX, labelsize=8)
        for spine in ax_left.spines.values():
            spine.set_visible(False)

        # Right: universal rollout - all units treated, varying dose intensity.
        # Encode dose via alpha (saturation) on the indigo color.
        ax_right.set_facecolor("white")
        ax_right.set_title(
            "Universal rollout", fontsize=13, fontweight="bold", color=INDIGO_HEX, pad=10
        )
        F = 5  # treatment date - common across all units
        # (label, dose level in [0.15, 1.0])
        right_units = [
            ("Unit A", 0.20),
            ("Unit B", 0.50),
            ("Unit C", 0.85),
            ("Unit D", 1.00),
        ]
        for row, (label, dose) in enumerate(right_units):
            y = len(right_units) - 1 - row
            for t in periods:
                if t < F:
                    color = untreated_color
                    alpha = 1.0
                else:
                    color = INDIGO_HEX
                    alpha = dose
                ax_right.barh(
                    y,
                    0.9,
                    left=t,
                    height=bar_height,
                    color=color,
                    alpha=alpha,
                    edgecolor="white",
                    linewidth=0.5,
                )
            ax_right.text(-0.5, y, label, ha="right", va="center", fontsize=10, color=NAVY_HEX)
            # Dose annotation on the right side
            ax_right.text(
                n_periods + 0.2,
                y,
                f"D = {dose:.2f}",
                ha="left",
                va="center",
                fontsize=9,
                color=INDIGO_DARK_HEX,
                family="monospace",
            )

        ax_right.set_xlim(-0.2, n_periods + 2.5)
        ax_right.set_ylim(-0.8, len(right_units) - 0.3)
        ax_right.set_xlabel("Time", fontsize=10, color=GRAY_HEX)
        ax_right.set_yticks([])
        ax_right.tick_params(axis="x", colors=GRAY_HEX, labelsize=8)
        for spine in ax_right.spines.values():
            spine.set_visible(False)

        fig.subplots_adjust(left=0.07, right=0.95, bottom=0.10, top=0.90)

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
        plt.close(fig)
        with PILImage.open(path) as img:
            pw, ph = img.size
        self._temp_files.append(path)
        return path, pw, ph

    # Slide-4 visual: dose histogram with boundary marker and local-linear fit

    def _render_dose_boundary(self):
        fig, ax = plt.subplots(figsize=(10, 5.0))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Single-axis scatter of (D, delta-Y) data
        rng = np.random.default_rng(7)
        n = 180
        d = rng.uniform(0.0, 1.0, n)
        dy = 0.30 * d + 0.08 * rng.standard_normal(n)

        # Anchor subset: least-treated units near the boundary
        anchor_mask = d <= 0.10

        # Plot the bulk of units in light gray (de-emphasized)
        ax.scatter(
            d[~anchor_mask],
            dy[~anchor_mask],
            color=LIGHT_GRAY_HEX,
            s=18,
            alpha=0.7,
            edgecolors="none",
            label="Treated units",
        )
        # Plot the anchor units in indigo (emphasized)
        ax.scatter(
            d[anchor_mask],
            dy[anchor_mask],
            color=INDIGO_HEX,
            s=44,
            alpha=0.95,
            edgecolors=NAVY_HEX,
            linewidths=0.5,
            label="Least-treated (anchor)",
        )

        # Local-linear fit line
        x_grid = np.linspace(0.0, 1.0, 100)
        y_grid = 0.30 * x_grid
        ax.plot(
            x_grid,
            y_grid,
            color=NAVY_HEX,
            linewidth=2.4,
            linestyle="-",
            zorder=4,
            label="Local-linear fit",
        )

        # Boundary point at d=0
        ax.axvline(0.0, color=INDIGO_DARK_HEX, linestyle="--", linewidth=1.4, zorder=2, alpha=0.6)
        ax.scatter(
            [0.0],
            [0.0],
            color=INDIGO_DARK_HEX,
            s=110,
            zorder=6,
            marker="o",
            edgecolors=NAVY_HEX,
            linewidths=1.0,
        )
        ax.annotate(
            "anchor at d = 0",
            xy=(0.0, 0.0),
            xytext=(0.18, -0.07),
            fontsize=11,
            color=NAVY_HEX,
            arrowprops=dict(arrowstyle="->", color=NAVY_HEX, lw=0.9),
        )

        ax.set_xlabel("Dose D", fontsize=13, color=NAVY_HEX)
        ax.set_ylabel(r"$\Delta Y$ (change in outcome)", fontsize=13, color=NAVY_HEX)
        ax.tick_params(colors=GRAY_HEX)
        for spine in ax.spines.values():
            spine.set_color(LIGHT_GRAY_HEX)

        ax.legend(loc="upper left", fontsize=10, frameon=False, labelcolor=NAVY_HEX)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.18, 0.45)
        fig.tight_layout()

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.10, facecolor="white")
        plt.close(fig)
        with PILImage.open(path) as img:
            pw, ph = img.size
        self._temp_files.append(path)
        return path, pw, ph

    # Slide-6 visual: three side-by-side dose-distribution histograms
    # representing the three auto-detected design paths

    def _render_three_designs(self):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"wspace": 0.35})
        fig.patch.set_facecolor("white")
        rng = np.random.default_rng(7)

        # Panel 1: continuous_at_zero (Design 1', d_lower = 0)
        d1 = rng.uniform(0.0, 1.0, 2000)
        d1[0] = 0.0
        # Panel 2: continuous_near_d_lower (Design 1, d_lower = 0.2)
        u = rng.beta(2, 2, 2000)
        d2 = 0.20 + 0.80 * u
        # Panel 3: mass_point (Design 1 with bunching at d_lower = 0.5)
        n_mass = 600
        d3 = np.concatenate([np.full(n_mass, 0.5), rng.uniform(0.5, 1.0, 2000 - n_mass)])

        panels = [
            (axes[0], d1, "Reaches zero", "continuous_at_zero", 0.0),
            (axes[1], d2, "Lifted above zero", "continuous_near_d_lower", 0.20),
            (axes[2], d3, "Bunching at minimum", "mass_point", 0.5),
        ]

        for ax, dose, title, dispatch, d_lower in panels:
            ax.set_facecolor("white")
            ax.hist(
                dose,
                bins=30,
                color=INDIGO_LIGHT_HEX,
                edgecolor=INDIGO_HEX,
                alpha=0.75,
                linewidth=0.4,
            )
            ax.axvline(d_lower, color=INDIGO_DARK_HEX, linestyle="--", linewidth=1.4)
            # Cap the y-axis so the mass-point spike does not dwarf the
            # continuous distribution.
            cap = 130
            ax.set_ylim(0, cap)
            counts, _ = np.histogram(dose, bins=30, range=(0, 1))
            # Only annotate "spike" when there's a genuine mass concentration
            # (mass-point case). Use 4x the median count as the threshold so
            # natural distribution peaks (e.g. Beta(2,2)) are not misread as
            # spikes.
            median_count = np.median(counts[counts > 0])
            if counts.max() > max(cap, 4 * median_count):
                spike_bin_x = (np.argmax(counts) + 0.5) / 30
                ax.annotate(
                    f"spike: {int(counts.max())} units",
                    xy=(spike_bin_x, cap),
                    xytext=(spike_bin_x + 0.15, cap * 0.85),
                    arrowprops=dict(arrowstyle="->", color=INDIGO_DARK_HEX, lw=0.9),
                    fontsize=8,
                    color=INDIGO_DARK_HEX,
                    ha="left",
                    va="top",
                )
            ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY_HEX, pad=6)
            ax.text(
                0.5,
                -0.32,
                f'design="{dispatch}"',
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                color=INDIGO_DARK_HEX,
                family="monospace",
            )
            ax.set_xlim(-0.02, 1.02)
            ax.tick_params(colors=GRAY_HEX, labelsize=8)
            ax.set_yticks([])
            ax.set_xlabel("Dose D", fontsize=10, color=GRAY_HEX)
            for spine in ax.spines.values():
                spine.set_color(LIGHT_GRAY_HEX)

        fig.subplots_adjust(left=0.04, right=0.98, bottom=0.30, top=0.85)

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.10, facecolor="white")
        plt.close(fig)
        with PILImage.open(path) as img:
            pw, ph = img.size
        self._temp_files.append(path)
        return path, pw, ph

    # Code block

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

    # ====================================================================
    # SLIDES
    # ====================================================================

    def slide_01_hook(self):
        """Slide 1: Manifesto - Most methods require a control group."""
        self.add_page()
        self.light_gradient_background()

        # Logo at top
        self.draw_split_logo(40, size=42)

        # Hero typography - takes up the middle ~half of the slide
        self.centered_text(112, "Most methods require", size=38)
        self.centered_text(150, "a control group.", size=38)

        # Indigo punchline below, bigger
        self.centered_text(206, "Reality doesn't.", size=54, color=INDIGO)

        # Tagline introducing the full estimator name. Place this BEFORE
        # decorations so we can keep curves clear of the byline area.
        self.set_xy(0, HEIGHT - 70)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*INDIGO)
        self.cell(WIDTH, 8, "Heterogeneous Adoption Designs (HAD).", align="C")
        self.set_xy(0, HEIGHT - 58)
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 8, "Now in diff-diff.", align="C")
        self.set_xy(0, HEIGHT - 46)
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 8, "de Chaisemartin et al. (2026).", align="C")
        self.set_xy(0, HEIGHT - 35)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*LIGHT_GRAY)
        self.cell(WIDTH, 8, "doi.org/10.48550/arXiv.2405.04465", align="C")

        # Decorations - bottom_clear=80 keeps curves below the tagline
        # (which sits around y=270, well above the upper_clip_y of
        # HEIGHT-35 = 302.5). Curves visible in the small strip between
        # ~302.5 and ~313.5.

        self.add_footer()

    def slide_02_problem(self):
        """Slide 2: Provocative hook - what if there's no control?"""
        self.add_page()
        self.light_gradient_background()

        # The provocative question is now the dominant headline
        self.centered_text(40, "Standard techniques", size=34)
        self.centered_text(62, "subtract a control group.", size=34)
        self.centered_text(102, "What if there isn't one?", size=42, color=INDIGO)

        # Timeline visual
        plot_path, ppw, pph = self._render_problem_timelines()
        plot_w = WIDTH * 0.90
        plot_aspect = pph / ppw
        plot_h = plot_w * plot_aspect
        plot_x = (WIDTH - plot_w) / 2
        plot_y = 152
        self.image(plot_path, plot_x, plot_y, plot_w)

        # Tiny supporting line below visual
        cap_y = plot_y + plot_h + 8
        self.centered_text(
            cap_y, "No control group, just different doses.", size=14, bold=False, color=GRAY
        )

        self.add_footer()

    def slide_03_real_world(self):
        """Slide 3: Where this happens."""
        self.add_page()
        self.light_gradient_background()

        # Headline: emphasis on "everyone" via split coloring
        self.set_font("Helvetica", "B", 40)
        text_a = "Sometimes "
        text_b = "everyone"
        w_a = self.get_string_width(text_a)
        w_b = self.get_string_width(text_b)
        total_w = w_a + w_b
        start_x = (WIDTH - total_w) / 2

        self.set_xy(start_x, 40)
        self.set_text_color(*NAVY)
        self.cell(w_a, 20, text_a)
        self.set_text_color(*INDIGO)
        self.cell(w_b, 20, text_b)

        self.centered_text(68, "gets treated.", size=40)

        margin = 30
        box_w = WIDTH - margin * 2
        box_h = 40
        gap = 5
        start_y = 120
        bar_w = 4

        scenarios = [
            (
                "Universal Pricing Rollout",
                "Every store moves to the new pricing; magnitude varies.",
            ),
            ("Industry-Wide Regulation", "Every firm is subject to the new rule; exposure varies."),
            (
                "Company-Wide Training",
                "Every employee receives the program; hours invested differ.",
            ),
            (
                "National Policy Floors",
                "Every state implements the federal floor; intensity differs.",
            ),
        ]

        for i, (title, desc) in enumerate(scenarios):
            by = start_y + i * (box_h + gap)
            self.set_fill_color(*WHITE)
            self.set_draw_color(220, 220, 230)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, box_h, "DF")
            self.set_fill_color(*INDIGO)
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

    def slide_04_introducing_had(self):
        """Slide 4: Formal HAD introduction - high-level overview."""
        self.add_page()
        self.light_gradient_background()

        # Headline
        self.centered_text(40, "The HAD estimator.", size=44)

        # Single-line subtitle (eliminates the prior 2-line spacing issue)
        self.centered_text(
            82,
            "For Heterogeneous Adoption Designs.",
            size=18,
            bold=False,
            italic=True,
            color=INDIGO,
        )

        # Lead description (tighter line spacing)
        self.centered_text(
            118, "Recovers treatment effects when there's", size=14, bold=False, color=GRAY
        )
        self.centered_text(132, "no untreated comparison.", size=14, bold=False, color=GRAY)

        # Three capability cards
        margin = 30
        box_w = WIDTH - margin * 2
        box_h = 40
        gap = 5
        start_y = 168
        bar_w = 4

        items = [
            ("Anchor on the least-treated.", "Treats them as a quasi-control group."),
            (
                "Recover the Weighted Average Slope.",
                "Local-linear fit at the dose-support boundary.",
            ),
            ("Auto-detect the design path.", "Three identification strategies, one API."),
        ]
        # Cards start higher now that subtitle is one line

        for i, (title, desc) in enumerate(items):
            by = start_y + i * (box_h + gap)
            self.set_fill_color(*WHITE)
            self.set_draw_color(220, 220, 230)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, box_h, "DF")
            self.set_fill_color(*INDIGO)
            self.rect(margin, by, bar_w, box_h, "F")

            self.set_xy(margin + bar_w + 12, by + 8)
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(*NAVY)
            self.cell(box_w - bar_w - 24, 10, title)

            self.set_xy(margin + bar_w + 12, by + 24)
            self.set_font("Helvetica", "", 12)
            self.set_text_color(*GRAY)
            self.cell(box_w - bar_w - 24, 10, desc)

        # Methodology footnote (transparency on the WAS / WAS_d_lower split)
        footnote_y = start_y + len(items) * (box_h + gap) + 8
        self.set_xy(0, footnote_y)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(*LIGHT_GRAY)
        self.cell(
            WIDTH,
            8,
            "Returns WAS on continuous-at-zero designs; WAS_d_lower otherwise.",
            align="C",
        )

        self.add_footer()

    def slide_05_insight(self):
        """Slide 4: The insight - least-treated as quasi-control."""
        self.add_page()
        self.light_gradient_background()

        # Bigger headline, tight stacking
        self.centered_text(40, "Use the least-treated", size=36)
        self.centered_text(64, "as a quasi-control.", size=36, color=INDIGO)

        # Visual
        plot_path, ppw, pph = self._render_dose_boundary()
        plot_w = WIDTH * 0.90
        plot_aspect = pph / ppw
        plot_h = plot_w * plot_aspect
        plot_x = (WIDTH - plot_w) / 2
        plot_y = 110
        self.image(plot_path, plot_x, plot_y, plot_w)

        # Single bold tagline below
        cap_y = plot_y + plot_h + 14
        self.centered_text(
            cap_y, "Local-linear at the dose boundary.", size=18, bold=True, color=INDIGO
        )

        # Decoration in bottom corners (curves clip above bottom_clear)

        self.add_footer()

    def slide_06_estimand(self):
        """Slide 5: Weighted Average Slope (WAS) equation."""
        self.add_page()
        self.light_gradient_background()

        # Push content down so the slide is vertically balanced
        self.centered_text(70, "Weighted Average Slope.", size=42)

        self.centered_text(
            110, "(Design 1', d_lower = 0)", size=14, bold=False, italic=True, color=GRAY
        )

        # Full identification equation with the boundary-limit term
        eq_path, epw, eph = self._render_equations(
            [
                r"$\mathrm{WAS}\;=\;\frac{E[\Delta Y]"
                r"\;-\;\lim_{d \downarrow 0}\,"
                r"E[\Delta Y \mid D \leq d]}"
                r"{E[D]}$"
            ],
            fontsize=22,
            color=INDIGO_HEX,
        )
        eq_h = self._place_equation_centered(eq_path, epw, eph, 138, max_w=240)

        # Plain-English gloss with TIGHT line spacing
        gloss_y = 138 + eq_h + 22
        self.centered_text(
            gloss_y, "Treatment effect per unit of dose,", size=15, bold=False, color=NAVY
        )
        self.centered_text(
            gloss_y + 12, "weighted by the dose itself.", size=15, bold=False, color=NAVY
        )

        # Consistent decoration footprint across slides

        self.add_footer()

    def slide_07_three_designs(self):
        """Slide 6: HAD reads the dose distribution."""
        self.add_page()
        self.light_gradient_background()

        # Reframed: "the HAD estimator" per formal naming. Smaller size
        # to fit the longer line. Content shifted down for vertical balance.
        self.centered_text(72, "The HAD estimator reads", size=36)
        self.centered_text(96, "the dose distribution.", size=36, color=INDIGO)

        # Three histograms
        plot_path, ppw, pph = self._render_three_designs()
        plot_w = WIDTH * 0.94
        plot_aspect = pph / ppw
        plot_h = plot_w * plot_aspect
        plot_x = (WIDTH - plot_w) / 2
        plot_y = 144
        self.image(plot_path, plot_x, plot_y, plot_w)

        # Single tight tagline
        cap_y = plot_y + plot_h + 16
        self.centered_text(
            cap_y, "One API, three identification strategies.", size=17, bold=True, color=INDIGO
        )

        self.add_footer()

    def slide_08_code(self):
        """Slide 7: The code."""
        self.add_page()
        self.light_gradient_background()

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
                ("from", INDIGO_CODE),
                (" diff_diff ", WHITE),
                ("import", INDIGO_CODE),
                (" HeterogeneousAdoptionDiD ", WHITE),
                ("as", INDIGO_CODE),
                (" HAD", WHITE),
            ],
            [],
            [
                ("result", WHITE),
                (" = ", INDIGO_CODE),
                ("HAD", GREEN_CODE),
                ("(", WHITE),
                ("design", WHITE),
                ("=", INDIGO_CODE),
                ("'continuous_at_zero'", GREEN_CODE),
                (").", WHITE),
                ("fit(", WHITE),
            ],
            [("    data,", WHITE)],
            [
                ("    ", WHITE),
                ("outcome_col", WHITE),
                ("=", INDIGO_CODE),
                ("'outcome'", GREEN_CODE),
                (",", WHITE),
            ],
            [
                ("    ", WHITE),
                ("dose_col", WHITE),
                ("=", INDIGO_CODE),
                ("'dose'", GREEN_CODE),
                (",", WHITE),
            ],
            [
                ("    ", WHITE),
                ("time_col", WHITE),
                ("=", INDIGO_CODE),
                ("'period'", GREEN_CODE),
                (",", WHITE),
            ],
            [
                ("    ", WHITE),
                ("unit_col", WHITE),
                ("=", INDIGO_CODE),
                ("'unit'", GREEN_CODE),
                (",", WHITE),
            ],
            [(")", WHITE)],
            [],
            [
                ("print(", WHITE),
                ("result.att", WHITE),
                (")", WHITE),
                ("        # WAS estimate", LIGHT_GRAY),
            ],
            [
                ("print(", WHITE),
                ("result.conf_int", WHITE),
                (")", WHITE),
                ("   # Bias-corrected 95% CI", LIGHT_GRAY),
            ],
            [
                ("print(", WHITE),
                ("result.design", WHITE),
                (")", WHITE),
                ("     # 'continuous_at_zero'", LIGHT_GRAY),
            ],
        ]

        code_h = self._add_code_block(
            margin,
            code_y,
            WIDTH - margin * 2,
            token_lines,
            font_size=11,
            line_height=10,
        )

        sub_y = min(code_y + code_h + 14, HEIGHT - 48)
        self.centered_text(
            sub_y, "Bias correction, three design paths,", size=13, bold=False, color=GRAY
        )
        self.centered_text(
            sub_y + 16, "event study - one fit() call.", size=13, bold=False, color=GRAY
        )

        self.add_footer()

    def slide_09_features(self):
        """Slide 8: What ships with it."""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(40, "Production-ready.", size=48, color=INDIGO)

        margin = 26
        grid_gap = 8
        card_w = (WIDTH - margin * 2 - grid_gap) / 2
        card_h = 56
        start_y = 90

        features = [
            ("Bias-Corrected CIs", "Calonico-Cattaneo-Farrell\nported in-house"),
            ("Auto Design Detection", "Three identification paths,\none API"),
            ("Dynamic Event Study", "Per-horizon estimates\nwith pointwise CIs"),
            ("Survey Support", "pweights, strata, PSU, FPC\nvia Binder TSL"),
            ("Sup-t Bands", "Simultaneous CIs across\nevent-study horizons"),
            ("Pre-Test Diagnostics", "QUG, Stute, Yatchew-HR,\njoint workflow"),
        ]

        for idx, (title, desc) in enumerate(features):
            row = idx // 2
            col = idx % 2
            cx = margin + col * (card_w + grid_gap)
            cy = start_y + row * (card_h + grid_gap)

            self.set_fill_color(*WHITE)
            self.set_draw_color(*INDIGO)
            self.set_line_width(0.6)
            self.rect(cx, cy, card_w, card_h, "DF")

            self.set_xy(cx + 10, cy + 8)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*INDIGO)
            self.cell(card_w - 20, 10, title)

            desc_lines = desc.split("\n")
            for j, line in enumerate(desc_lines):
                self.set_xy(cx + 10, cy + 24 + j * 12)
                self.set_font("Helvetica", "", 11)
                self.set_text_color(*GRAY)
                self.cell(card_w - 20, 10, line)

        self.add_footer()

    def slide_10_validated(self):
        """Slide 9: Validated against did_had (R)."""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(40, "Validated against R.", size=42, color=INDIGO)
        self.centered_text(
            82,
            "End-to-end match against DIDHAD v2.0.0 on continuous-at-zero designs.",
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
            ("R Parity vs DIDHAD", "End-to-end match against the DIDHAD R package (v2.0.0)."),
            (
                "Paper-Equation Parity",
                "Theorem 1 / Equation 7 and Theorem 3 / Equation 11 implemented as written.",
            ),
            (
                "Bias-Corrected CI Bit-Identity",
                "Uniform-weights matches the in-house nprobust port at atol=1e-14.",
            ),
            (
                "Monte Carlo Oracle Consistency",
                "Recovers the known-tau DGP; coverage near nominal level.",
            ),
        ]

        for i, (title, desc) in enumerate(items):
            by = start_y + i * (box_h + gap)
            self.set_fill_color(*WHITE)
            self.set_draw_color(*INDIGO)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, box_h, "DF")
            self.set_fill_color(*INDIGO)
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
        """Slide 10: CTA - First Python implementation."""
        self.add_page()
        self.light_gradient_background()

        # Kicker line above the hero, then big indigo product name
        self.centered_text(58, "Now in diff-diff.", size=24, bold=False, italic=True, color=GRAY)
        self.centered_text(88, "The HAD estimator.", size=50, color=INDIGO)

        # pip install badge
        badge_w = 230
        badge_h = 42
        badge_x = (WIDTH - badge_w) / 2
        badge_y = 158
        self.set_fill_color(*INDIGO)
        self.rect(badge_x, badge_y, badge_w, badge_h, "F")

        self.set_xy(badge_x, badge_y + 12)
        self.set_font("Courier", "B", 16)
        self.set_text_color(*WHITE)
        self.cell(badge_w, 16, "$ pip install --upgrade diff-diff", align="C")

        self.centered_text(222, "github.com/igerber/diff-diff", size=18, color=INDIGO)

        self.draw_split_logo(258, size=28)

        self.centered_text(
            284, "Difference-in-Differences for Python", size=14, bold=False, color=GRAY
        )

        self.add_footer()


def main():
    pdf = HADCarouselPDF()
    try:
        pdf.slide_01_hook()
        pdf.slide_02_problem()
        pdf.slide_03_real_world()
        pdf.slide_04_introducing_had()
        pdf.slide_05_insight()
        pdf.slide_06_estimand()
        pdf.slide_07_three_designs()
        pdf.slide_08_code()
        pdf.slide_09_features()
        pdf.slide_10_validated()
        pdf.slide_11_cta()

        output_path = Path(__file__).parent / "diff-diff-had-carousel.pdf"
        pdf.output(str(output_path))
        print(f"PDF saved to: {output_path}")
    finally:
        pdf.cleanup()


if __name__ == "__main__":
    main()
