#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for diff-diff v2.6 release."""

import math
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from fpdf import FPDF  # noqa: E402

# Use Computer Modern math font (LaTeX-like)
plt.rcParams["mathtext.fontset"] = "cm"

# LinkedIn carousel dimensions (4:5 aspect ratio)
WIDTH = 270  # mm
HEIGHT = 337.5  # mm

# Colors - Light theme with violet accent
MID_BLUE = (59, 130, 246)  # #3b82f6
NAVY = (15, 23, 42)  # #0f172a
WHITE = (255, 255, 255)
RED = (220, 38, 38)  # #dc2626
GREEN = (22, 163, 74)  # #16a34a
GRAY = (100, 116, 139)  # #64748b
LIGHT_GRAY = (148, 163, 184)  # #94a3b8
VIOLET = (124, 58, 237)  # #7c3aed - v2.6 accent
DARK_SLATE = (30, 41, 59)  # #1e293b - code block bg

VIOLET_TINT = (245, 240, 255)  # callout background

# Hex colors for matplotlib
NAVY_HEX = "#0f172a"
VIOLET_HEX = "#7c3aed"
VIOLET_LIGHT_HEX = "#c4b5fd"


class CarouselV26PDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format=(WIDTH, HEIGHT))
        self.set_auto_page_break(False)
        self._temp_files = []

    def cleanup(self):
        """Remove temporary image files."""
        for f in self._temp_files:
            try:
                os.unlink(f)
            except OSError:
                pass

    # ── Equation Rendering ────────────────────────────────────────────

    def _render_equations(self, latex_lines, fontsize=26):
        """Render one or more LaTeX equations to a single PNG image.

        Args:
            latex_lines: list of LaTeX math strings (each wrapped in $...$)
            fontsize: matplotlib font size

        Returns:
            (path, pixel_width, pixel_height)
        """
        n = len(latex_lines)
        fig_h = max(0.7, 0.55 * n + 0.15)
        fig = plt.figure(figsize=(10, fig_h))

        for i, line in enumerate(latex_lines):
            y_frac = 1.0 - (2 * i + 1) / (2 * n)
            fig.text(
                0.5, y_frac, line,
                fontsize=fontsize, ha="center", va="center",
                color=NAVY_HEX,
            )

        fig.patch.set_alpha(0)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.06,
                    transparent=True)
        plt.close(fig)

        with PILImage.open(path) as img:
            pw, ph = img.size

        self._temp_files.append(path)
        return path, pw, ph

    def _place_equation(self, path, pw, ph, box_x, _box_y, box_w,
                        content_top, content_bottom):
        """Place an equation image centered in a region of a box."""
        max_w = box_w * 0.82
        aspect = ph / pw
        display_w = max_w
        display_h = display_w * aspect

        # Shrink if too tall for the available space
        avail_h = content_bottom - content_top
        if display_h > avail_h:
            display_h = avail_h
            display_w = display_h / aspect

        eq_x = box_x + (box_w - display_w) / 2
        eq_y = content_top + (avail_h - display_h) / 2
        self.image(path, eq_x, eq_y, display_w)

    # ── Dose-Response Figure ──────────────────────────────────────────

    def _render_dose_response_figure(self):
        """Render illustrative ATT(d) dose-response curve to PNG.

        Returns:
            (path, pixel_width, pixel_height)
        """
        d = np.linspace(0.5, 10, 200)
        att = 1 + 2 * np.log(d)
        # Widening SE at extremes
        se = 0.3 + 0.15 * (d - 5) ** 2 / 25 + 0.1 * np.exp(-d)
        upper = att + 1.96 * se
        lower = att - 1.96 * se

        fig, ax = plt.subplots(figsize=(8, 4.5))

        # Confidence band
        ax.fill_between(d, lower, upper, alpha=0.25, color=VIOLET_LIGHT_HEX,
                         label="95% CI")
        # ATT(d) curve
        ax.plot(d, att, color=VIOLET_HEX, linewidth=2.5, label="ATT(d)")
        # Zero line
        ax.axhline(0, color=NAVY_HEX, linewidth=0.8, linestyle="--", alpha=0.5)

        ax.set_xlabel("Dose (d)", fontsize=13, color=NAVY_HEX)
        ax.set_ylabel("ATT(d)", fontsize=13, color=NAVY_HEX)
        ax.tick_params(colors=NAVY_HEX, labelsize=11)
        for spine in ax.spines.values():
            spine.set_color(NAVY_HEX)
            spine.set_linewidth(0.8)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.legend(fontsize=11, framealpha=0.9)

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1,
                    facecolor="white")
        plt.close(fig)

        with PILImage.open(path) as img:
            pw, ph = img.size

        self._temp_files.append(path)
        return path, pw, ph

    # ── Helper Methods ────────────────────────────────────────────────

    def add_connector_graphic(self, position="right"):
        """Add decorative connector graphic to bottom corner."""
        if position == "right":
            cx = WIDTH + 20
            cy = HEIGHT - 40
        else:
            cx = -20
            cy = HEIGHT - 40

        self.set_draw_color(*MID_BLUE)
        for i, radius in enumerate([60, 80, 100]):
            self.set_line_width(2.5 - i * 0.5)
            segments = 30
            if position == "right":
                start_angle = math.pi * 0.5
                end_angle = math.pi * 1.0
            else:
                start_angle = 0
                end_angle = math.pi * 0.5

            for j in range(segments):
                t1 = start_angle + (end_angle - start_angle) * j / segments
                t2 = start_angle + (end_angle - start_angle) * (j + 1) / segments
                x1 = cx + radius * math.cos(t1)
                y1 = cy + radius * math.sin(t1)
                x2 = cx + radius * math.cos(t2)
                y2 = cy + radius * math.sin(t2)
                self.line(x1, y1, x2, y2)

        self.set_fill_color(*MID_BLUE)
        if position == "right":
            dot_positions = [(35, HEIGHT - 60), (50, HEIGHT - 45), (30, HEIGHT - 35)]
        else:
            dot_positions = [
                (WIDTH - 35, HEIGHT - 60),
                (WIDTH - 50, HEIGHT - 45),
                (WIDTH - 30, HEIGHT - 35),
            ]
        for i, (dx, dy) in enumerate(dot_positions):
            dot_radius = 3 - i * 0.5
            self.ellipse(
                dx - dot_radius, dy - dot_radius, dot_radius * 2, dot_radius * 2, "F"
            )

    def light_gradient_background(self):
        """Draw light gradient background (top #e1f0ff fading to white)."""
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
        """Add footer with logo."""
        self.set_xy(0, HEIGHT - 25)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 10, "diff-diff", align="C")

    def centered_text(self, y, text, size=28, bold=True, color=NAVY):
        """Add centered text."""
        self.set_xy(0, y)
        self.set_font("Helvetica", "B" if bold else "", size)
        self.set_text_color(*color)
        self.cell(WIDTH, size * 0.5, text, align="C")

    def add_list_item(self, y, icon, text, icon_color, text_size=22):
        """Add a list item with icon."""
        margin = 50
        self.set_xy(margin, y)
        self.set_font("Helvetica", "B", text_size + 2)
        self.set_text_color(*icon_color)
        self.cell(25, 12, icon, align="C")
        self.set_text_color(*NAVY)
        self.set_font("Helvetica", "", text_size)
        self.cell(WIDTH - margin * 2 - 25, 12, text)

    def draw_split_logo(self, y, size=18):
        """Draw the split-color diff-diff logo."""
        self.set_xy(0, y)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="R")
        self.set_text_color(*MID_BLUE)
        self.cell(10, 10, "-", align="C")
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="L")

    # ── Slide 1: Hook ─────────────────────────────────────────────────

    def slide_hook(self):
        """Slide 1: What if treatment isn't binary?"""
        self.add_page()
        self.light_gradient_background()

        self.draw_split_logo(55, size=60)
        self.centered_text(120, "v2.6", size=50, color=VIOLET)

        self.centered_text(170, "What if treatment", size=26)
        self.centered_text(193, "isn't binary?", size=26)

        teasers = [
            "Dose-response curves for continuous treatments",
            "Callaway, Goodman-Bacon & Sant'Anna (2024)",
            "B-spline smoothing with analytical SEs",
        ]
        y_start = 230
        for i, teaser in enumerate(teasers):
            self.set_xy(0, y_start + i * 22)
            self.set_font("Helvetica", "", 17)
            self.set_text_color(*GRAY)
            self.cell(WIDTH, 10, teaser, align="C")

        self.add_footer()

    # ── Slide 2: The Problem ──────────────────────────────────────────

    def slide_problem(self):
        """Slide 2: Binary DiD Loses Information."""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(30, "Binary DiD", size=38)
        self.centered_text(63, "Loses Information", size=38, color=RED)

        # Visual comparison: bars showing binarization vs continuous dose
        margin = 40
        panel_y = 105
        panel_w = (WIDTH - margin * 3) / 2
        bar_h = 14
        bar_gap = 6
        n_bars = 5

        # Dose values for illustration (varying widths)
        doses = [0.3, 0.7, 0.5, 0.9, 0.45]

        # Left panel: Binary — all bars collapsed to same "Treated" width
        left_x = margin
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*NAVY)
        self.set_xy(left_x, panel_y - 16)
        self.cell(panel_w, 10, "Binary: D = {0, 1}", align="C")

        for i in range(n_bars):
            y = panel_y + i * (bar_h + bar_gap)
            bar_w = panel_w * 0.85  # all same width (binarized)
            self.set_fill_color(*RED)
            self.rect(left_x, y, bar_w, bar_h, "F")

        # "Treated" label below
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(*RED)
        bars_bottom = panel_y + n_bars * (bar_h + bar_gap)
        self.set_xy(left_x, bars_bottom - 2)
        self.cell(panel_w, 10, "All collapsed to \"Treated\"", align="C")

        # Right panel: Continuous — bars proportional to dose
        right_x = margin * 2 + panel_w
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*NAVY)
        self.set_xy(right_x, panel_y - 16)
        self.cell(panel_w, 10, "Continuous: D > 0", align="C")

        # Violet gradient for varying doses
        for i, dose in enumerate(doses):
            y = panel_y + i * (bar_h + bar_gap)
            bar_w = panel_w * dose * 0.95
            # Darker violet for higher dose
            intensity = 0.4 + 0.6 * dose
            r = int(124 * intensity + 245 * (1 - intensity))
            g = int(58 * intensity + 240 * (1 - intensity))
            b = int(237 * intensity + 255 * (1 - intensity))
            self.set_fill_color(r, g, b)
            self.rect(right_x, y, bar_w, bar_h, "F")

        # "Dose preserved" label below
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(*VIOLET)
        self.set_xy(right_x, bars_bottom - 2)
        self.cell(panel_w, 10, "Dose variation preserved", align="C")

        # Callout box
        callout_y = bars_bottom + 14
        callout_margin = 35
        callout_w = WIDTH - callout_margin * 2
        self.set_fill_color(*VIOLET_TINT)
        self.set_draw_color(*VIOLET)
        self.set_line_width(0.8)
        self.rect(callout_margin, callout_y, callout_w, 34, "DF")
        self.set_xy(callout_margin + 8, callout_y + 5)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*VIOLET)
        self.multi_cell(
            callout_w - 16, 10,
            "TWFE with a continuous treatment is biased --\n"
            "negative weights, contamination, and scale dependence.",
            align="C",
        )

        # Gray annotations
        ann_y = callout_y + 42
        self.centered_text(
            ann_y,
            "Binarizing discards the dose-response relationship entirely",
            size=13, bold=False, color=GRAY,
        )
        self.centered_text(
            ann_y + 16,
            "You need level effects AND marginal effects",
            size=13, bold=False, color=GRAY,
        )

        self.add_footer()

    # ── Slide 3: The Solution ─────────────────────────────────────────

    def slide_solution(self):
        """Slide 3: Continuous DiD — the three-step procedure."""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(30, "Continuous DiD", size=36, color=NAVY)

        # Citation
        self.set_xy(0, 65)
        self.set_font("Helvetica", "I", 15)
        self.set_text_color(*GRAY)
        self.cell(
            WIDTH, 8,
            "Callaway, Goodman-Bacon & Sant'Anna (2024)  |  NBER WP 32117",
            align="C",
        )

        # Three numbered step boxes
        margin = 35
        box_width = WIDTH - margin * 2
        box_height = 42
        circle_r = 14
        step_y_start = 95
        total_step_unit = 60

        steps = [
            "Compute outcome changes relative to control group mean",
            "Fit B-spline regression of demeaned outcomes on dose",
            "Evaluate ATT(d) level effects and ACRT(d) marginal effects",
        ]

        for i, step_text in enumerate(steps):
            y = step_y_start + i * total_step_unit

            # Step number circle
            circle_x = margin + circle_r
            circle_y = y + box_height / 2

            self.set_fill_color(*VIOLET)
            self.ellipse(
                circle_x - circle_r, circle_y - circle_r,
                circle_r * 2, circle_r * 2, "F",
            )
            self.set_xy(circle_x - circle_r, circle_y - 6)
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(*WHITE)
            self.cell(circle_r * 2, 12, str(i + 1), align="C")

            # Step text box
            text_x = margin + circle_r * 2 + 10
            text_width = box_width - circle_r * 2 - 10

            self.set_fill_color(*WHITE)
            self.set_draw_color(*VIOLET)
            self.set_line_width(0.8)
            self.rect(text_x, y, text_width, box_height, "DF")

            self.set_font("Helvetica", "", 15)
            self.set_text_color(*NAVY)
            self.set_xy(text_x + 10, y + (box_height - 10) / 2)
            self.cell(text_width - 20, 10, step_text)

            # Downward arrow between steps (except after last)
            if i < len(steps) - 1:
                arrow_x = margin + circle_r
                arrow_top = y + box_height + 2
                arrow_bottom = y + total_step_unit - 2
                self.set_draw_color(*VIOLET)
                self.set_line_width(1.2)
                self.line(arrow_x, arrow_top, arrow_x, arrow_bottom)
                # Arrowhead
                head_size = 5
                self.line(
                    arrow_x - head_size, arrow_bottom - head_size,
                    arrow_x, arrow_bottom,
                )
                self.line(
                    arrow_x + head_size, arrow_bottom - head_size,
                    arrow_x, arrow_bottom,
                )

        # Footer text
        footer_y = step_y_start + 3 * total_step_unit + 2
        self.centered_text(
            footer_y,
            "From binary to the full dose-response curve.",
            size=15, bold=False, color=GRAY,
        )

        self.add_footer()

    # ── Slide 4: The Math ─────────────────────────────────────────────

    def slide_math(self):
        """Slide 4: Three equation boxes — B-spline OLS, ATT(d), ACRT(d)."""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(30, "The Math", size=38)
        self.centered_text(63, "Three targets, one estimation", size=18,
                           bold=False, color=GRAY)

        margin = 30
        box_w = WIDTH - margin * 2
        badge_w = 80
        badge_h = 18

        # Pre-render equations
        eq1_path, eq1_pw, eq1_ph = self._render_equations(
            [r"$\Delta\tilde{Y}_i = \psi^K(D_i)'\beta + \varepsilon_i$"]
        )
        eq2_path, eq2_pw, eq2_ph = self._render_equations(
            [r"$\mathrm{ATT}(d) = \psi^K(d)'\hat{\beta}$"]
        )
        eq3_path, eq3_pw, eq3_ph = self._render_equations(
            [r"$\mathrm{ACRT}(d) = \frac{\partial \psi^K(d)}{\partial d}"
             r" \cdot \hat{\beta}$"]
        )

        boxes = [
            {
                "badge": "B-Spline OLS",
                "eq": (eq1_path, eq1_pw, eq1_ph),
                "annotation": "(demeaned outcome ~ B-spline basis of dose)",
                "height": 50,
            },
            {
                "badge": "ATT(d)",
                "eq": (eq2_path, eq2_pw, eq2_ph),
                "annotation": "(level effect: total impact at dose d)",
                "height": 50,
            },
            {
                "badge": "ACRT(d)",
                "eq": (eq3_path, eq3_pw, eq3_ph),
                "annotation": "(marginal effect: response per unit of dose)",
                "height": 55,
            },
        ]

        y_cursor = 88
        box_gap = 10

        for box in boxes:
            bh = box["height"]

            # White box with violet border
            self.set_fill_color(*WHITE)
            self.set_draw_color(*VIOLET)
            self.set_line_width(0.8)
            self.rect(margin, y_cursor, box_w, bh, "DF")

            # Violet badge overlapping top edge
            badge_x = margin + 8
            badge_y = y_cursor - badge_h / 2
            self.set_fill_color(*VIOLET)
            self.rect(badge_x, badge_y, badge_w, badge_h, "F")
            self.set_xy(badge_x, badge_y + 3)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*WHITE)
            self.cell(badge_w, 12, box["badge"], align="C")

            # Determine content region
            content_top = y_cursor + badge_h / 2 + 2
            if box["annotation"]:
                ann_y = y_cursor + bh - 14
                content_bottom = ann_y - 2
            else:
                content_bottom = y_cursor + bh - 6

            # Place equation image
            eq_path, eq_pw, eq_ph = box["eq"]
            self._place_equation(
                eq_path, eq_pw, eq_ph,
                margin, y_cursor, box_w,
                content_top, content_bottom,
            )

            # Annotation text
            if box["annotation"]:
                self.set_xy(margin, ann_y)
                self.set_font("Helvetica", "I", 12)
                self.set_text_color(*GRAY)
                self.cell(box_w, 10, box["annotation"], align="C")

            y_cursor += bh + box_gap

        # Below boxes
        self.centered_text(
            y_cursor + 2,
            "Influence-function SEs + multiplier bootstrap",
            size=14, bold=True, color=VIOLET,
        )

        self.add_footer()

    # ── Slide 5: Dose-Response Curve ──────────────────────────────────

    def slide_dose_response(self):
        """Slide 5: Visual centerpiece — matplotlib dose-response figure."""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(25, "Dose-Response", size=38, color=VIOLET)
        self.centered_text(58, "Curve", size=38, color=VIOLET)

        # Render and place the figure
        fig_path, fig_pw, fig_ph = self._render_dose_response_figure()
        fig_margin = 25
        fig_w = WIDTH - fig_margin * 2
        fig_aspect = fig_ph / fig_pw
        fig_h = fig_w * fig_aspect
        fig_y = 80
        self.image(fig_path, fig_margin, fig_y, fig_w)

        # Annotations below figure
        ann_y = fig_y + fig_h + 8
        self.centered_text(
            ann_y,
            "ATT(d): total impact at each dose level",
            size=16, bold=True, color=NAVY,
        )
        self.centered_text(
            ann_y + 20,
            "Confidence bands from influence functions or bootstrap",
            size=14, bold=False, color=GRAY,
        )

        self.add_footer()

    # ── Slide 6: Two Questions, One Estimator ─────────────────────────

    def slide_two_questions(self):
        """Slide 6: ATT(d) vs ACRT(d) side-by-side panels."""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(25, "Two Questions", size=38, color=VIOLET)
        self.centered_text(58, "One Estimator", size=38, color=VIOLET)

        # Two side-by-side panels — compact to fill without dead space
        margin = 25
        gap = 12
        panel_w = (WIDTH - margin * 2 - gap) / 2
        panel_h = 115
        panel_y = 95

        panels = [
            {
                "x": margin,
                "border_color": VIOLET,
                "title": "ATT(d)",
                "subtitle": "Level Effect",
                "question": "What is the total impact\nat dose d?",
                "example": "A $500 subsidy reduces\nemissions by 12 tons",
            },
            {
                "x": margin + panel_w + gap,
                "border_color": MID_BLUE,
                "title": "ACRT(d)",
                "subtitle": "Marginal Effect",
                "question": "What is the return to one\nmore unit of dose?",
                "example": "Each additional $100 reduces\nemissions by 1.8 tons",
            },
        ]

        for panel in panels:
            px = panel["x"]

            # Panel box
            self.set_fill_color(*WHITE)
            self.set_draw_color(*panel["border_color"])
            self.set_line_width(1.2)
            self.rect(px, panel_y, panel_w, panel_h, "DF")

            # Title
            self.set_xy(px, panel_y + 8)
            self.set_font("Helvetica", "B", 26)
            self.set_text_color(*panel["border_color"])
            self.cell(panel_w, 14, panel["title"], align="C")

            # Subtitle
            self.set_xy(px, panel_y + 30)
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(*NAVY)
            self.cell(panel_w, 10, panel["subtitle"], align="C")

            # Horizontal rule
            rule_y = panel_y + 46
            self.set_draw_color(*panel["border_color"])
            self.set_line_width(0.4)
            self.line(px + 15, rule_y, px + panel_w - 15, rule_y)

            # Question
            self.set_xy(px + 8, panel_y + 52)
            self.set_font("Helvetica", "I", 14)
            self.set_text_color(*GRAY)
            self.multi_cell(panel_w - 16, 13, panel["question"], align="C")

            # Example
            self.set_xy(px + 8, panel_y + 82)
            self.set_font("Helvetica", "", 13)
            self.set_text_color(*NAVY)
            self.multi_cell(panel_w - 16, 13, panel["example"], align="C")

        # Callout below panels
        callout_y = panel_y + panel_h + 15
        callout_margin = 40
        callout_w = WIDTH - callout_margin * 2
        self.set_fill_color(*VIOLET_TINT)
        self.set_draw_color(*VIOLET)
        self.set_line_width(0.8)
        self.rect(callout_margin, callout_y, callout_w, 26, "DF")
        self.set_xy(callout_margin, callout_y + 6)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*VIOLET)
        self.cell(
            callout_w, 12,
            "Both are functions of d -- not just single numbers.",
            align="C",
        )

        # Annotation
        self.centered_text(
            callout_y + 34,
            "Plus global summaries: overall ATT and overall ACRT",
            size=14, bold=False, color=GRAY,
        )

        self.add_footer()

    # ── Slide 7: Assumptions ─────────────────────────────────────────

    def slide_assumptions(self):
        """Slide 7: PT vs SPT — what you need to believe."""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(25, "What You Need", size=38, color=VIOLET)
        self.centered_text(58, "to Believe", size=38, color=VIOLET)

        # Two assumption rows — PT (standard) and SPT (strong)
        margin = 28
        row_w = WIDTH - margin * 2
        row_h = 68
        row_gap = 10
        row_y = 95

        rows = [
            {
                "badge": "Standard PT",
                "badge_color": MID_BLUE,
                "headline": "Parallel trends in untreated outcomes",
                "identifies": "ATT(d|d) level effects within each dose group",
                "meaning": "Counterfactual trends are the same across\n"
                           "all dose groups and the untreated.",
            },
            {
                "badge": "Strong PT",
                "badge_color": VIOLET,
                "headline": "No selection into dose based on effects",
                "identifies": "ATT(d) + ACRT(d) dose-response curves",
                "meaning": "Units don't choose their dose based on\n"
                           "how much they would benefit from it.",
            },
        ]

        for i, row in enumerate(rows):
            y = row_y + i * (row_h + row_gap)

            # Row box
            self.set_fill_color(*WHITE)
            self.set_draw_color(*row["badge_color"])
            self.set_line_width(1.0)
            self.rect(margin, y, row_w, row_h, "DF")

            # Badge overlapping top edge
            badge_w = 80
            badge_h = 18
            badge_x = margin + 8
            badge_y = y - badge_h / 2
            self.set_fill_color(*row["badge_color"])
            self.rect(badge_x, badge_y, badge_w, badge_h, "F")
            self.set_xy(badge_x, badge_y + 3)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*WHITE)
            self.cell(badge_w, 12, row["badge"], align="C")

            # Headline
            self.set_xy(margin + 12, y + 12)
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(*NAVY)
            self.cell(row_w - 24, 10, row["headline"])

            # Identifies label
            self.set_xy(margin + 12, y + 28)
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(*row["badge_color"])
            self.cell(row_w - 24, 10,
                      "Identifies:  " + row["identifies"])

            # Meaning
            self.set_xy(margin + 12, y + 43)
            self.set_font("Helvetica", "", 14)
            self.set_text_color(*GRAY)
            self.multi_cell(row_w - 24, 11, row["meaning"])

        # Key insight callout
        callout_y = row_y + 2 * (row_h + row_gap) + 5
        callout_margin = 32
        callout_w = WIDTH - callout_margin * 2
        self.set_fill_color(*VIOLET_TINT)
        self.set_draw_color(*VIOLET)
        self.set_line_width(0.8)
        self.rect(callout_margin, callout_y, callout_w, 34, "DF")
        self.set_xy(callout_margin + 8, callout_y + 4)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*VIOLET)
        self.multi_cell(
            callout_w - 16, 12,
            "Note: Under standard PT, the slope of the dose-response\n"
            "curve does NOT identify the causal marginal effect.",
            align="C",
        )

        self.add_footer()

    # ── Slide 8: Code Example ─────────────────────────────────────────

    def slide_code(self):
        """Slide 8: Drop-in API code example."""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(30, "Drop-in", size=36)
        self.centered_text(60, "API", size=36, color=VIOLET)

        margin = 30
        code_y = 95
        code_lines = [
            ("from diff_diff import ContinuousDiD", 1.0),
            ("", 0.5),
            ("est = ContinuousDiD(seed=42)", 1.0),
            ("results = est.fit(", 1.0),
            ("    data,", 1.0),
            ("    outcome='outcome',", 1.0),
            ("    unit='unit',", 1.0),
            ("    time='period',", 1.0),
            ("    first_treat='first_treat',", 1.0),
            ("    dose='dose',", 1.0),
            ("    aggregate='dose',", 1.0),
            (")", 1.0),
            ("", 0.5),
            ("results.overall_att    # level effect", 1.0),
            ("results.overall_acrt   # marginal effect", 1.0),
        ]
        line_height = 11
        total_lines = sum(h for _, h in code_lines)
        code_height = total_lines * line_height + 20

        self.set_fill_color(*DARK_SLATE)
        self.rect(margin, code_y, WIDTH - margin * 2, code_height, "F")

        self.set_font("Courier", "", 13)
        self.set_text_color(*WHITE)
        cumulative_y = 0.0
        for line_text, height_mult in code_lines:
            self.set_xy(margin + 15, code_y + 10 + cumulative_y)
            self.cell(0, 10, line_text)
            cumulative_y += line_height * height_mult

        subtitle_y = code_y + code_height + 12
        self.centered_text(
            subtitle_y,
            "Same fit() API as every other diff-diff estimator.",
            size=15, bold=False, color=GRAY,
        )
        self.centered_text(
            subtitle_y + 17,
            "Full walkthrough in Tutorial 14: Continuous DiD",
            size=15, bold=False, color=GRAY,
        )

        self.add_footer()

    # ── Slide 9: Use Cases ────────────────────────────────────────────

    def slide_use_cases(self):
        """Slide 9: Where Dose Matters — 2x2 grid of use-case cards."""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(25, "Where Dose", size=38, color=VIOLET)
        self.centered_text(58, "Matters", size=38, color=VIOLET)

        margin = 28
        gap = 10
        card_w = (WIDTH - margin * 2 - gap) / 2
        card_h = 68
        grid_y = 95

        cards = [
            {
                "title": "Job Training",
                "desc": "Training hours as dose. How do\nearnings respond to each hour?",
            },
            {
                "title": "Minimum Wage",
                "desc": "Different increases across states.\nEmployment effect per dollar?",
            },
            {
                "title": "Subsidies",
                "desc": "Varying grant amounts. What is the\nmarginal return to spending?",
            },
            {
                "title": "Pollution Exposure",
                "desc": "Distance from source as dose.\nHow does health vary with proximity?",
            },
        ]

        for i, card in enumerate(cards):
            col = i % 2
            row = i // 2
            x = margin + col * (card_w + gap)
            y = grid_y + row * (card_h + gap)

            # Card box
            self.set_fill_color(*WHITE)
            self.set_draw_color(*VIOLET)
            self.set_line_width(0.8)
            self.rect(x, y, card_w, card_h, "DF")

            # Card title
            self.set_xy(x + 10, y + 8)
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(*VIOLET)
            self.cell(card_w - 20, 12, card["title"])

            # Card description
            self.set_xy(x + 10, y + 24)
            self.set_font("Helvetica", "", 14)
            self.set_text_color(*NAVY)
            self.multi_cell(card_w - 20, 12, card["desc"], align="L")

        # Subtitle below cards
        cards_bottom = grid_y + 2 * (card_h + gap)
        self.centered_text(
            cards_bottom + 6,
            "Any setting where treatment intensity varies across units.",
            size=15, bold=False, color=GRAY,
        )

        self.add_footer()

    # ── Slide 10: Full Toolkit ────────────────────────────────────────

    def slide_full_toolkit(self):
        """Slide 10: Every Method You Need — 6x2 grid (11 methods, last centered)."""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(20, "Every Method", size=36)
        self.centered_text(50, "You Need", size=36, color=MID_BLUE)

        margin = 25
        box_width = (WIDTH - margin * 3) / 2
        box_height = 34
        gap_y = 3
        y_start = 80

        methods = [
            ("Basic DiD / TWFE", "Classic 2x2 and panel", False),
            ("Callaway-Sant'Anna", "Staggered adoption (2021)", False),
            ("Sun-Abraham", "Interaction-weighted (2021)", False),
            ("Imputation DiD", "Borusyak et al. (2024)", False),
            ("Two-Stage DiD", "Gardner (2022)", False),
            ("Stacked DiD", "Wing et al. (2024)", False),
            ("Continuous DiD", "Callaway et al. (2024)", True),
            ("Synthetic DiD", "Arkhangelsky et al. (2021)", False),
            ("Triple Difference", "DDD with proper covariates", False),
            ("Honest DiD", "Rambachan-Roth sensitivity", False),
            ("Bacon Decomposition", "TWFE diagnostic weights", False),
        ]

        for i, (title, desc, is_new) in enumerate(methods):
            if i < 10:
                col = i % 2
                row = i // 2
                x = margin + col * (box_width + margin)
            else:
                row = 5
                x = (WIDTH - box_width) / 2

            y = y_start + row * (box_height + gap_y)

            self.set_fill_color(*WHITE)
            if is_new:
                self.set_draw_color(*VIOLET)
                self.set_line_width(1.2)
            else:
                self.set_draw_color(*MID_BLUE)
                self.set_line_width(0.8)
            self.rect(x, y, box_width, box_height, "DF")

            # Title
            self.set_xy(x + 5, y + 3)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*VIOLET if is_new else MID_BLUE)
            display_title = title + "  [NEW]" if is_new else title
            self.cell(box_width - 10, 10, display_title, align="C")

            # Description
            self.set_xy(x + 5, y + 19)
            self.set_font("Helvetica", "", 11)
            self.set_text_color(*GRAY)
            self.cell(box_width - 10, 10, desc, align="C")

        # Subtitle below grid
        grid_bottom = y_start + 6 * (box_height + gap_y)
        self.centered_text(
            grid_bottom + 2,
            "The most complete DiD toolkit in any language.",
            size=15, bold=False, color=GRAY,
        )
        self.add_footer()

    # ── Slide 11: CTA ─────────────────────────────────────────────────

    def slide_cta(self):
        """Slide 11: Upgrade to v2.6."""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(45, "Upgrade to", size=38)
        self.centered_text(78, "v2.6", size=38, color=VIOLET)

        box_width = 210
        box_x = (WIDTH - box_width) / 2
        box_y = 125
        box_h = 36
        self.set_fill_color(*MID_BLUE)
        self.rect(box_x, box_y, box_width, box_h, "F")

        self.set_xy(box_x, box_y + 10)
        self.set_font("Courier", "B", 15)
        self.set_text_color(*WHITE)
        self.cell(box_width, 14, "$ pip install --upgrade diff-diff", align="C")

        self.centered_text(200, "github.com/igerber/diff-diff", size=20,
                           color=MID_BLUE)

        self.centered_text(
            232, "Full documentation & 14 tutorials included",
            size=16, bold=False, color=GRAY,
        )
        self.centered_text(
            252, "MIT Licensed  |  Open Source",
            size=16, bold=False, color=GRAY,
        )

        self.draw_split_logo(278, size=28)

        self.centered_text(
            298, "Difference-in-Differences for Python",
            size=14, bold=False, color=GRAY,
        )


def main():
    pdf = CarouselV26PDF()

    pdf.slide_hook()
    pdf.slide_problem()
    pdf.slide_solution()
    pdf.slide_math()
    pdf.slide_dose_response()
    pdf.slide_two_questions()
    pdf.slide_assumptions()
    pdf.slide_code()
    pdf.slide_use_cases()
    pdf.slide_full_toolkit()
    pdf.slide_cta()

    output_path = Path(__file__).parent / "diff-diff-v26-carousel.pdf"
    pdf.output(str(output_path))
    print(f"PDF saved to: {output_path}")

    pdf.cleanup()


if __name__ == "__main__":
    main()
