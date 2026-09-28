#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF announcing diff-diff v3.0 survey support."""

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from fpdf import FPDF  # noqa: E402

# Computer Modern for math
plt.rcParams["mathtext.fontset"] = "cm"

# Page dimensions (4:5 portrait)
WIDTH = 270     # mm
HEIGHT = 337.5  # mm

# ── Paper/scholarly palette ──────────────────────────────────────
IVORY = (253, 251, 247)         # #FDFBF7 — page background
SIENNA = (180, 83, 9)           # #B45309 — primary accent
STONE_900 = (28, 25, 23)        # #1C1917 — body text
DARK_RED = (153, 27, 27)        # #991B1B — problem accent
STONE_500 = (120, 113, 108)     # #78716C — secondary text
STONE_300 = (214, 211, 209)     # #D6D3D1 — borders/rules
CODE_BG = (41, 37, 36)          # #292524 — code block bg
CALLOUT_BG = (254, 249, 243)    # #FEF9F3 — callout box bg
WHITE = (255, 255, 255)
GREEN_CHECK = (22, 163, 74)     # #16A34A
RED_X = (185, 28, 28)           # #B91C1C

# Code syntax colors (on dark warm bg)
CODE_GOLD = (245, 208, 123)     # #F5D07B — class names
CODE_GREEN = (134, 239, 172)    # #86EFAC — string literals

# Hex colors for matplotlib
IVORY_HEX = "#FDFBF7"
SIENNA_HEX = "#B45309"
STONE_900_HEX = "#1C1917"
DARK_RED_HEX = "#991B1B"
STONE_500_HEX = "#78716C"
STONE_300_HEX = "#D6D3D1"
GREEN_CHECK_HEX = "#16A34A"
RED_X_HEX = "#B91C1C"


class V3SurveyCarouselPDF(FPDF):
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

    # ── Background & Footer ──────────────────────────────────────

    def ivory_background(self):
        """Draw flat warm ivory background."""
        self.set_fill_color(*IVORY)
        self.rect(0, 0, WIDTH, HEIGHT, "F")

    def add_footer(self):
        """Add footer with stone rule and version text."""
        rule_y = HEIGHT - 28
        self.set_draw_color(*STONE_300)
        self.set_line_width(0.5)
        self.line(50, rule_y, WIDTH - 50, rule_y)

        self.set_font("Helvetica", "B", 12)
        dd_text = "diff-diff "
        v_text = "v3.0"
        dd_w = self.get_string_width(dd_text)
        v_w = self.get_string_width(v_text)
        start_x = (WIDTH - dd_w - v_w) / 2

        self.set_xy(start_x, HEIGHT - 22)
        self.set_text_color(*STONE_500)
        self.cell(dd_w, 10, dd_text)
        self.set_text_color(*SIENNA)
        self.cell(v_w, 10, v_text)

    # ── Text Helpers ─────────────────────────────────────────────

    def centered_text(self, y, text, size=28, bold=True, color=STONE_900,
                      italic=False):
        """Add centered text."""
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
        """Draw the split-color diff-diff logo."""
        self.set_xy(0, y)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*STONE_900)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="R")
        self.set_text_color(*SIENNA)
        self.cell(10, 10, "-", align="C")
        self.set_text_color(*STONE_900)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="L")

    # ── Equation Rendering ───────────────────────────────────────

    def _render_equations(self, latex_lines, fontsize=26, color=STONE_900_HEX):
        """Render LaTeX equations to transparent PNG."""
        n = len(latex_lines)
        fig_h = max(0.7, 0.55 * n + 0.15)
        fig = plt.figure(figsize=(10, fig_h))

        for i, line in enumerate(latex_lines):
            y_frac = 1.0 - (2 * i + 1) / (2 * n)
            fig.text(
                0.5, y_frac, line,
                fontsize=fontsize, ha="center", va="center",
                color=color,
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

    def _place_equation_centered(self, path, pw, ph, y, max_w=200):
        """Place equation image centered on page at given y."""
        aspect = ph / pw
        display_w = min(max_w, WIDTH * 0.75)
        display_h = display_w * aspect
        eq_x = (WIDTH - display_w) / 2
        self.image(path, eq_x, y, display_w)
        return display_h

    # ── Matplotlib Visuals ───────────────────────────────────────

    def _render_deff_comparison(self):
        """Render paired CI bars: naive vs design-based SEs."""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor(IVORY_HEX)
        ax.set_facecolor(IVORY_HEX)

        att = 1.2  # point estimate

        # Naive SEs — too narrow, falsely excludes zero
        naive_lo, naive_hi = 0.3, 2.1
        naive_y = 2.0
        ax.plot([naive_lo, naive_hi], [naive_y, naive_y],
                color=DARK_RED_HEX, linewidth=8, solid_capstyle="round")
        ax.plot(att, naive_y, "o", color=DARK_RED_HEX, markersize=14,
                zorder=5)
        ax.text(-2.6, naive_y + 0.05, "Naive SEs", color=DARK_RED_HEX,
                fontsize=15, ha="left", va="center", fontweight="bold",
                fontfamily="sans-serif")
        ax.text(naive_hi + 0.3, naive_y + 0.05, '"Significant"',
                color=DARK_RED_HEX, fontsize=12, ha="left", va="center",
                style="italic", fontfamily="sans-serif")

        # Design-based SEs — correct width, includes zero
        design_lo, design_hi = -0.8, 3.2
        design_y = 0.8
        ax.plot([design_lo, design_hi], [design_y, design_y],
                color=SIENNA_HEX, linewidth=8, solid_capstyle="round")
        ax.plot(att, design_y, "o", color=SIENNA_HEX, markersize=14,
                zorder=5)
        ax.text(-2.6, design_y + 0.05, "Design-based", color=SIENNA_HEX,
                fontsize=15, ha="left", va="center", fontweight="bold",
                fontfamily="sans-serif")
        ax.text(design_hi + 0.3, design_y + 0.05, "Not significant",
                color=SIENNA_HEX, fontsize=12, ha="left", va="center",
                style="italic", fontfamily="sans-serif")

        # Zero reference line
        ax.axvline(0, color=STONE_500_HEX, linewidth=1.2, linestyle="--",
                   alpha=0.7)
        ax.text(0, 0.15, "0", ha="center", va="top", fontsize=11,
                color=STONE_500_HEX, fontfamily="sans-serif")

        ax.set_xlim(-3.0, 5.0)
        ax.set_ylim(0.0, 2.8)
        ax.axis("off")

        fig.tight_layout(pad=0.3)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1,
                    facecolor=IVORY_HEX)
        plt.close(fig)

        with PILImage.open(path) as img:
            pw, ph = img.size

        self._temp_files.append(path)
        return path, pw, ph

    def _render_gap_table(self):
        """Render competitive gap table with Unicode checkmarks."""
        fig, ax = plt.subplots(figsize=(7.5, 2.8))
        fig.patch.set_facecolor(IVORY_HEX)
        ax.set_facecolor(IVORY_HEX)
        ax.axis("off")

        # Column positions (normalized 0-1)
        col_label = 0.02
        col1 = 0.46   # Strata
        col2 = 0.62   # FPC
        col3 = 0.78   # Replicate Wts

        # Column headers
        header_y = 0.88
        header_props = dict(fontsize=12, ha="center", va="center",
                            color=STONE_500_HEX, fontweight="bold",
                            fontfamily="sans-serif")
        ax.text(col1, header_y, "Strata", **header_props)
        ax.text(col2, header_y, "FPC", **header_props)
        ax.text(col3, header_y, "Replicate\nWeights",
                fontsize=12, ha="center", va="center",
                color=STONE_500_HEX, fontweight="bold",
                fontfamily="sans-serif", linespacing=0.9)

        # Header separator line
        ax.axhline(y=0.75, xmin=0.01, xmax=0.99,
                   color=STONE_300_HEX, linewidth=0.8)

        # Rows
        rows = [
            ("diff-diff v3", [True, True, True], 0.58),
            ("R did", [False, False, False], 0.38),
            ("Stata csdid", [False, False, False], 0.18),
        ]

        for label, checks, row_y in rows:
            # Row label
            is_diffdiff = "diff-diff" in label
            ax.text(col_label, row_y, label,
                    fontsize=13 if is_diffdiff else 12,
                    ha="left", va="center",
                    fontweight="bold" if is_diffdiff else "normal",
                    color=SIENNA_HEX if is_diffdiff else STONE_900_HEX,
                    fontfamily="sans-serif")

            for col_x, supported in zip([col1, col2, col3], checks):
                if supported:
                    ax.text(col_x, row_y, "\u2713",
                            fontsize=18, ha="center", va="center",
                            color=GREEN_CHECK_HEX, fontweight="bold")
                else:
                    ax.text(col_x, row_y, "\u2717",
                            fontsize=18, ha="center", va="center",
                            color=RED_X_HEX, fontweight="bold")

        fig.tight_layout(pad=0.2)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08,
                    facecolor=IVORY_HEX)
        plt.close(fig)

        with PILImage.open(path) as img:
            pw, ph = img.size

        self._temp_files.append(path)
        return path, pw, ph

    # ── Code Block ───────────────────────────────────────────────

    def _add_code_block(self, x, y, w, token_lines, font_size=12,
                        line_height=11):
        """Render syntax-highlighted code on a dark warm panel."""
        n_lines = len(token_lines)
        total_h = n_lines * line_height + 24

        self.set_fill_color(*CODE_BG)
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

    # ════════════════════════════════════════════════════════════
    # SLIDES
    # ════════════════════════════════════════════════════════════

    def slide_01_hook(self):
        """Slide 1: Hook — The first DiD library with design-based survey inference.

        Claims & sources:
        - "First DiD library": docs/methodology/survey-theory.md line 149:
          "diff-diff is the only package --- across R, Stata, and Python ---
          that provides design-based variance estimation"
        - "All 16 estimators": 15 inference estimators + BaconDecomposition
          (diagnostic). Bacon accepts survey_design for weighted means only,
          does not support replicate weights (bacon.py line 467-470).
        - Teasers verified: SurveyDesign params (survey.py:58-61),
          replicate methods (survey.py), test_survey_real_data.py
        """
        self.add_page()
        self.ivory_background()

        self.draw_split_logo(50, size=56)

        # v3.0 in sienna
        self.centered_text(115, "v3.0", size=48, color=SIENNA)

        # Main claim — two lines
        self.centered_text(165, "The first DiD library with", size=24)
        self.centered_text(190, "design-based survey inference.", size=24,
                           color=SIENNA)

        # Teasers
        teasers = [
            "Strata + PSU + FPC",
            "TSL, replicate weights, survey bootstrap",
            "Survey support across all 16 estimators*",
            "Validated against R's survey package",
        ]
        y_start = 225
        for i, teaser in enumerate(teasers):
            self.set_xy(0, y_start + i * 18)
            self.set_font("Helvetica", "", 15)
            self.set_text_color(*STONE_500)
            self.cell(WIDTH, 10, teaser, align="C")

        # Bacon asterisk fine print
        self.set_xy(0, y_start + len(teasers) * 18 + 4)
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(*STONE_300)
        self.cell(WIDTH, 10,
                  "*Variance paths vary by estimator. See docs for support matrix.",
                  align="C")

        self.add_footer()

    def slide_02_problem(self):
        """Slide 2: The Forced Tradeoff.

        Claims & sources:
        - Competitive gap: survey-theory.md Section 1.3 (lines 92-111)
          documents R did (weightsname only, no strata/FPC),
          Stata csdid (pweight only, no svy: prefix, no strata/FPC).
        - Footnote cites actual package docs, not our own document.
        """
        self.add_page()
        self.ivory_background()

        self.centered_text(35, "The Forced", size=38)
        self.centered_text(68, "Tradeoff", size=38, color=DARK_RED)

        # Problem statement
        self.centered_text(108, "Correct DiD method with wrong SEs,",
                           size=17, bold=False, color=STONE_500)
        self.centered_text(128, "or correct SEs with naive DiD.",
                           size=17, bold=False, color=STONE_500)

        # Gap table (matplotlib rendered)
        table_path, tpw, tph = self._render_gap_table()
        table_w = WIDTH - 70
        table_aspect = tph / tpw
        table_h = table_w * table_aspect
        table_x = (WIDTH - table_w) / 2
        table_y = 155

        # Card background for table
        card_pad = 6
        self.set_fill_color(*CALLOUT_BG)
        self.set_draw_color(*STONE_300)
        self.set_line_width(0.8)
        self.rect(table_x - card_pad, table_y - card_pad,
                  table_w + card_pad * 2, table_h + card_pad * 2, "DF")

        self.image(table_path, table_x, table_y, table_w)

        # Footnote citing package docs
        footnote_y = table_y + table_h + card_pad + 6
        self.set_xy(0, footnote_y)
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(*STONE_300)
        self.cell(WIDTH, 10,
                  "All packages support cluster-robust inference.",
                  align="C")
        self.set_xy(0, footnote_y + 12)
        self.cell(WIDTH, 10,
                  "R did v2.1 CRAN; Stata help csdid (Rios-Avila et al.)",
                  align="C")

        self.add_footer()

    def slide_03_deff_visual(self):
        """Slide 3: The Design Effect — paired CI bars.

        Claims & sources:
        - "2-5x variance inflation": survey-theory.md Section 1.1
        - Synthetic data for illustration. ATT=1.2, naive CI [0.3, 2.1]
          (excludes zero), design-based CI [-0.8, 3.2] (includes zero).
        """
        self.add_page()
        self.ivory_background()

        self.centered_text(28, "The Design Effect", size=36)
        self.centered_text(62, "Same ATT, Different Conclusions",
                           size=18, bold=False, italic=True, color=STONE_500)

        # DEFF comparison plot
        plot_path, ppw, pph = self._render_deff_comparison()
        plot_w = WIDTH * 0.88
        plot_aspect = pph / ppw
        plot_h = plot_w * plot_aspect
        plot_x = (WIDTH - plot_w) / 2
        plot_y = 90
        self.image(plot_path, plot_x, plot_y, plot_w)

        # Annotation below plot
        ann_y = plot_y + plot_h + 12
        self.centered_text(ann_y,
                           "Ignoring survey design doesn't just",
                           size=16, bold=False, color=STONE_900)
        self.centered_text(ann_y + 20,
                           "affect precision -- it changes your conclusions.",
                           size=16, bold=True, color=SIENNA)

        self.add_footer()

    def slide_04_theory(self):
        """Slide 4: Why It Works — Binder's theorem applied to DiD IFs.

        Claims & sources:
        - Equation: IF-based form from survey-theory.md Section 4
          (lines 415-425). General version for all estimators.
          Regression-specific score-total form (T_hj) is in Section 5.
        - Three-step argument: survey-theory.md Section 4.1-4.4
        - Binder (1983): "On the variances of asymptotically normal
          estimators from complex surveys"
        - Demnati & Rao (2004): "Linearization variance estimators for
          survey data"
        """
        self.add_page()
        self.ivory_background()

        self.centered_text(28, "Why It Works", size=36)
        self.centered_text(62,
                           "Binder (1983) applied to DiD influence functions",
                           size=16, bold=False, italic=True, color=STONE_500)

        # Binder variance formula — IF-based form
        eq_path, epw, eph = self._render_equations(
            [r"$\hat{V}(\hat{\theta}) = \sum_h (1 - f_h) \,"
             r"\frac{n_h}{n_h - 1} \,"
             r"\sum_{j=1}^{n_h} (\psi_{hj} - \bar{\psi}_h)^2$"],
            fontsize=22,
        )
        eq_h = self._place_equation_centered(eq_path, epw, eph, 92,
                                             max_w=220)

        # PSU-level IF total — render as "where" text + inline equation
        self.centered_text(92 + eq_h + 8,
                           "where",
                           size=15, bold=False, italic=True, color=STONE_500)

        eq2_path, e2pw, e2ph = self._render_equations(
            [r"$\psi_{hj} = \sum_{i \in \mathrm{PSU}\, j,"
             r"\; \mathrm{stratum}\, h}"
             r"\; \psi_i$"],
            fontsize=16,
        )
        eq2_y = 92 + eq_h + 20
        eq2_h = self._place_equation_centered(eq2_path, e2pw, e2ph, eq2_y,
                                              max_w=120)

        # Reasoning chain — scoped to IF-amenable estimators
        margin = 42
        y_cursor = eq2_y + eq2_h + 14
        items = [
            "Most modern DiD estimators are smooth functionals of F",
            "Their IFs are well-defined and design-independent",
            "Binder's theorem: plug IFs into the survey variance formula",
            "SyntheticDiD and TROP: Rao-Wu survey bootstrap instead",
        ]

        for text in items:
            # Sienna dash
            self.set_xy(margin, y_cursor)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*SIENNA)
            self.cell(14, 10, "--")

            # Text
            self.set_xy(margin + 14, y_cursor)
            self.set_font("Helvetica", "", 13)
            self.set_text_color(*STONE_900)
            self.cell(WIDTH - margin * 2 - 14, 10, text)

            y_cursor += 20

        # Citation
        self.centered_text(y_cursor + 6,
                           "Binder (1983), Demnati & Rao (2004)",
                           size=12, bold=False, italic=True, color=STONE_500)

        # Link to full derivation (GitHub URL)
        self.centered_text(y_cursor + 20,
                           "github.com/igerber/diff-diff/blob/main/docs/methodology/survey-theory.md",
                           size=9, bold=False, italic=True, color=SIENNA)

        self.add_footer()

    def slide_05_supported(self):
        """Slide 5: What's Supported — feature cards + estimator grid.

        Claims & sources:
        - TSL: survey.py compute_survey_vcov()
        - Replicate weights (BRR, Fay, JK1, JKn, SDR): survey.py
          _replicate_variance()
        - Survey bootstrap: staggered_bootstrap.py, sdid.py, trop.py
        - 16 estimators: __init__.py
        """
        self.add_page()
        self.ivory_background()

        self.centered_text(28, "What's Supported", size=36)

        # Three feature cards with sienna left accent bars
        margin = 35
        box_w = WIDTH - margin * 2
        box_h = 40
        gap = 5
        start_y = 68
        bar_w = 4

        features = [
            ("Taylor Series Linearization",
             "Strata + PSU + FPC + lonely PSU handling"),
            ("Replicate Weights",
             "BRR, Fay, JK1, JKn, SDR -- five methods"),
            ("Survey-Aware Bootstrap",
             "Design structure preserved in resampling"),
        ]

        for i, (title, desc) in enumerate(features):
            by = start_y + i * (box_h + gap)

            # Card
            self.set_fill_color(*WHITE)
            self.set_draw_color(*STONE_300)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, box_h, "DF")

            # Sienna accent bar
            self.set_fill_color(*SIENNA)
            self.rect(margin, by, bar_w, box_h, "F")

            # Title
            self.set_xy(margin + bar_w + 12, by + 8)
            self.set_font("Helvetica", "B", 17)
            self.set_text_color(*STONE_900)
            self.cell(box_w - bar_w - 24, 10, title)

            # Description
            self.set_xy(margin + bar_w + 12, by + 24)
            self.set_font("Helvetica", "", 13)
            self.set_text_color(*STONE_500)
            self.cell(box_w - bar_w - 24, 10, desc)

        # Bold callout
        callout_y = start_y + 3 * (box_h + gap) + 8
        self.centered_text(callout_y, "Survey support across all 16 estimators.*",
                           size=20, color=SIENNA)

        # Asterisk fine print
        self.centered_text(callout_y + 26,
                           "*Variance paths vary by estimator. See docs for support matrix.",
                           size=10, bold=False, italic=True, color=STONE_300)

        self.add_footer()

    def slide_06_validation(self):
        """Slide 6: Validated Against R.

        Claims & sources:
        - Precision "< 1e-10": tests/test_survey_real_data.py line 40:
          "observed gaps are < 1e-10, so 1e-8 guards against"
        - API dataset: test_survey_real_data.py tests A1-A7
        - NHANES: test_survey_real_data.py tests B1-B4
        - RECS 2020: test_survey_real_data.py tests C1-C2
        """
        self.add_page()
        self.ivory_background()

        self.centered_text(28, "Validated Against R", size=36)
        self.centered_text(62,
                           "Cross-validated against R's survey package",
                           size=17, bold=False, color=STONE_500)

        # Three dataset cards with sienna accent bar + precision badge
        margin = 35
        box_w = WIDTH - margin * 2
        box_h = 46
        gap = 5
        start_y = 86
        bar_w = 4

        datasets = [
            ("API Dataset",
             "TSL with strata, FPC, Fay's BRR replicates",
             "< 1e-10"),
            ("NHANES",
             "TSL with strata + PSU + nest=TRUE",
             "< 1e-10"),
            ("RECS 2020",
             "JK1 replicate weights, 60 replicate columns",
             "< 1e-10"),
        ]

        for i, (title, desc, precision) in enumerate(datasets):
            by = start_y + i * (box_h + gap)

            # Card
            self.set_fill_color(*WHITE)
            self.set_draw_color(*STONE_300)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, box_h, "DF")

            # Sienna accent bar
            self.set_fill_color(*SIENNA)
            self.rect(margin, by, bar_w, box_h, "F")

            # Title
            self.set_xy(margin + bar_w + 12, by + 8)
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(*STONE_900)
            self.cell(box_w - bar_w - 24, 10, title)

            # Description
            self.set_xy(margin + bar_w + 12, by + 26)
            self.set_font("Helvetica", "", 13)
            self.set_text_color(*STONE_500)
            self.cell(box_w - bar_w - 24, 10, desc)

            # Precision badge
            badge_w = 60
            badge_h = 20
            badge_x = margin + box_w - badge_w - 8
            badge_y = by + 8
            self.set_fill_color(*SIENNA)
            self.rect(badge_x, badge_y, badge_w, badge_h, "F")
            self.set_xy(badge_x, badge_y + 3)
            self.set_font("Courier", "B", 12)
            self.set_text_color(*WHITE)
            self.cell(badge_w, 12, precision, align="C")

        # Callout box
        callout_y = start_y + 3 * (box_h + gap) + 6
        callout_h = 28
        self.set_fill_color(*CALLOUT_BG)
        self.set_draw_color(*SIENNA)
        self.set_line_width(1.0)
        self.rect(margin, callout_y, box_w, callout_h, "DF")

        self.set_xy(margin, callout_y + 6)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*SIENNA)
        self.cell(box_w, 14,
                  "Machine precision on point estimates and standard errors.",
                  align="C")

        # Validation scope footnote
        fn_y = callout_y + callout_h + 4
        self.set_xy(0, fn_y)
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(*STONE_300)
        self.cell(WIDTH, 10,
                  "DiD and TWFE cross-validated on real federal survey data.",
                  align="C")
        self.set_xy(0, fn_y + 12)
        self.cell(WIDTH, 10,
                  "7 estimators validated against R reference implementations.",
                  align="C")

        self.add_footer()

    def slide_07_code(self):
        """Slide 7: The Code — CallawaySantAnna + SurveyDesign.

        Claims & sources:
        - from diff_diff import CallawaySantAnna: __init__.py
        - from diff_diff import SurveyDesign: __init__.py
        - SurveyDesign params: survey.py lines 58-61
          (weights, strata, psu, fpc)
        - CallawaySantAnna.fit() params: staggered.py lines 1372-1382
          (data, outcome, unit, time, first_treat, survey_design)
        """
        self.add_page()
        self.ivory_background()

        self.centered_text(28, "The Code", size=36)
        self.centered_text(58,
                           "Three lines from flat weights to design-based inference",
                           size=15, bold=False, color=STONE_500)

        margin = 28
        code_y = 82

        token_lines = [
            [("from", SIENNA), (" diff_diff ", WHITE),
             ("import", SIENNA), (" CallawaySantAnna", WHITE)],
            [("from", SIENNA), (" diff_diff ", WHITE),
             ("import", SIENNA), (" SurveyDesign", WHITE)],
            [],  # blank
            [("design", WHITE), (" = ", WHITE),
             ("SurveyDesign", CODE_GOLD), ("(", WHITE)],
            [("    ", WHITE), ("weights", WHITE), ("=", SIENNA),
             ("'pw'", CODE_GREEN), (",", WHITE)],
            [("    ", WHITE), ("strata", WHITE), ("=", SIENNA),
             ("'stratum'", CODE_GREEN), (",", WHITE)],
            [("    ", WHITE), ("psu", WHITE), ("=", SIENNA),
             ("'cluster'", CODE_GREEN), (",", WHITE)],
            [("    ", WHITE), ("fpc", WHITE), ("=", SIENNA),
             ("'pop_size'", CODE_GREEN), (")", WHITE)],
            [],  # blank
            [("cs", WHITE), (" = ", WHITE),
             ("CallawaySantAnna", CODE_GOLD), ("()", WHITE)],
            [("result", WHITE), (" = cs.fit(data,", WHITE)],
            [("    ", WHITE), ("outcome", WHITE), ("=", SIENNA),
             ("'y'", CODE_GREEN), (",", WHITE)],
            [("    ", WHITE), ("unit", WHITE), ("=", SIENNA),
             ("'id'", CODE_GREEN), (", ", WHITE),
             ("time", WHITE), ("=", SIENNA), ("'t'", CODE_GREEN),
             (",", WHITE)],
            [("    ", WHITE), ("first_treat", WHITE), ("=", SIENNA),
             ("'g'", CODE_GREEN), (",", WHITE)],
            [("    ", WHITE), ("survey_design", WHITE), ("=", SIENNA),
             ("design", CODE_GOLD), (")", WHITE)],
        ]

        code_h = self._add_code_block(
            margin, code_y, WIDTH - margin * 2, token_lines,
        )

        # Subtitles — keep above footer (rule at HEIGHT-28)
        sub_y = min(code_y + code_h + 10, HEIGHT - 58)
        self.centered_text(sub_y,
                           "Same fit() API as every diff-diff estimator.",
                           size=15, bold=False, color=STONE_500)
        self.centered_text(sub_y + 18,
                           "Just add survey_design=...",
                           size=15, bold=True, color=SIENNA)

        self.add_footer()

    def slide_08_cta(self):
        """Slide 8: CTA — Get Started."""
        self.add_page()
        self.ivory_background()

        self.centered_text(50, "Get Started", size=42)

        # pip install badge (sienna)
        badge_w = 210
        badge_h = 36
        badge_x = (WIDTH - badge_w) / 2
        badge_y = 115
        self.set_fill_color(*SIENNA)
        self.rect(badge_x, badge_y, badge_w, badge_h, "F")

        self.set_xy(badge_x, badge_y + 9)
        self.set_font("Courier", "B", 15)
        self.set_text_color(*WHITE)
        self.cell(badge_w, 16, "$ pip install --upgrade diff-diff",
                  align="C")

        # Links
        self.centered_text(178, "github.com/igerber/diff-diff",
                           size=18, color=SIENNA)

        # Wordmark
        self.draw_split_logo(235, size=28)

        # Subtitle
        self.centered_text(262, "Difference-in-Differences for Python",
                           size=15, bold=False, color=STONE_500)

        self.add_footer()


def main():
    pdf = V3SurveyCarouselPDF()
    try:
        pdf.slide_01_hook()
        pdf.slide_02_problem()
        pdf.slide_03_deff_visual()
        pdf.slide_04_theory()
        pdf.slide_05_supported()
        pdf.slide_06_validation()
        pdf.slide_07_code()
        pdf.slide_08_cta()

        output_path = Path(__file__).parent / "diff-diff-v3-carousel.pdf"
        pdf.output(str(output_path))
        print(f"PDF saved to: {output_path}")
    finally:
        pdf.cleanup()


if __name__ == "__main__":
    main()
