#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for TROP estimator announcement."""

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from fpdf import FPDF  # noqa: E402

# Computer Modern for math
plt.rcParams["mathtext.fontset"] = "cm"

# Page dimensions (4:5 portrait)
WIDTH = 270     # mm
HEIGHT = 337.5  # mm

# Dark theme palette (gold accent, shared dark base with V27)
BG = (26, 26, 46)          # #1a1a2e
GOLD = (245, 166, 35)      # #f5a623  (primary accent)
WHITE = (255, 255, 255)    # #ffffff
GRAY = (136, 146, 176)     # #8892b0
DARK_PANEL = (22, 33, 62)  # #16213e
GREEN = (80, 250, 123)     # #50fa7b  (code strings)

# Hex colors for matplotlib
GOLD_HEX = "#f5a623"
WHITE_HEX = "#ffffff"
GRAY_HEX = "#8892b0"
DARK_PANEL_HEX = "#16213e"
BG_HEX = "#1a1a2e"


class TROPCarouselPDF(FPDF):
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

    # ── Background & Footer ────────────────────────────────────────

    def _add_dark_bg(self):
        """Fill page with dark background."""
        self.set_fill_color(*BG)
        self.rect(0, 0, WIDTH, HEIGHT, "F")

    def _add_footer(self):
        """Add footer with gold rule and version text."""
        rule_y = HEIGHT - 28
        self.set_draw_color(*GOLD)
        self.set_line_width(0.5)
        self.line(50, rule_y, WIDTH - 50, rule_y)

        self.set_font("Helvetica", "B", 12)
        dd_text = "diff-diff "
        v_text = "v2.7"
        dd_w = self.get_string_width(dd_text)
        v_w = self.get_string_width(v_text)
        start_x = (WIDTH - dd_w - v_w) / 2

        self.set_xy(start_x, HEIGHT - 22)
        self.set_text_color(*GRAY)
        self.cell(dd_w, 10, dd_text)
        self.set_text_color(*GOLD)
        self.cell(v_w, 10, v_text)

    # ── Text Helpers ───────────────────────────────────────────────

    def _centered_text(self, y, text, size=28, bold=True, color=WHITE,
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

    # ── Equation Rendering ─────────────────────────────────────────

    def _render_equations(self, latex_lines, fontsize=28, color=GOLD_HEX):
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

    # ── Diagram: Convergence (Slide 3) ─────────────────────────────

    def _render_convergence(self):
        """Three boxes (DiD, MC, SC) with arrows converging to TROP circle."""
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.2, 5.2)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        # Three source boxes (left side)
        labels = ["DiD", "Matrix\nCompletion", "Synthetic\nControl"]
        y_positions = [4.0, 2.5, 1.0]

        for label, yp in zip(labels, y_positions):
            box = mpatches.FancyBboxPatch(
                (0.2, yp - 0.45), 2.8, 0.9, boxstyle="round,pad=0.15",
                facecolor=DARK_PANEL_HEX, edgecolor=GRAY_HEX, linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(1.6, yp, label, color=WHITE_HEX,
                    fontsize=14, ha="center", va="center", fontweight="bold")

            # Arrow to TROP circle
            ax.annotate(
                "", xy=(6.3, 2.5), xytext=(3.2, yp),
                arrowprops=dict(arrowstyle="-|>", color=GOLD_HEX, lw=2.0,
                                alpha=0.8, mutation_scale=18),
            )

        # TROP circle (right side)
        circle = plt.Circle((7.5, 2.5), 1.2, facecolor="none",
                            edgecolor=GOLD_HEX, linewidth=3)
        ax.add_patch(circle)
        ax.text(7.5, 2.5, "TROP", color=GOLD_HEX,
                fontsize=22, ha="center", va="center", fontweight="bold")

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1,
                    transparent=True)
        plt.close(fig)

        with PILImage.open(path) as img:
            pw, ph = img.size

        self._temp_files.append(path)
        return path, pw, ph

    # ── Diagram: Three Pillars (Slide 4) ───────────────────────────

    def _render_three_pillars(self):
        """Three pillars supporting a 'Consistent Estimate' bar."""
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5.5)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        pillar_w = 2.2
        pillar_h = 3.0
        gap = 0.7
        total_w = 3 * pillar_w + 2 * gap
        start_x = (10 - total_w) / 2
        pillar_bottom = 0.5
        pillar_top = pillar_bottom + pillar_h

        labels = ["Factor\nModel", "Unit\nWeights", "Time\nWeights"]

        for i, label in enumerate(labels):
            px = start_x + i * (pillar_w + gap)
            # Dark panel pillars with gold border
            pillar = mpatches.FancyBboxPatch(
                (px, pillar_bottom), pillar_w, pillar_h,
                boxstyle="round,pad=0.1",
                facecolor=DARK_PANEL_HEX, edgecolor=GOLD_HEX,
                linewidth=2.0,
            )
            ax.add_patch(pillar)
            ax.text(px + pillar_w / 2, pillar_bottom + pillar_h / 2, label,
                    color=GOLD_HEX, fontsize=14, ha="center", va="center",
                    fontweight="bold")

        # Solid gold top bar — visually distinct from pillars
        bar_h = 0.55
        bar_y = pillar_top + 0.15
        bar = mpatches.FancyBboxPatch(
            (start_x - 0.2, bar_y), total_w + 0.4, bar_h,
            boxstyle="round,pad=0.08",
            facecolor=GOLD_HEX, edgecolor="none", alpha=1.0,
        )
        ax.add_patch(bar)
        ax.text(5, bar_y + bar_h / 2, "Consistent Estimate",
                color=BG_HEX, fontsize=15, ha="center", va="center",
                fontweight="bold")

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1,
                    transparent=True)
        plt.close(fig)

        with PILImage.open(path) as img:
            pw, ph = img.size

        self._temp_files.append(path)
        return path, pw, ph

    # ── Diagram: Equation Labels (Slide 4) ────────────────────────

    def _render_equation_labels(self):
        """Render equation component labels with LaTeX Greek symbols."""
        fig = plt.figure(figsize=(8, 1.4))

        # Line 1: alpha, beta = unit and time fixed effects
        fig.text(0.05, 0.72, r"$\alpha_i,\; \beta_t$",
                 fontsize=18, color=GRAY_HEX, va="center")
        fig.text(0.20, 0.72, "=  unit and time fixed effects",
                 fontsize=14, color=GRAY_HEX, va="center")

        # Line 2: L_it = low-rank factor structure (highlighted gold)
        fig.text(0.05, 0.28, r"$L_{it}$",
                 fontsize=18, color=GOLD_HEX, va="center",
                 fontstyle="italic")
        fig.text(0.20, 0.28, "=  low-rank factor structure (key innovation)",
                 fontsize=14, color=GOLD_HEX, va="center")

        fig.patch.set_alpha(0)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06,
                    transparent=True)
        plt.close(fig)

        with PILImage.open(path) as img:
            pw, ph = img.size

        self._temp_files.append(path)
        return path, pw, ph

    # ── Code Block ─────────────────────────────────────────────────

    def _add_code_block(self, x, y, w, token_lines, font_size=13,
                        line_height=12):
        """Render syntax-highlighted code on a dark panel."""
        n_lines = len(token_lines)
        total_h = n_lines * line_height + 24

        self.set_fill_color(*DARK_PANEL)
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

    # ════════════════════════════════════════════════════════════════
    # SLIDES
    # ════════════════════════════════════════════════════════════════

    def slide_01_hook(self):
        """Slide 1: Hook — Triply RObust Panel (TROP) Estimator.

        Claims & sources:
        - "Triply RObust Panel": paper title, arXiv:2508.21536
        - "One estimator, three robustness guarantees": triple robustness
          property, paper Section 3 / Theorem 1
        """
        self.add_page()
        self._add_dark_bg()

        # Hero — method name
        self._centered_text(60, "Triply RObust Panel", size=48, color=GOLD)
        self._centered_text(112, "(TROP) Estimator", size=42, color=GOLD)

        # Badge
        badge_w = 170
        badge_h = 34
        badge_x = (WIDTH - badge_w) / 2
        badge_y = 175
        self.set_draw_color(*GOLD)
        self.set_line_width(1.5)
        self.rect(badge_x, badge_y, badge_w, badge_h, "D")

        self.set_xy(badge_x, badge_y + 8)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*GOLD)
        self.cell(badge_w, 16, "diff-diff v2.7", align="C")

        # Citation
        self.set_xy(0, 230)
        self.set_font("Helvetica", "I", 17)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 10, "Athey, Imbens, Qu & Viviano (2025)", align="C")

        # Tagline
        self._centered_text(265, "One estimator, three robustness guarantees",
                            size=17, bold=False, color=GRAY)

        self._add_footer()

    def slide_02_dilemma(self):
        """Slide 2: The Dilemma — which estimator do you trust?

        Claims & sources:
        - "Different assumptions that are difficult to validate or even
          compare in practice": Athey et al. (2025) Section 1, para 1
        - DiD/MC/SC assumption descriptions: REGISTRY.md lines 1309-1312,
          paper Section 2.2 special cases
        """
        self.add_page()
        self._add_dark_bg()

        self._centered_text(28, "Which Estimator", size=38, color=WHITE)
        self._centered_text(62, "Do You Trust?", size=38, color=WHITE)

        # Three stacked estimator boxes
        margin = 30
        box_w = WIDTH - margin * 2
        box_h = 44
        gap = 7
        start_y = 98

        estimators = [
            ("Difference-in-Differences", "Assumes parallel trends"),
            ("Matrix Completion", "Assumes low-rank factor model"),
            ("Synthetic Control", "Assumes similar donor units exist"),
        ]

        for i, (name, assumption) in enumerate(estimators):
            by = start_y + i * (box_h + gap)

            # Dark panel box
            self.set_fill_color(*DARK_PANEL)
            self.set_draw_color(*GRAY)
            self.set_line_width(0.8)
            self.rect(margin, by, box_w, box_h, "DF")

            # Estimator name
            self.set_xy(margin + 15, by + 8)
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(*WHITE)
            self.cell(box_w - 30, 10, name)

            # Assumption
            self.set_xy(margin + 15, by + 24)
            self.set_font("Helvetica", "", 15)
            self.set_text_color(*GRAY)
            self.cell(box_w - 30, 10, assumption)

        # Pull quote (italic gold) — exact wording from paper Section 1
        quote_y = start_y + 3 * (box_h + gap) + 6
        self._centered_text(quote_y,
                            "Different assumptions that are difficult",
                            size=16, bold=False, italic=True, color=GOLD)
        self._centered_text(quote_y + 17,
                            "to validate or even compare in practice",
                            size=16, bold=False, italic=True, color=GOLD)

        # Attribution
        self._centered_text(quote_y + 36, "Athey et al. (2025)",
                            size=14, bold=False, color=GRAY)

        self._add_footer()

    def slide_03_answer(self):
        """Slide 3: The Answer — TROP subsumes all three approaches.

        Claims & sources:
        - "TROP subsumes all three approaches -- each is a special case":
          REGISTRY.md lines 1309-1312, paper Section 2.2 special cases:
          lambda_nn=inf,uniform -> DID/TWFE; uniform,lambda_nn<inf -> MC;
          lambda_nn=inf,specific weights -> SC/SDID
        - "Consistent if any one modeling component is correct": triple
          robustness property, paper Section 3 / Theorem 1
        """
        self.add_page()
        self._add_dark_bg()

        self._centered_text(25, "What If You Didn't", size=38, color=WHITE)
        self._centered_text(58, "Have to Choose?", size=38, color=WHITE)

        # Body text
        self._centered_text(98, "TROP subsumes all three approaches",
                            size=18, bold=False, color=GRAY)
        self._centered_text(118, "-- each is a special case.",
                            size=18, bold=False, color=GRAY)

        # Convergence diagram
        diag_path, dpw, dph = self._render_convergence()
        diag_w = WIDTH * 0.80
        diag_aspect = dph / dpw
        diag_h = diag_w * diag_aspect
        diag_x = (WIDTH - diag_w) / 2
        diag_y = 140
        self.image(diag_path, diag_x, diag_y, diag_w)

        # Key callout
        callout_y = diag_y + diag_h + 8
        self._centered_text(callout_y,
                            "Consistent if any one modeling",
                            size=20, bold=True, color=GOLD)
        self._centered_text(callout_y + 24,
                            "component is correct",
                            size=20, bold=True, color=GOLD)

        self._add_footer()

    def slide_04_triple_robustness(self):
        """Slide 4: Triple Robustness — three pillars + equation.

        Claims & sources:
        - Three components (Factor Model, Unit Weights, Time Weights):
          REGISTRY.md lines 1271-1307, paper Equations 2-3
        - Working model Y_it(0) = alpha_i + beta_t + L_it + eps_it:
          REGISTRY.md lines 1273-1279, paper Section 2.2
        - "Any one pillar is sufficient": triple robustness property,
          paper Section 3 / Theorem 1
        """
        self.add_page()
        self._add_dark_bg()

        self._centered_text(20, "Triple Robustness", size=38, color=WHITE)

        # Three pillars diagram
        pillar_path, ppw, pph = self._render_three_pillars()
        pillar_w = WIDTH * 0.78
        pillar_aspect = pph / ppw
        pillar_h = pillar_w * pillar_aspect
        pillar_x = (WIDTH - pillar_w) / 2
        pillar_y = 58
        self.image(pillar_path, pillar_x, pillar_y, pillar_w)

        # Subtitle
        sub_y = pillar_y + pillar_h + 6
        self._centered_text(sub_y, "Any one pillar is sufficient",
                            size=17, bold=False, color=GRAY)

        # Equation intro
        intro_y = sub_y + 22
        self._centered_text(intro_y,
                            "TROP models counterfactual outcomes as:",
                            size=16, bold=False, color=GRAY)

        # Equation: working model
        eq_path, epw, eph = self._render_equations(
            [r"$Y_{it}(0) = \alpha_i + \beta_t + L_{it}"
             r" + \varepsilon_{it}$"],
            fontsize=28,
        )
        eq_y = intro_y + 18
        eq_h = self._place_equation_centered(eq_path, epw, eph, eq_y,
                                             max_w=190)

        # Component labels (rendered via matplotlib for Greek symbols)
        label_path, lpw, lph = self._render_equation_labels()
        label_w = WIDTH * 0.72
        label_aspect = lph / lpw
        label_h = label_w * label_aspect
        label_x = (WIDTH - label_w) / 2
        label_y = eq_y + eq_h + 4
        self.image(label_path, label_x, label_y, label_w)

        self._add_footer()

    def slide_05_when_to_use(self):
        """Slide 5: When to Use TROP."""
        self.add_page()
        self._add_dark_bg()

        self._centered_text(30, "When to Use TROP", size=38, color=WHITE)

        items = [
            ("Unobserved confounders",
             "Factor-structured interactive fixed effects"),
            ("Staggered adoption",
             "Different treatment timing across units"),
            ("Uncertain assumptions",
             "You don't know which estimator to trust"),
            ("Heterogeneous effects",
             "Individual treatment effects per unit and time"),
        ]

        y_cursor = 90
        margin = 42

        for title, description in items:
            # Gold dash
            self.set_xy(margin, y_cursor)
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(*GOLD)
            self.cell(14, 10, "--")

            # Title
            self.set_xy(margin + 14, y_cursor)
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(*WHITE)
            self.cell(WIDTH - margin * 2 - 14, 10, title)

            # Description
            self.set_xy(margin + 14, y_cursor + 22)
            self.set_font("Helvetica", "", 15)
            self.set_text_color(*GRAY)
            self.cell(WIDTH - margin * 2 - 14, 10, description)

            y_cursor += 55

        self._add_footer()

    def slide_06_code(self):
        """Slide 6: The Code — syntax-highlighted TROP API example."""
        self.add_page()
        self._add_dark_bg()

        self._centered_text(30, "The Code", size=38, color=WHITE)

        margin = 28
        code_y = 80

        token_lines = [
            [("from", GOLD), (" diff_diff ", WHITE),
             ("import", GOLD), (" TROP", WHITE)],
            [],  # blank line
            [("trop ", WHITE), ("=", GOLD), (" TROP(", WHITE),
             ("method", WHITE), ("=", GOLD),
             ("'local'", GREEN), (")", WHITE)],
            [("results ", WHITE), ("=", GOLD), (" trop.fit(", WHITE)],
            [("    data, ", WHITE), ("outcome", WHITE),
             ("=", GOLD), ('"y"', GREEN), (",", WHITE)],
            [("    ", WHITE), ("treatment", WHITE),
             ("=", GOLD), ('"D"', GREEN), (", ", WHITE),
             ("unit", WHITE), ("=", GOLD), ('"id"', GREEN),
             (",", WHITE)],
            [("    ", WHITE), ("time", WHITE),
             ("=", GOLD), ('"t"', GREEN), (")", WHITE)],
            [],  # blank line
            [("results.print_summary()", WHITE)],
        ]

        code_h = self._add_code_block(
            margin, code_y, WIDTH - margin * 2, token_lines,
        )

        # Subtitle
        self._centered_text(code_y + code_h + 18,
                            "sklearn-like API  |  optional Rust backend",
                            size=18, bold=False, color=GRAY)

        self._add_footer()

    def slide_07_tuning(self):
        """Slide 7: Data-Driven Tuning.

        Claims & sources:
        - "All tuning parameters selected automatically via LOOCV":
          REGISTRY.md lines 1314-1335, paper Equation 5, Footnote 2
        - Individual treatment effects tau_it: REGISTRY.md lines 1263-1268,
          paper Equation 1
        - Factor diagnostics / effective rank: REGISTRY.md lines 1371-1376
        - Local vs global methods: REGISTRY.md lines 1382-1414
        """
        self.add_page()
        self._add_dark_bg()

        self._centered_text(25, "Data-Driven Tuning", size=38, color=WHITE)

        # Body text
        self._centered_text(70,
                            "All tuning parameters selected automatically via",
                            size=18, bold=False, color=GRAY)
        self._centered_text(90,
                            "leave-one-out cross-validation.",
                            size=18, bold=False, color=GRAY)

        # Four highlight items with gold left-accent bar
        items = [
            ("LOOCV",
             "Selects all three lambda parameters"),
            ("Individual effects",
             "tau(i,t) for each treated observation"),
            ("Factor diagnostics",
             "Estimated factor matrix and effective rank"),
            ("Two methods",
             "Local (per-observation) or global (faster, simultaneous)"),
        ]

        margin = 35
        bar_w = 4
        bar_h = 32
        y_cursor = 125

        for title, desc in items:
            # Gold accent bar
            self.set_fill_color(*GOLD)
            self.rect(margin, y_cursor, bar_w, bar_h, "F")

            # Title
            self.set_xy(margin + bar_w + 10, y_cursor + 2)
            self.set_font("Helvetica", "B", 17)
            self.set_text_color(*WHITE)
            self.cell(WIDTH - margin * 2, 10, title)

            # Description
            self.set_xy(margin + bar_w + 10, y_cursor + 18)
            self.set_font("Helvetica", "", 14)
            self.set_text_color(*GRAY)
            self.cell(WIDTH - margin * 2, 10, desc)

            y_cursor += 42

        self._add_footer()

    def slide_08_cta(self):
        """Slide 8: CTA — Get Started."""
        self.add_page()
        self._add_dark_bg()

        self._centered_text(55, "Get Started", size=42, color=WHITE)

        # pip install badge (gold background)
        badge_w = 210
        badge_h = 36
        badge_x = (WIDTH - badge_w) / 2
        badge_y = 115
        self.set_fill_color(*GOLD)
        self.rect(badge_x, badge_y, badge_w, badge_h, "F")

        self.set_xy(badge_x, badge_y + 9)
        self.set_font("Courier", "B", 16)
        self.set_text_color(*BG)
        self.cell(badge_w, 16, "pip install diff-diff", align="C")

        # Links
        self._centered_text(178, "github.com/igerber/diff-diff",
                            size=18, color=GOLD)
        self._centered_text(205, "arXiv:2508.21536",
                            size=18, color=GOLD)

        # Wordmark
        self.set_font("Helvetica", "B", 36)
        dd_text = "diff-diff "
        v_text = "v2.7"
        dd_w = self.get_string_width(dd_text)
        v_w = self.get_string_width(v_text)
        start_x = (WIDTH - dd_w - v_w) / 2

        self.set_xy(start_x, 255)
        self.set_text_color(*WHITE)
        self.cell(dd_w, 20, dd_text)
        self.set_text_color(*GOLD)
        self.cell(v_w, 20, v_text)

        # Subtitle
        self._centered_text(288, "Difference-in-Differences for Python",
                            size=15, bold=False, color=GRAY)

        self._add_footer()


def main():
    pdf = TROPCarouselPDF()
    try:
        pdf.slide_01_hook()
        pdf.slide_02_dilemma()
        pdf.slide_03_answer()
        pdf.slide_04_triple_robustness()
        pdf.slide_05_when_to_use()
        pdf.slide_06_code()
        pdf.slide_07_tuning()
        pdf.slide_08_cta()

        output_path = Path(__file__).parent / "diff-diff-trop-carousel.pdf"
        pdf.output(str(output_path))
        print(f"PDF saved to: {output_path}")
    finally:
        pdf.cleanup()


if __name__ == "__main__":
    main()
