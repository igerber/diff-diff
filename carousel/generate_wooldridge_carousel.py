#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for Wooldridge ETWFE estimator announcement."""

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from fpdf import FPDF  # noqa: E402

# Computer Modern for math
plt.rcParams["mathtext.fontset"] = "cm"

# Page dimensions (4:5 portrait)
WIDTH = 270     # mm
HEIGHT = 337.5  # mm

# Colors - Light theme with emerald accent
EMERALD = (16, 185, 129)       # #10b981  — primary accent
NAVY = (15, 23, 42)            # #0f172a  — primary text
MID_BLUE = (59, 130, 246)      # #3b82f6  — logo dash, pip badge bg
WHITE = (255, 255, 255)        # #ffffff
RED = (220, 38, 38)            # #dc2626  — problem accent (slide 2)
GRAY = (100, 116, 139)         # #64748b  — secondary text
LIGHT_GRAY = (148, 163, 184)   # #94a3b8  — fine print
EMERALD_TINT = (236, 253, 245) # #ecfdf5  — callout box bg
DARK_SLATE = (30, 41, 59)      # #1e293b  — code block bg
GREEN_CODE = (80, 250, 123)    # #50fa7b  — code string literals

# Hex colors for matplotlib
NAVY_HEX = "#0f172a"
EMERALD_HEX = "#10b981"
EMERALD_LIGHT_HEX = "#6ee7b7"
RED_HEX = "#dc2626"
GRAY_HEX = "#64748b"


class WooldridgeCarouselPDF(FPDF):
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

    # ── Background & Footer ──────────────��─────────────────────────

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
        """Add footer with emerald rule and version text."""
        rule_y = HEIGHT - 28
        self.set_draw_color(*EMERALD)
        self.set_line_width(0.5)
        self.line(50, rule_y, WIDTH - 50, rule_y)

        self.set_font("Helvetica", "B", 12)
        dd_text = "diff-diff "
        v_text = "v2.9"
        dd_w = self.get_string_width(dd_text)
        v_w = self.get_string_width(v_text)
        start_x = (WIDTH - dd_w - v_w) / 2

        self.set_xy(start_x, HEIGHT - 22)
        self.set_text_color(*GRAY)
        self.cell(dd_w, 10, dd_text)
        self.set_text_color(*EMERALD)
        self.cell(v_w, 10, v_text)

    # ── Text Helpers ───────────────────────────────────────────────

    def centered_text(self, y, text, size=28, bold=True, color=NAVY,
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
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="R")
        self.set_text_color(*MID_BLUE)
        self.cell(10, 10, "-", align="C")
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="L")

    # ── Equation Rendering ────���────────────────────────────────────

    def _render_equations(self, latex_lines, fontsize=26, color=NAVY_HEX):
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

    # ── Event Study Plot ─────────��─────────────────────────────────

    def _render_event_study_plot(self):
        """Render synthetic event study plot for Slide 6.

        Uses hardcoded ATT values (no runtime library dependency).
        """
        ks = np.arange(-5, 7)
        # Pre-treatment: near zero (parallel trends hold)
        atts_pre = [0.05, -0.12, 0.08, -0.03, 0.10]
        # Post-treatment: stable ~2.0 effect
        atts_post = [1.95, 2.05, 2.10, 2.00, 2.15, 1.90, 2.05]
        atts = np.array(atts_pre + atts_post)
        ses = np.array([0.25, 0.22, 0.20, 0.18, 0.20,
                        0.22, 0.20, 0.22, 0.24, 0.25, 0.28, 0.30])
        ci_lower = atts - 1.96 * ses
        ci_upper = atts + 1.96 * ses

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # CI band
        ax.fill_between(ks, ci_lower, ci_upper,
                         color=EMERALD_LIGHT_HEX, alpha=0.3)

        # ATT line + dots
        ax.plot(ks, atts, "o-", color=EMERALD_HEX, linewidth=2.5,
                markersize=7, zorder=5)

        # Reference lines
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(-0.5, color=RED_HEX, linestyle=":", linewidth=1.5,
                   label="Treatment onset")

        ax.set_xlabel("Relative period (k = t \u2212 g)", fontsize=13,
                       color=NAVY_HEX)
        ax.set_ylabel("ATT", fontsize=13, color=NAVY_HEX)
        ax.set_title("ETWFE Event Study", fontsize=16, fontweight="bold",
                      color=NAVY_HEX)
        ax.tick_params(colors=NAVY_HEX)
        for spine in ax.spines.values():
            spine.set_color(GRAY_HEX)

        ax.legend(loc="upper left", fontsize=11)
        fig.tight_layout()

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1,
                    facecolor="white")
        plt.close(fig)

        with PILImage.open(path) as img:
            pw, ph = img.size

        self._temp_files.append(path)
        return path, pw, ph

    # ── Code Block ��─────────────────────────────────────��──────────

    def _add_code_block(self, x, y, w, token_lines, font_size=13,
                        line_height=12):
        """Render syntax-highlighted code on a dark panel."""
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

    # ════════════��══════════════════════════════��════════════════════
    # SLIDES
    # ══════════════════════════════��═════════════════════════════��═══

    def slide_01_hook(self):
        """Slide 1: Hook — When Linear DiD Isn't Enough.

        Claims & sources:
        - "diff-diff v2.9": version confirmed in pyproject.toml
        - "When Linear DiD Isn't Enough": Wooldridge (2023) Section 1
          motivates nonlinear alternatives as robustness checks for
          settings where parallel trends on the level scale is implausible
        - Teaser items: confirmed in wooldridge.py and wooldridge_results.py
        """
        self.add_page()
        self.light_gradient_background()

        self.draw_split_logo(55, size=60)
        self.centered_text(120, "v2.9", size=50, color=EMERALD)

        self.centered_text(170, "When Linear DiD", size=26)
        self.centered_text(193, "Isn't Enough.", size=26)

        teasers = [
            "Wooldridge (2023, 2025) ETWFE estimator",
            "OLS, Logit, and Poisson QMLE in one class",
            "Delta-method SEs, four aggregation types",
        ]
        y_start = 235
        for i, teaser in enumerate(teasers):
            self.set_xy(0, y_start + i * 22)
            self.set_font("Helvetica", "", 17)
            self.set_text_color(*GRAY)
            self.cell(WIDTH, 10, teaser, align="C")

        self.add_footer()

    def slide_02_problem(self):
        """Slide 2: The Nonlinear Gap.

        Claims & sources:
        - Binary, count, and rate outcomes may make level-scale parallel
          trends implausible: Wooldridge (2023) Section 1
        - "Standard parallel trends may not hold on the level scale":
          Wooldridge (2023) motivates nonlinear link functions to place
          parallel trends on a more plausible scale (log-odds, log)
        """
        self.add_page()
        self.light_gradient_background()

        self.centered_text(35, "The Nonlinear", size=38)
        self.centered_text(68, "Gap", size=38, color=RED)

        # Three stacked problem cards with red left-accent bars
        margin = 35
        box_w = WIDTH - margin * 2
        box_h = 44
        gap = 6
        start_y = 100
        bar_w = 4

        problems = [
            ("Binary Outcomes",
             "Voting, adoption, mortality -- bounded [0, 1]"),
            ("Count Data",
             "Patent filings, ER visits -- non-negative integers"),
            ("Rates / Proportions",
             "Employment rates, market shares -- fractional"),
        ]

        for i, (title, desc) in enumerate(problems):
            by = start_y + i * (box_h + gap)

            # White card
            self.set_fill_color(*WHITE)
            self.set_draw_color(220, 220, 230)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, box_h, "DF")

            # Red accent bar
            self.set_fill_color(*RED)
            self.rect(margin, by, bar_w, box_h, "F")

            # Title
            self.set_xy(margin + bar_w + 12, by + 10)
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(*NAVY)
            self.cell(box_w - bar_w - 24, 10, title)

            # Description
            self.set_xy(margin + bar_w + 12, by + 28)
            self.set_font("Helvetica", "", 15)
            self.set_text_color(*GRAY)
            self.cell(box_w - bar_w - 24, 10, desc)

        # Emerald callout box
        callout_y = start_y + 3 * (box_h + gap) + 8
        callout_h = 32
        self.set_fill_color(*EMERALD_TINT)
        self.set_draw_color(*EMERALD)
        self.set_line_width(1.0)
        self.rect(margin, callout_y, box_w, callout_h, "DF")

        self.set_xy(margin, callout_y + 8)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*EMERALD)
        self.cell(box_w, 14,
                  "Standard parallel trends may not hold on the level scale.",
                  align="C")

        # Fine print citation
        self.set_xy(0, callout_y + callout_h + 8)
        self.set_font("Helvetica", "I", 13)
        self.set_text_color(*LIGHT_GRAY)
        self.cell(WIDTH, 10,
                  "Wooldridge (2023): Simple approaches to nonlinear DiD",
                  align="C")

        self.add_footer()

    def slide_03_method(self):
        """Slide 3: Wooldridge ETWFE — the method.

        Claims & sources:
        - Wooldridge (2023): "Simple approaches to nonlinear
          difference-in-differences with panel data." The Econometrics
          Journal, 26(3), C31-C66.
        - Wooldridge (2025): "Two-Way Fixed Effects, the Two-Way Mundlak
          Regression, and Difference-in-Differences Estimators." Empirical
          Economics, 69(5), 2545-2587.
        - Stata jwdid: Rios-Avila (2021), SSC s459114
        - "Single saturated regression": Wooldridge (2023) framework
        - Equation shown is the OLS specification with unit+time FE
          (REGISTRY.md lines 1094-1103). Logit/Poisson use additive
          cohort+time dummies instead of unit FE to avoid incidental
          parameters bias (REGISTRY.md lines 1152-1153).
        - "Nonlinear link functions": _VALID_METHODS = ("ols", "logit",
          "poisson") in wooldridge.py line 25
        """
        self.add_page()
        self.light_gradient_background()

        self.centered_text(30, "Wooldridge ETWFE", size=36)
        self.centered_text(62, "Extended Two-Way Fixed Effects",
                           size=18, bold=False, italic=True, color=GRAY)

        # Citation
        self.centered_text(85, "Wooldridge (2023, 2025)  |  Stata: jwdid",
                           size=15, bold=False, italic=True, color=GRAY)

        # OLS equation label
        self.centered_text(105, "OLS specification (unit + time FE absorbed):",
                           size=14, bold=False, color=GRAY)

        # Saturated regression equation
        eq_path, epw, eph = self._render_equations(
            [r"$Y_{it} = \alpha_i + \gamma_t + "
             r"\sum_{g} \sum_{t \geq g} \delta_{g,t}"
             r"\cdot D_{ig} \cdot f_t"
             r" + \mathbf{X}'\beta + \varepsilon_{it}$"],
            fontsize=24,
        )
        eq_h = self._place_equation_centered(eq_path, epw, eph, 120,
                                             max_w=220)

        # Nonlinear note
        nl_y = 120 + eq_h + 4
        self.centered_text(nl_y,
                           "Logit / Poisson: cohort + time dummies, ASF for ATT",
                           size=14, bold=False, color=EMERALD)

        # Three key insight bullets with emerald dashes
        margin = 42
        y_cursor = nl_y + 22
        items = [
            ("Single saturated regression",
             "All ATT(g,t) estimated jointly"),
            ("Heterogeneous treatment effects",
             "Correctly handled across cohorts and time"),
            ("Nonlinear link functions",
             "OLS, Logit, or Poisson QMLE (the key insight)"),
        ]

        for title, desc in items:
            # Emerald dash
            self.set_xy(margin, y_cursor)
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(*EMERALD)
            self.cell(14, 10, "--")

            # Title
            self.set_xy(margin + 14, y_cursor)
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(*NAVY)
            self.cell(WIDTH - margin * 2 - 14, 10, title)

            # Description
            self.set_xy(margin + 14, y_cursor + 20)
            self.set_font("Helvetica", "", 14)
            self.set_text_color(*GRAY)
            self.cell(WIDTH - margin * 2 - 14, 10, desc)

            y_cursor += 46

        self.add_footer()

    def slide_04_three_likelihoods(self):
        """Slide 4: Three Likelihoods, One Framework.

        Claims & sources:
        - _VALID_METHODS = ("ols", "logit", "poisson"): wooldridge.py line 25
        - OLS for continuous, Logit for binary/fractional, Poisson for
          counts: wooldridge.py lines 208-210, Wooldridge (2023) Secs 3-4
        """
        self.add_page()
        self.light_gradient_background()

        self.centered_text(30, "Three Likelihoods", size=38)
        self.centered_text(63, "One Framework", size=38, color=EMERALD)

        margin = 35
        box_w = WIDTH - margin * 2
        box_h = 48
        gap = 6
        start_y = 100
        bar_w = 4
        badge_h = 22

        methods = [
            ("OLS", "Continuous outcomes",
             "Wages, test scores, log-employment"),
            ("Logit", "Binary / fractional outcomes",
             "Voting, adoption, participation rates"),
            ("Poisson", "Count / non-negative outcomes",
             "Patents, ER visits, hiring"),
        ]

        for i, (method, subtitle, examples) in enumerate(methods):
            by = start_y + i * (box_h + gap)

            # White card
            self.set_fill_color(*WHITE)
            self.set_draw_color(*EMERALD)
            self.set_line_width(0.8)
            self.rect(margin, by, box_w, box_h, "DF")

            # Emerald accent bar
            self.set_fill_color(*EMERALD)
            self.rect(margin, by, bar_w, box_h, "F")

            # Method badge
            badge_w = 70
            badge_x = margin + bar_w + 12
            badge_y = by + 8
            self.set_fill_color(*EMERALD)
            self.rect(badge_x, badge_y, badge_w, badge_h, "F")
            self.set_xy(badge_x, badge_y + 4)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*WHITE)
            self.cell(badge_w, 12, method, align="C")

            # Subtitle
            self.set_xy(badge_x + badge_w + 10, badge_y + 4)
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(*NAVY)
            self.cell(box_w - badge_w - bar_w - 34, 12, subtitle)

            # Examples
            self.set_xy(margin + bar_w + 12, by + 35)
            self.set_font("Helvetica", "", 14)
            self.set_text_color(*GRAY)
            self.cell(box_w - bar_w - 24, 10, examples)

        # Callout text
        callout_y = start_y + 3 * (box_h + gap) + 6
        self.centered_text(callout_y,
                           "Choose the link function.",
                           size=17, bold=True, color=EMERALD)
        self.centered_text(callout_y + 22,
                           "The estimator handles the rest.",
                           size=17, bold=True, color=EMERALD)

        self.add_footer()

    def slide_05_technical(self):
        """Slide 5: Under the Hood — ATT extraction, delta-method, aggregations.

        Claims & sources:
        - OLS ATT: direct coefficient extraction, wooldridge.py lines 599-610
        - Nonlinear ASF-based ATT: wooldridge.py lines 794-819
        - Delta-method SE: wooldridge_results.py line 93
          var = w' @ vcov @ w, SE = sqrt(var)
        - Four aggregation types: wooldridge_results.py lines 105-161
          simple (105-108), group (110-126), calendar (128-143),
          event (145-161). Matches Stata jwdid_estat.
        """
        self.add_page()
        self.light_gradient_background()

        self.centered_text(25, "Under the Hood", size=36)

        # OLS path
        self.centered_text(62,
                           "OLS:  ATT(g,t) directly from coefficients",
                           size=15, bold=False, color=GRAY)

        # Nonlinear ASF equation
        self.centered_text(80,
                           "Logit / Poisson:  ASF-based ATT",
                           size=15, bold=False, color=EMERALD)

        eq_path, epw, eph = self._render_equations(
            [r"$\mathrm{ATT}(g,t) = \frac{1}{N_{g,t}}"
             r" \sum_{i:\, G_i=g}"
             r" \left[ f(\hat{\eta}_{i,1})"
             r" - f(\hat{\eta}_{i,0}) \right]$"],
            fontsize=24,
        )
        eq_h = self._place_equation_centered(eq_path, epw, eph, 96,
                                             max_w=220)

        # f annotation
        f_y = 96 + eq_h + 4
        self.centered_text(f_y,
                           "f = logistic  |  exponential",
                           size=14, bold=False, color=EMERALD)

        # Delta-method subtitle
        self.centered_text(f_y + 22,
                           "Delta-method SEs for all aggregations",
                           size=17, bold=True, color=NAVY)

        # 2x2 grid of aggregation types
        margin = 30
        grid_gap = 8
        card_w = (WIDTH - margin * 2 - grid_gap) / 2
        card_h = 52
        grid_y = f_y + 48

        agg_types = [
            ("simple", "Overall weighted average ATT"),
            ("group", "ATT by treatment cohort"),
            ("calendar", "ATT by calendar period"),
            ("event", "ATT by relative time k = t - g"),
        ]

        for idx, (name, desc) in enumerate(agg_types):
            row = idx // 2
            col = idx % 2
            cx = margin + col * (card_w + grid_gap)
            cy = grid_y + row * (card_h + grid_gap)

            # Card with emerald border
            self.set_fill_color(*WHITE)
            self.set_draw_color(*EMERALD)
            self.set_line_width(0.8)
            self.rect(cx, cy, card_w, card_h, "DF")

            # Name
            self.set_xy(cx + 10, cy + 8)
            self.set_font("Helvetica", "B", 17)
            self.set_text_color(*EMERALD)
            self.cell(card_w - 20, 10, name)

            # Description
            self.set_xy(cx + 10, cy + 28)
            self.set_font("Helvetica", "", 13)
            self.set_text_color(*GRAY)
            self.cell(card_w - 20, 10, desc)

        # Stata note
        stata_y = grid_y + 2 * (card_h + grid_gap) + 4
        self.centered_text(stata_y,
                           "Matches Stata jwdid_estat output",
                           size=14, bold=False, italic=True, color=LIGHT_GRAY)

        self.add_footer()

    def slide_06_event_study(self):
        """Slide 6: Event Study Plot.

        Claims & sources:
        - Event study aggregation: wooldridge_results.py lines 145-161
        - Plot shows synthetic data with hardcoded ATT values mimicking
          a well-behaved staggered DiD with true effect ~2.0
        """
        self.add_page()
        self.light_gradient_background()

        self.centered_text(25, "Event Study", size=38)

        # Event study plot
        plot_path, ppw, pph = self._render_event_study_plot()
        plot_w = WIDTH * 0.85
        plot_aspect = pph / ppw
        plot_h = plot_w * plot_aspect
        plot_x = (WIDTH - plot_w) / 2
        plot_y = 65
        self.image(plot_path, plot_x, plot_y, plot_w)

        # Annotations below plot
        ann_y = plot_y + plot_h + 10
        self.centered_text(ann_y,
                           "Pre-treatment: near zero (parallel trends hold)",
                           size=15, bold=False, color=GRAY)
        self.centered_text(ann_y + 20,
                           "Post-treatment: stable, significant effect",
                           size=15, bold=True, color=EMERALD)

        self.add_footer()

    def slide_07_code(self):
        """Slide 7: The Code — Poisson QMLE example.

        Claims & sources:
        - from diff_diff import WooldridgeDiD: __init__.py
        - WooldridgeDiD(method='poisson'): wooldridge.py line 25
        - fit() API: wooldridge.py lines 322-332
        - .aggregate('event').summary('event'): wooldridge_results.py
          summary() defaults to 'simple'; must pass 'event' explicitly
        """
        self.add_page()
        self.light_gradient_background()

        self.centered_text(30, "The Code", size=38)

        margin = 28
        code_y = 80

        token_lines = [
            [("from", EMERALD), (" diff_diff ", WHITE),
             ("import", EMERALD), (" WooldridgeDiD", WHITE)],
            [],  # blank
            [("est ", WHITE), ("=", EMERALD),
             (" WooldridgeDiD(", WHITE),
             ("method", WHITE), ("=", EMERALD),
             ("'poisson'", GREEN_CODE), (")", WHITE)],
            [("results ", WHITE), ("=", EMERALD),
             (" est.fit(", WHITE)],
            [("    data, ", WHITE), ("outcome", WHITE),
             ("=", EMERALD), ("'emp'", GREEN_CODE), (",", WHITE)],
            [("    ", WHITE), ("unit", WHITE),
             ("=", EMERALD), ("'county'", GREEN_CODE), (", ", WHITE),
             ("time", WHITE), ("=", EMERALD), ("'year'", GREEN_CODE),
             (",", WHITE)],
            [("    ", WHITE), ("cohort", WHITE),
             ("=", EMERALD), ("'first_treat'", GREEN_CODE), (")", WHITE)],
            [],  # blank
            [("results.aggregate(", WHITE),
             ("'event'", GREEN_CODE),
             (").summary(", WHITE),
             ("'event'", GREEN_CODE),
             (")", WHITE)],
        ]

        code_h = self._add_code_block(
            margin, code_y, WIDTH - margin * 2, token_lines,
        )

        # Subtitles
        self.centered_text(code_y + code_h + 16,
                           "Same fit() API as every diff-diff estimator",
                           size=16, bold=False, color=GRAY)
        self.centered_text(code_y + code_h + 38,
                           "OLS, Logit, or Poisson -- just change the method",
                           size=16, bold=False, color=GRAY)

        self.add_footer()

    def slide_08_milestone(self):
        """Slide 8: 15 Estimators — milestone + community credit.

        Claims & sources:
        - 15 estimators: counted from __init__.py — all classes with fit()
          that estimate treatment effects. Excludes BaconDecomposition
          (diagnostic), HonestDiD (sensitivity), PowerAnalysis, PreTrendsPower.
        - PR #216 from @wenddymacro: confirmed in git log (commit 54bea41)
        - "Most comprehensive standalone DiD library": diff-diff implements
          all solvers from scratch (no pyfixest/statsmodels dependency).
          The etwfe package on PyPI is a thin wrapper around pyfixest.
        """
        self.add_page()
        self.light_gradient_background()

        # Large counter
        self.set_xy(0, 65)
        self.set_font("Helvetica", "B", 90)
        self.set_text_color(*EMERALD)
        self.cell(WIDTH, 45, "15", align="C")

        self.centered_text(145, "estimators and counting", size=24)

        self.centered_text(185,
                           "The most comprehensive standalone",
                           size=17, bold=False, color=GRAY)
        self.centered_text(205,
                           "DiD library for Python",
                           size=17, bold=False, color=GRAY)

        # Community credit box
        margin = 40
        box_w = WIDTH - margin * 2
        box_h = 40
        box_y = 240

        self.set_fill_color(*EMERALD_TINT)
        self.set_draw_color(*EMERALD)
        self.set_line_width(0.8)
        self.rect(margin, box_y, box_w, box_h, "DF")

        self.set_xy(margin, box_y + 6)
        self.set_font("Helvetica", "", 15)
        self.set_text_color(*GRAY)
        self.cell(box_w, 12,
                  "Community-contributed via PR #216",
                  align="C")
        self.set_xy(margin, box_y + 22)
        self.cell(box_w, 12,
                  "by @wenddymacro",
                  align="C")

        self.add_footer()

    def slide_09_cta(self):
        """Slide 9: CTA — Get Started."""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(50, "Get Started", size=42)

        # pip install badge
        badge_w = 210
        badge_h = 36
        badge_x = (WIDTH - badge_w) / 2
        badge_y = 115
        self.set_fill_color(*MID_BLUE)
        self.rect(badge_x, badge_y, badge_w, badge_h, "F")

        self.set_xy(badge_x, badge_y + 9)
        self.set_font("Courier", "B", 15)
        self.set_text_color(*WHITE)
        self.cell(badge_w, 16, "$ pip install --upgrade diff-diff",
                  align="C")

        # Links
        self.centered_text(178, "github.com/igerber/diff-diff",
                           size=18, color=EMERALD)
        self.centered_text(205, "Tutorial 16: Wooldridge ETWFE",
                           size=18, color=EMERALD)

        # Wordmark
        self.draw_split_logo(255, size=28)

        # Subtitle
        self.centered_text(282, "Difference-in-Differences for Python",
                           size=15, bold=False, color=GRAY)

        self.add_footer()


def main():
    pdf = WooldridgeCarouselPDF()
    try:
        pdf.slide_01_hook()
        pdf.slide_02_problem()
        pdf.slide_03_method()
        pdf.slide_04_three_likelihoods()
        pdf.slide_05_technical()
        pdf.slide_06_event_study()
        pdf.slide_07_code()
        pdf.slide_08_milestone()
        pdf.slide_09_cta()

        output_path = Path(__file__).parent / "diff-diff-wooldridge-carousel.pdf"
        pdf.output(str(output_path))
        print(f"PDF saved to: {output_path}")
    finally:
        pdf.cleanup()


if __name__ == "__main__":
    main()
