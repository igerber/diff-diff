#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for dCDH estimator announcement."""

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from fpdf import FPDF  # noqa: E402

# Computer Modern for math
plt.rcParams["mathtext.fontset"] = "cm"

# Page dimensions (4:5 portrait)
WIDTH = 270     # mm
HEIGHT = 337.5  # mm

# ── Dark cool-slate + teal palette ──────────────────────────────
SLATE_900 = (15, 23, 42)        # #0f172a — page background
SLATE_800 = (30, 41, 59)        # #1e293b — card/panel bg
SLATE_700 = (51, 65, 85)        # #334155 — elevated panels
TEAL = (20, 184, 166)           # #14b8a6 — primary accent
TEAL_LIGHT = (94, 234, 212)     # #5eead4 — CI bands, secondary
TEAL_DARK = (13, 148, 136)      # #0d9488 — accent bars
WHITE = (255, 255, 255)         # #ffffff — primary text
SLATE_400 = (148, 163, 184)     # #94a3b8 — secondary text
SLATE_300 = (203, 213, 225)     # #cbd5e1 — fine print
CORAL = (251, 113, 133)         # #fb7185 — problem accent
CODE_BG = (2, 6, 23)            # #020617 — code block bg
GREEN_CODE = (134, 239, 172)    # #86efac — code string literals

# Hex colors for matplotlib
SLATE_900_HEX = "#0f172a"
SLATE_800_HEX = "#1e293b"
SLATE_700_HEX = "#334155"
TEAL_HEX = "#14b8a6"
TEAL_LIGHT_HEX = "#5eead4"
TEAL_DARK_HEX = "#0d9488"
CORAL_HEX = "#fb7185"
WHITE_HEX = "#ffffff"
SLATE_400_HEX = "#94a3b8"
SLATE_300_HEX = "#cbd5e1"


class DCDHCarouselPDF(FPDF):
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

    def dark_bg(self):
        """Fill page with dark slate background."""
        self.set_fill_color(*SLATE_900)
        self.rect(0, 0, WIDTH, HEIGHT, "F")

    def add_footer(self):
        """Add footer with teal rule and diff-diff wordmark."""
        rule_y = HEIGHT - 28
        self.set_draw_color(*TEAL_DARK)
        self.set_line_width(0.5)
        self.line(50, rule_y, WIDTH - 50, rule_y)

        self.set_font("Helvetica", "B", 12)
        d1 = "diff"
        dash = "-"
        d2 = "diff"
        d1_w = self.get_string_width(d1)
        dash_w = self.get_string_width(dash)
        d2_w = self.get_string_width(d2)
        total_w = d1_w + dash_w + d2_w
        start_x = (WIDTH - total_w) / 2

        self.set_xy(start_x, HEIGHT - 22)
        self.set_text_color(*SLATE_400)
        self.cell(d1_w, 10, d1)
        self.set_text_color(*TEAL)
        self.cell(dash_w, 10, dash)
        self.set_text_color(*SLATE_400)
        self.cell(d2_w, 10, d2)

    # ── Text Helpers ─────────────────────────────────────────────

    def centered_text(self, y, text, size=28, bold=True, color=WHITE,
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
        self.set_text_color(*WHITE)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="R")
        self.set_text_color(*TEAL)
        self.cell(10, 10, "-", align="C")
        self.set_text_color(*WHITE)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="L")

    # ── Equation Rendering ───────────────────────────────────────

    def _render_equations(self, latex_lines, fontsize=26, color=TEAL_HEX):
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

    # ── Treatment Timeline Visual ────────────────────────────────

    def _render_treatment_timeline(self):
        """Render side-by-side treatment timeline comparison.

        Left: typical staggered DiD (all units adopt, never leave).
        Right: dCDH (some join, some leave, same never-treated control).
        """
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(11, 4.5), gridspec_kw={"wspace": 0.35},
        )
        fig.patch.set_facecolor(SLATE_900_HEX)

        n_periods = 10
        periods = np.arange(n_periods)
        bar_height = 0.6

        untreated_color = SLATE_700_HEX
        treated_left = TEAL_DARK_HEX
        treated_right = TEAL_HEX

        # ── Left panel: typical staggered DiD ──
        ax_left.set_facecolor(SLATE_900_HEX)
        ax_left.set_title("Typical staggered DiD",
                          fontsize=13, fontweight="bold",
                          color=SLATE_300_HEX, pad=10,
                          fontfamily="sans-serif")

        left_units = [
            ("Prospect A", 3),   # adopts at t=3
            ("Prospect B", 5),   # adopts at t=5
            ("Prospect C", 7),   # adopts at t=7
            ("Prospect D", None),  # never treated
        ]

        for row, (label, onset) in enumerate(left_units):
            y = len(left_units) - 1 - row
            for t in periods:
                if onset is not None and t >= onset:
                    color = treated_left
                else:
                    color = untreated_color
                ax_left.barh(y, 0.9, left=t, height=bar_height,
                             color=color, edgecolor=SLATE_900_HEX,
                             linewidth=0.5)

            ax_left.text(-0.5, y, label, ha="right", va="center",
                         fontsize=9, color=SLATE_300_HEX,
                         fontfamily="sans-serif")

        ax_left.set_xlim(-0.2, n_periods)
        ax_left.set_ylim(-0.8, len(left_units) - 0.3)
        ax_left.set_xlabel("Time", fontsize=10, color=SLATE_400_HEX,
                           fontfamily="sans-serif")
        ax_left.set_yticks([])
        ax_left.tick_params(axis="x", colors=SLATE_400_HEX, labelsize=8)
        for spine in ax_left.spines.values():
            spine.set_visible(False)

        # "All adopt" annotation — vertically aligned with right panel legend
        ax_left.text(n_periods / 2, -1.8, "All treatment flows one way",
                     ha="center", va="center", fontsize=9,
                     color=SLATE_400_HEX, style="italic",
                     fontfamily="sans-serif")

        # ── Right panel: dCDH ──
        ax_right.set_facecolor(SLATE_900_HEX)
        ax_right.set_title("dCDH",
                           fontsize=13, fontweight="bold",
                           color=TEAL_HEX, pad=10,
                           fontfamily="sans-serif")

        # (label, list of treated periods, transition_type)
        right_units = [
            ("Prospect A", list(range(4, 10)), "join", 4),
            ("Prospect B", list(range(0, 6)), "leave", 6),
            ("Prospect C", list(range(6, 10)), "join", 6),
            ("Prospect D", [], None, None),  # never treated (same control)
        ]

        for row, (label, treated_periods, trans_type, trans_t) in enumerate(
            right_units
        ):
            y = len(right_units) - 1 - row
            for t in periods:
                if t in treated_periods:
                    color = treated_right
                else:
                    color = untreated_color
                ax_right.barh(y, 0.9, left=t, height=bar_height,
                              color=color, edgecolor=SLATE_900_HEX,
                              linewidth=0.5)

            ax_right.text(-0.5, y, label, ha="right", va="center",
                          fontsize=9, color=SLATE_300_HEX,
                          fontfamily="sans-serif")

            # No transition markers — the color blocks tell the story

        ax_right.set_xlim(-0.2, n_periods)
        ax_right.set_ylim(-0.8, len(right_units) - 0.3)
        ax_right.set_xlabel("Time", fontsize=10, color=SLATE_400_HEX,
                            fontfamily="sans-serif")
        ax_right.set_yticks([])
        ax_right.tick_params(axis="x", colors=SLATE_400_HEX, labelsize=8)
        for spine in ax_right.spines.values():
            spine.set_visible(False)

        # Legend below right panel
        join_patch = mpatches.Patch(color=TEAL_HEX, label="Targeted")
        leave_patch = mpatches.Patch(color=untreated_color,
                                     label="Not targeted")
        ax_right.legend(handles=[join_patch, leave_patch],
                        loc="lower center", bbox_to_anchor=(0.5, -0.25),
                        ncol=2, fontsize=9, frameon=False,
                        labelcolor=SLATE_300_HEX)

        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.90,
                            wspace=0.35)

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15,
                    facecolor=SLATE_900_HEX)
        plt.close(fig)

        with PILImage.open(path) as img:
            pw, ph = img.size

        self._temp_files.append(path)
        return path, pw, ph

    # ── Event Study Plot ─────────────────────────────────────────

    def _render_event_study(self):
        """Render dynamic event study plot with placebos.

        Uses hardcoded ATT values (no runtime library dependency).
        Placebos at negative horizons near zero, effects at positive horizons.
        """
        horizons = np.arange(-4, 6)
        # Placebos: near zero (pre-trend validation)
        atts_pre = [0.08, -0.05, 0.12, -0.02]
        # Post-treatment: positive effects, growing then stabilizing
        atts_post = [0.0, 1.40, 1.85, 2.10, 1.95, 2.05]
        atts = np.array(atts_pre + atts_post)
        ses = np.array([0.28, 0.25, 0.22, 0.20,
                        0.0, 0.24, 0.26, 0.28, 0.32, 0.35])

        # 95% pointwise CI
        ci_lower = atts - 1.96 * ses
        ci_upper = atts + 1.96 * ses

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(SLATE_900_HEX)
        ax.set_facecolor(SLATE_900_HEX)

        # Shade pre and post regions
        ax.axvspan(-4.5, -0.5, alpha=0.08, color=SLATE_400_HEX)
        ax.axvspan(-0.5, 5.5, alpha=0.05, color=TEAL_HEX)

        # Region labels at top of plot area
        y_top = max(ci_upper) + 0.45
        ax.text(-2.5, y_top, "Placebos",
                ha="center", va="bottom", fontsize=12,
                color=SLATE_400_HEX, style="italic",
                fontfamily="sans-serif")
        ax.text(2.5, y_top, "Treatment effects",
                ha="center", va="bottom", fontsize=12,
                color=TEAL_LIGHT_HEX, style="italic",
                fontfamily="sans-serif")

        # CI band (skip horizon 0 which is the reference)
        mask = horizons != 0
        ax.fill_between(horizons[mask], ci_lower[mask], ci_upper[mask],
                        color=TEAL_LIGHT_HEX, alpha=0.2)

        # ATT line + dots
        ax.plot(horizons[mask], atts[mask], "o-", color=TEAL_HEX,
                linewidth=2.5, markersize=7, zorder=5)

        # Reference point at horizon 0 (normalized to zero)
        ax.plot(0, 0, "D", color=SLATE_400_HEX, markersize=8, zorder=6)

        # Reference lines
        ax.axhline(0, color=SLATE_400_HEX, linestyle="--", linewidth=0.8)
        ax.axvline(-0.5, color=CORAL_HEX, linestyle=":", linewidth=1.5,
                   label="First treatment change", zorder=3)

        ax.set_xlabel("Horizon (l)", fontsize=12, color=SLATE_300_HEX,
                      fontfamily="sans-serif")
        ax.set_ylabel("Effect estimate", fontsize=12, color=SLATE_300_HEX,
                      fontfamily="sans-serif")

        ax.tick_params(colors=SLATE_400_HEX)
        ax.set_xticks(horizons)
        for spine in ax.spines.values():
            spine.set_color(SLATE_700_HEX)

        ax.legend(loc="upper left", fontsize=10, frameon=False,
                  labelcolor=SLATE_300_HEX)

        fig.tight_layout()

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1,
                    facecolor=SLATE_900_HEX)
        plt.close(fig)

        with PILImage.open(path) as img:
            pw, ph = img.size

        self._temp_files.append(path)
        return path, pw, ph

    # ── Code Block ───────────────────────────────────────────────

    def _add_code_block(self, x, y, w, token_lines, font_size=12,
                        line_height=11):
        """Render syntax-highlighted code on a dark panel."""
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
        """Slide 1: Hook - When Treatment Isn't Permanent.

        Claims & sources:
        - "When Treatment Isn't Permanent": captures the non-absorbing
          treatment assumption relaxed by dCDH. REGISTRY.md line 466:
          "This is the only modern staggered estimator in the library
          that handles non-absorbing (reversible) treatments"
        - "de Chaisemartin & D'Haultfoeuille (2024)": REGISTRY.md lines
          462-464, NBER WP 29873 revised 2024
        - "Joiners and leavers in one model": REGISTRY.md lines 486-491,
          DID_{+,t} for joiners, DID_{-,t} for leavers
        - "Dynamic event study with pre-trend placebos": REGISTRY.md
          lines 518-534, DID_l and DID^{pl}_l
        - "Part of a 16-estimator DiD toolkit": counted from __init__.py,
          16 treatment-effect estimators with fit() including dCDH
        """
        self.add_page()
        self.dark_bg()

        self.draw_split_logo(55, size=56)

        # Hook - two lines
        self.centered_text(125, "When Treatment", size=32)
        self.centered_text(155, "Isn't Permanent.", size=32, color=TEAL)

        # Teasers
        teasers = [
            "de Chaisemartin & D'Haultfoeuille (2024)",
            "Joiners and leavers in one model",
            "Dynamic event study with pre-trend placebos",
        ]
        y_start = 205
        for i, teaser in enumerate(teasers):
            self.set_xy(0, y_start + i * 20)
            self.set_font("Helvetica", "", 15)
            self.set_text_color(*SLATE_400)
            self.cell(WIDTH, 10, teaser, align="C")

        self.add_footer()

    def slide_02_problem(self):
        """Slide 2: The Permanent Treatment Assumption.

        Claims & sources:
        - "Most staggered DiD estimators assume treatment is absorbing":
          Callaway-Sant'Anna (staggered.py) requires first_treat to be
          permanent. Sun-Abraham, Imputation, TwoStage, Stacked all
          require absorbing treatment (one-way adoption).
        - Visual contrast: left panel (absorbing) vs right (dCDH).
          Right panel uses marketing-flavored labels per storyboard.
        - "Requires never-treated or never-switching controls":
          REGISTRY.md line 476, stable controls needed for both
          joiner and leaver estimation.
        """
        self.add_page()
        self.dark_bg()

        self.centered_text(25, "The Permanent Treatment", size=32)
        self.centered_text(55, "Assumption", size=32, color=CORAL)

        # Subtitle
        self.centered_text(90,
                           "Most staggered DiD estimators assume absorbing treatment.",
                           size=15, bold=False, color=SLATE_400)

        # Treatment timeline visual
        plot_path, ppw, pph = self._render_treatment_timeline()
        plot_w = WIDTH * 0.92
        plot_aspect = pph / ppw
        plot_h = plot_w * plot_aspect
        plot_x = (WIDTH - plot_w) / 2
        plot_y = 128

        self.image(plot_path, plot_x, plot_y, plot_w)

        # Annotation below
        ann_y = plot_y + plot_h + 10
        self.centered_text(ann_y,
                           "Some units adopt while others discontinue.",
                           size=15, bold=False, color=SLATE_300)
        self.centered_text(ann_y + 20,
                           "You need both directions.",
                           size=16, bold=True, color=TEAL)

        self.add_footer()

    def slide_03_real_world(self):
        """Slide 3: Where Treatment Isn't Permanent.

        Claims & sources:
        - Marketing campaigns: natural fit for dCDH. New prospects
          get targeted (joiners) while others are untargeted (leavers).
        - Policy adoption/repeal: states can adopt AND repeal policies.
          dCDH paper motivates with policy examples.
        - Clinical: patients start and discontinue treatments.
        - Pricing: promotions launch and sunset across stores.
        All are standard motivating examples for non-absorbing
        treatment in the dCDH literature.
        """
        self.add_page()
        self.dark_bg()

        self.centered_text(25, "Where Treatment", size=34)
        self.centered_text(55, "Isn't Permanent", size=34, color=TEAL)

        margin = 35
        box_w = WIDTH - margin * 2
        box_h = 42
        gap = 5
        start_y = 85
        bar_w = 4

        scenarios = [
            ("Marketing Campaigns",
             "Target new prospects, untarget others - measure both"),
            ("Policy Adoption & Repeal",
             "States adopt policies while others repeal them"),
            ("Clinical Treatments",
             "Patients start a drug while others discontinue"),
            ("Pricing & Promotions",
             "Promotions launch in some stores, sunset in others"),
        ]

        for i, (title, desc) in enumerate(scenarios):
            by = start_y + i * (box_h + gap)

            # Card
            self.set_fill_color(*SLATE_800)
            self.set_draw_color(*SLATE_700)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, box_h, "DF")

            # Teal accent bar
            self.set_fill_color(*TEAL)
            self.rect(margin, by, bar_w, box_h, "F")

            # Title
            self.set_xy(margin + bar_w + 12, by + 7)
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(*WHITE)
            self.cell(box_w - bar_w - 24, 10, title)

            # Description
            self.set_xy(margin + bar_w + 12, by + 24)
            self.set_font("Helvetica", "", 13)
            self.set_text_color(*SLATE_400)
            self.cell(box_w - bar_w - 24, 10, desc)

        self.add_footer()

    def slide_04_method(self):
        """Slide 4: The dCDH Method.

        Claims & sources:
        - de Chaisemartin & D'Haultfoeuille (2020, 2024):
          REGISTRY.md lines 462-464. AER 2020 + NBER WP 29873.
        - DID_M equation: REGISTRY.md lines 498-500.
          DID_M = (1/N_S) sum (N_{1,0,t} DID_{+,t} + N_{0,1,t} DID_{-,t})
        - Joiners vs stable-untreated: REGISTRY.md line 491,
          DID_{+,t} compares joiners against stable_0(t)
        - Leavers vs stable-treated: REGISTRY.md line 490,
          DID_{-,t} compares stable_1(t) against leavers
        - R DIDmultiplegtDYN: R package by the paper authors.
          Stata did_multiplegt_dyn: Stata implementation.
        """
        self.add_page()
        self.dark_bg()

        self.centered_text(25, "The dCDH Estimator", size=36)

        # Citation
        self.centered_text(62,
                           "de Chaisemartin & D'Haultfoeuille (2020, 2024)",
                           size=15, bold=False, italic=True, color=SLATE_400)

        # DID_M equation
        eq_path, epw, eph = self._render_equations(
            [r"$\mathrm{DID}_M = \frac{1}{N_S}"
             r" \sum_{t \geq 2}"
             r" \left( N_{1,0,t} \cdot \mathrm{DID}_{+,t}"
             r" + N_{0,1,t} \cdot \mathrm{DID}_{-,t} \right)$"],
            fontsize=22,
        )
        eq_h = self._place_equation_centered(eq_path, epw, eph, 100,
                                             max_w=230)

        # Two explanation cards: joiners and leavers
        margin = 38
        card_w = (WIDTH - margin * 2 - 10) / 2
        card_h = 58
        card_y = 100 + eq_h + 16

        # Joiners card
        cx = margin
        self.set_fill_color(*SLATE_800)
        self.set_draw_color(*TEAL)
        self.set_line_width(0.8)
        self.rect(cx, card_y, card_w, card_h, "DF")

        self.set_xy(cx + 10, card_y + 8)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*TEAL)
        self.cell(card_w - 20, 10, "Joiners (DID+)")

        self.set_xy(cx + 10, card_y + 26)
        self.set_font("Helvetica", "", 12)
        self.set_text_color(*SLATE_300)
        self.cell(card_w - 20, 10, "Units switching ON")

        self.set_xy(cx + 10, card_y + 40)
        self.set_text_color(*SLATE_400)
        self.cell(card_w - 20, 10, "vs stable-untreated controls")

        # Leavers card
        cx = margin + card_w + 10
        self.set_fill_color(*SLATE_800)
        self.set_draw_color(*CORAL)
        self.set_line_width(0.8)
        self.rect(cx, card_y, card_w, card_h, "DF")

        self.set_xy(cx + 10, card_y + 8)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*CORAL)
        self.cell(card_w - 20, 10, "Leavers (DID-)")

        self.set_xy(cx + 10, card_y + 26)
        self.set_font("Helvetica", "", 12)
        self.set_text_color(*SLATE_300)
        self.cell(card_w - 20, 10, "Units switching OFF")

        self.set_xy(cx + 10, card_y + 40)
        self.set_text_color(*SLATE_400)
        self.cell(card_w - 20, 10, "vs stable-treated controls")

        # Bottom note
        note_y = card_y + card_h + 16
        self.centered_text(note_y,
                           "Both directions estimated jointly,",
                           size=16, bold=False, color=SLATE_300)
        self.centered_text(note_y + 18,
                           "weighted by the number of switchers in each period.",
                           size=16, bold=False, color=SLATE_300)

        self.add_footer()

    def slide_05_features(self):
        """Slide 5: What Ships With It.

        Claims & sources:
        - Multi-horizon DID_l: REGISTRY.md lines 518-528
        - Pre-trend placebos: REGISTRY.md line 534
        - Covariate adjustment DID^X: REGISTRY.md, Web Appendix Section 1.2
        - Group-specific trends: REGISTRY.md, Web Appendix Section 1.3
        - Non-binary treatment: REGISTRY.md line 471, treatment supports
          ordinal or continuous values
        - HonestDiD sensitivity: honest_did.py, Rambachan & Roth (2023)
        - Sup-t confidence bands: chaisemartin_dhaultfoeuille.py,
          simultaneous inference across horizons
        """
        self.add_page()
        self.dark_bg()

        self.centered_text(25, "What Ships With It", size=36)

        margin = 30
        grid_gap = 8
        card_w = (WIDTH - margin * 2 - grid_gap) / 2
        card_h = 50
        start_y = 72

        features = [
            ("Dynamic Event Study",
             "Multi-horizon DID_l\nfor l = 1..L_max"),
            ("Pre-Trend Placebos",
             "DID^pl_l tests consistency\nwith parallel trends"),
            ("Covariate Adjustment",
             "DID^X residualization\nfor covariates"),
            ("Group-Specific Trends",
             "Linear and nonparametric\ntrend removal"),
            ("Non-Binary Treatment",
             "Ordinal or continuous\ntreatment magnitudes"),
            ("HonestDiD Sensitivity",
             "Rambachan-Roth bounds\non pre-trends"),
            ("Sup-t Bands",
             "Simultaneous inference\nacross all horizons"),
            ("Multiplier Bootstrap",
             "Rademacher, Mammen,\nor Webb weights"),
        ]

        for idx, (title, desc) in enumerate(features):
            row = idx // 2
            col = idx % 2
            cx = margin + col * (card_w + grid_gap)
            cy = start_y + row * (card_h + grid_gap)

            # Card
            self.set_fill_color(*SLATE_800)
            self.set_draw_color(*SLATE_700)
            self.set_line_width(0.5)
            self.rect(cx, cy, card_w, card_h, "DF")

            # Title
            self.set_xy(cx + 10, cy + 8)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*TEAL)
            self.cell(card_w - 20, 10, title)

            # Description (handle line breaks)
            desc_lines = desc.split("\n")
            for j, line in enumerate(desc_lines):
                self.set_xy(cx + 10, cy + 24 + j * 12)
                self.set_font("Helvetica", "", 11)
                self.set_text_color(*SLATE_400)
                self.cell(card_w - 20, 10, line)

        self.add_footer()

    def slide_06_event_study(self):
        """Slide 6: Dynamic Event Study Plot.

        Claims & sources:
        - Event study aggregation: REGISTRY.md lines 518-528,
          DID_l for l = 1..L_max
        - Placebos: REGISTRY.md line 534, DID^{pl}_l for pre-trend
          validation
        - Plot shows synthetic data with hardcoded ATT values mimicking
          a well-behaved dataset with true effect ~2.0 and flat placebos.
          Horizon 0 is the reference period (normalized to zero).
        """
        self.add_page()
        self.dark_bg()

        self.centered_text(20, "Dynamic Event Study", size=36)

        # Event study plot
        plot_path, ppw, pph = self._render_event_study()
        plot_w = WIDTH * 0.88
        plot_aspect = pph / ppw
        plot_h = plot_w * plot_aspect
        plot_x = (WIDTH - plot_w) / 2
        plot_y = 58
        self.image(plot_path, plot_x, plot_y, plot_w)

        # Annotations below
        ann_y = plot_y + plot_h + 8
        self.centered_text(ann_y,
                           "Placebos near zero: consistent with parallel trends",
                           size=15, bold=False, color=SLATE_400)
        self.centered_text(ann_y + 18,
                           "Post-treatment: significant, persistent effects",
                           size=15, bold=True, color=TEAL)

        self.add_footer()

    def slide_07_code(self):
        """Slide 7: The Code.

        Claims & sources:
        - ChaisemartinDHaultfoeuille: __init__.py imports from
          chaisemartin_dhaultfoeuille.py
        - fit() params: chaisemartin_dhaultfoeuille.py lines 472-478
          (data, outcome, group, time, treatment)
        - L_max parameter: constructor param,
          chaisemartin_dhaultfoeuille.py line 389
        - placebo parameter: constructor param, default True,
          chaisemartin_dhaultfoeuille.py line 396
        - plot_event_study(): available on results object,
          chaisemartin_dhaultfoeuille_results.py
        """
        self.add_page()
        self.dark_bg()

        self.centered_text(25, "The Code", size=36)
        self.centered_text(58,
                           "Same sklearn-like API as every diff-diff estimator",
                           size=15, bold=False, color=SLATE_400)

        margin = 24
        code_y = 82

        token_lines = [
            [("from", TEAL), (" diff_diff ", WHITE),
             ("import", TEAL), (" (", WHITE)],
            [("    ChaisemartinDHaultfoeuille ", WHITE),
             ("as", TEAL), (" DCDH,", WHITE)],
            [("    plot_event_study", WHITE)],
            [(")", WHITE)],
            [],  # blank
            [("est", WHITE), (" = ", TEAL),
             ("DCDH", GREEN_CODE), ("()", WHITE)],
            [("results", WHITE), (" = ", TEAL),
             ("est.fit(", WHITE)],
            [("    data,", WHITE)],
            [("    ", WHITE), ("outcome", WHITE), ("=", TEAL),
             ("'conversions'", GREEN_CODE), (",", WHITE)],
            [("    ", WHITE), ("group", WHITE), ("=", TEAL),
             ("'prospect_id'", GREEN_CODE), (",", WHITE)],
            [("    ", WHITE), ("time", WHITE), ("=", TEAL),
             ("'week'", GREEN_CODE), (",", WHITE)],
            [("    ", WHITE), ("treatment", WHITE), ("=", TEAL),
             ("'targeted'", GREEN_CODE), (",", WHITE)],
            [("    ", WHITE), ("L_max", WHITE), ("=", TEAL),
             ("3", TEAL_LIGHT), (", ", WHITE),
             ("placebo", WHITE), ("=", TEAL),
             ("True", TEAL_LIGHT), (")", WHITE)],
            [],  # blank
            [("plot_event_study", WHITE), ("(results)", SLATE_400)],
        ]

        code_h = self._add_code_block(
            margin, code_y, WIDTH - margin * 2, token_lines,
        )

        # Subtitle — keep above footer (rule at HEIGHT-28)
        sub_y = min(code_y + code_h + 18, HEIGHT - 48)
        self.centered_text(sub_y,
                           "Placebos, event study, bootstrap - all from one fit() call.",
                           size=14, bold=False, color=SLATE_400)

        self.add_footer()

    def slide_08_validation(self):
        """Slide 8: Validation.

        Claims & sources:
        - 245 tests: pytest --co -q on
          test_chaisemartin_dhaultfoeuille.py (230) +
          test_chaisemartin_dhaultfoeuille_parity.py (15) = 245.
          (test_honest_did.py dCDH tests are part of the 230 count.)
        - R DIDmultiplegtDYN parity:
          test_chaisemartin_dhaultfoeuille_parity.py, 15 parity tests
          covering point estimates, SEs, covariates, trends
        - HonestDiD integration: honest_did.py, Rambachan & Roth (2023)
          sensitivity analysis on dCDH placebo surface
        - Multiplier bootstrap: chaisemartin_dhaultfoeuille.py,
          Rademacher/Mammen/Webb weight distributions
        """
        self.add_page()
        self.dark_bg()

        self.centered_text(30, "Validated Against R", size=36)

        # Hero claim
        self.centered_text(72,
                           "Point estimates match R's DIDmultiplegtDYN.",
                           size=17, bold=False, color=SLATE_300)

        # Validation items
        margin = 38
        box_w = WIDTH - margin * 2
        box_h = 42
        gap = 5
        start_y = 108
        bar_w = 4

        items = [
            ("Point Estimate Parity",
             "ATT validated against R reference outputs"),
            ("SE Parity (pure-direction)",
             "Analytical SEs match R on joiners-only and leavers-only"),
            ("HonestDiD Integration",
             "Rambachan-Roth sensitivity on the placebo surface"),
            ("Multiplier Bootstrap",
             "Rademacher, Mammen, and Webb weight distributions"),
        ]

        for i, (title, desc) in enumerate(items):
            by = start_y + i * (box_h + gap)

            # Card
            self.set_fill_color(*SLATE_800)
            self.set_draw_color(*SLATE_700)
            self.set_line_width(0.5)
            self.rect(margin, by, box_w, box_h, "DF")

            # Teal accent bar
            self.set_fill_color(*TEAL)
            self.rect(margin, by, bar_w, box_h, "F")

            # Title
            self.set_xy(margin + bar_w + 12, by + 7)
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(*WHITE)
            self.cell(box_w - bar_w - 24, 10, title)

            # Description
            self.set_xy(margin + bar_w + 12, by + 24)
            self.set_font("Helvetica", "", 12)
            self.set_text_color(*SLATE_400)
            self.cell(box_w - bar_w - 24, 10, desc)

        self.add_footer()

    def slide_09_cta(self):
        """Slide 9: CTA - Get Started."""
        self.add_page()
        self.dark_bg()

        self.centered_text(50, "Get Started", size=42)

        # pip install badge
        badge_w = 210
        badge_h = 36
        badge_x = (WIDTH - badge_w) / 2
        badge_y = 115
        self.set_fill_color(*TEAL_DARK)
        self.rect(badge_x, badge_y, badge_w, badge_h, "F")

        self.set_xy(badge_x, badge_y + 9)
        self.set_font("Courier", "B", 15)
        self.set_text_color(*WHITE)
        self.cell(badge_w, 16, "$ pip install --upgrade diff-diff",
                  align="C")

        # Links
        self.centered_text(178, "github.com/igerber/diff-diff",
                           size=18, color=TEAL)

        # Wordmark
        self.draw_split_logo(235, size=28)

        # Subtitle
        self.centered_text(262, "Difference-in-Differences for Python",
                           size=15, bold=False, color=SLATE_400)

        self.add_footer()


def main():
    pdf = DCDHCarouselPDF()
    try:
        pdf.slide_01_hook()
        pdf.slide_02_problem()
        pdf.slide_03_real_world()
        pdf.slide_04_method()
        pdf.slide_05_features()
        pdf.slide_06_event_study()
        pdf.slide_07_code()
        pdf.slide_08_validation()
        pdf.slide_09_cta()

        output_path = Path(__file__).parent / "diff-diff-dcdh-carousel.pdf"
        pdf.output(str(output_path))
        print(f"PDF saved to: {output_path}")
    finally:
        pdf.cleanup()


if __name__ == "__main__":
    main()
