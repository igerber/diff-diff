#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for diff-diff library."""

from fpdf import FPDF

# LinkedIn carousel dimensions (4:5 aspect ratio)
WIDTH = 270  # mm
HEIGHT = 337.5  # mm

# Colors - LIGHTER THEME
LIGHT_BLUE_BG = (235, 245, 255)  # Light blue-white background
MID_BLUE = (59, 130, 246)  # #3b82f6 - Primary blue
DARK_BLUE = (30, 64, 175)  # #1e40af - Dark blue for text
NAVY = (15, 23, 42)  # #0f172a - For main text
BLUE_ACCENT = (37, 99, 235)  # #2563eb
WHITE = (255, 255, 255)
RED = (220, 38, 38)  # #dc2626
GREEN = (22, 163, 74)  # #16a34a
GRAY = (100, 116, 139)  # #64748b
LIGHT_GRAY = (148, 163, 184)  # #94a3b8


class CarouselPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format=(WIDTH, HEIGHT))
        self.set_auto_page_break(False)
        self.page_count = 0

    def add_connector_graphic(self, position="right"):
        """Add a decorative connector graphic to bottom corner.

        position: 'right' for bottom-right, 'left' for bottom-left
        These connect visually when slides are side-by-side.
        """
        # Draw a series of concentric arcs/circles
        if position == "right":
            cx = WIDTH + 20  # Center off the right edge
            cy = HEIGHT - 40
        else:
            cx = -20  # Center off the left edge
            cy = HEIGHT - 40

        # Draw concentric circular arcs
        self.set_draw_color(*MID_BLUE)
        for i, radius in enumerate([60, 80, 100]):
            alpha = 0.3 - i * 0.08  # Decreasing opacity effect via line weight
            self.set_line_width(2.5 - i * 0.5)

            # Draw arc segments
            import math
            segments = 30
            if position == "right":
                start_angle = math.pi * 0.5   # 90 degrees
                end_angle = math.pi * 1.0     # 180 degrees
            else:
                start_angle = 0               # 0 degrees
                end_angle = math.pi * 0.5     # 90 degrees

            for j in range(segments):
                t1 = start_angle + (end_angle - start_angle) * j / segments
                t2 = start_angle + (end_angle - start_angle) * (j + 1) / segments
                x1 = cx + radius * math.cos(t1)
                y1 = cy + radius * math.sin(t1)
                x2 = cx + radius * math.cos(t2)
                y2 = cy + radius * math.sin(t2)
                self.line(x1, y1, x2, y2)

        # Add a few dots for extra visual interest
        self.set_fill_color(*MID_BLUE)
        dot_positions = [(35, HEIGHT - 60), (50, HEIGHT - 45), (30, HEIGHT - 35)] if position == "right" else [(WIDTH - 35, HEIGHT - 60), (WIDTH - 50, HEIGHT - 45), (WIDTH - 30, HEIGHT - 35)]
        for i, (dx, dy) in enumerate(dot_positions):
            dot_radius = 3 - i * 0.5
            self.ellipse(dx - dot_radius, dy - dot_radius, dot_radius * 2, dot_radius * 2, "F")

    def light_gradient_background(self):
        """Draw light gradient background."""
        steps = 50
        for i in range(steps):
            ratio = i / steps
            # Light blue at top fading to white at bottom
            r = int(225 + (255 - 225) * ratio)
            g = int(240 + (255 - 240) * ratio)
            b = int(255)
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

    def slide_hook(self):
        """Slide 1: Hook"""
        self.add_page()
        self.light_gradient_background()

        # Logo
        self.set_xy(0, 35)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="R")
        self.set_text_color(*MID_BLUE)
        self.cell(10, 10, "-", align="C")
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 5, 10, "diff", align="L")

        # Main headline
        self.centered_text(75, "Difference-in-", size=38)
        self.centered_text(100, "Differences.", size=38)
        self.centered_text(135, "Now in Python.", size=38, color=MID_BLUE)

        # Subheadline
        self.centered_text(175, "Up to 2,000x faster than R", size=22, color=BLUE_ACCENT)

        # pip install box
        box_width = 200
        box_x = (WIDTH - box_width) / 2
        self.set_fill_color(*MID_BLUE)
        self.rect(box_x, 210, box_width, 35, "F")

        self.set_xy(box_x, 218)
        self.set_font("Courier", "B", 20)
        self.set_text_color(*WHITE)
        self.cell(box_width, 12, "$ pip install diff-diff", align="C")

        self.add_connector_graphic("right")
        self.add_footer()

    def slide_problem(self):
        """Slide 2: The Problem"""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(45, "DiD in Python", size=38)
        self.centered_text(78, "Was Broken", size=38, color=RED)

        items = [
            "Fragmented packages",
            "Missing modern methods",
            "Slow performance",
            "R required for serious work"
        ]

        y_start = 130
        for i, item in enumerate(items):
            self.add_list_item(y_start + i * 32, "X", item, RED, text_size=24)

        self.centered_text(285, "Researchers deserved better.", size=18, bold=False, color=GRAY)
        self.add_connector_graphic("left")
        self.add_footer()

    def slide_solution(self):
        """Slide 3: The Solution"""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(45, "Introducing", size=38)
        self.centered_text(78, "diff-diff", size=38, color=MID_BLUE)

        items = [
            "Complete DiD toolkit",
            "sklearn-like API",
            "All modern methods",
            "Blazing fast"
        ]

        y_start = 130
        for i, item in enumerate(items):
            self.add_list_item(y_start + i * 32, "+", item, GREEN, text_size=24)

        self.centered_text(285, "One library. Everything you need.", size=18, bold=False, color=GRAY)
        self.add_connector_graphic("right")
        self.add_footer()

    def slide_performance(self):
        """Slide 4: Performance - FIXED ALIGNMENT"""
        self.add_page()
        self.light_gradient_background()

        # Line 1: "R: 24 minutes" - properly centered
        self.set_xy(0, 40)
        self.set_font("Helvetica", "B", 32)
        self.set_text_color(*NAVY)
        # Calculate widths for proper centering
        r_text = "R: "
        minutes_text = "24 minutes"
        self.set_font("Helvetica", "B", 32)

        # Center the combined text
        total_text = "R: 24 minutes"
        self.cell(WIDTH, 18, "", align="C")  # spacer
        self.set_xy(0, 40)
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 - 10, 18, "R:", align="R")
        self.set_text_color(*RED)
        self.cell(WIDTH / 2 + 10, 18, " 24 minutes", align="L")

        # Line 2: "diff-diff: 2.6 seconds" - properly centered
        self.set_xy(0, 75)
        self.set_text_color(*NAVY)
        self.cell(WIDTH / 2 + 15, 18, "diff-diff:", align="R")
        self.set_text_color(*MID_BLUE)
        self.cell(WIDTH / 2 - 15, 18, " 2.6 seconds", align="L")

        # Bar chart
        bar_data = [
            ("Synthetic DiD", 100, "2,234x faster"),
            ("Basic DiD", 45, "18x faster"),
            ("Callaway-SA", 40, "14x faster"),
        ]

        margin = 35
        bar_height = 28
        y_start = 130

        for i, (label, pct, speedup) in enumerate(bar_data):
            y = y_start + i * 50

            # Label
            self.set_xy(margin, y + 5)
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(*NAVY)
            self.cell(70, bar_height - 10, label, align="R")

            # Bar background
            bar_x = margin + 75
            bar_width = WIDTH - margin - bar_x - 10
            self.set_fill_color(220, 230, 245)
            self.rect(bar_x, y, bar_width, bar_height, "F")

            # Bar fill
            fill_width = bar_width * pct / 100
            self.set_fill_color(*MID_BLUE)
            self.rect(bar_x, y, fill_width, bar_height, "F")

            # Speedup text
            self.set_xy(bar_x, y + 5)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*WHITE)
            self.cell(fill_width - 8, bar_height - 10, speedup, align="R")

        self.centered_text(295, "Benchmarked at 10K scale vs R packages", size=16, bold=False, color=GRAY)
        self.add_connector_graphic("left")
        self.add_footer()

    def slide_methods(self):
        """Slide 5: Methods"""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(35, "Every Method", size=38)
        self.centered_text(68, "You Need", size=38, color=MID_BLUE)

        methods = [
            ("Basic DiD / TWFE", "Classic 2x2 and panel"),
            ("Callaway-Sant'Anna", "Staggered adoption (2021)"),
            ("Sun-Abraham", "Interaction-weighted (2021)"),
            ("Synthetic DiD", "Arkhangelsky et al. (2021)"),
            ("Triple Difference", "DDD with proper covariates"),
            ("Honest DiD", "Rambachan-Roth sensitivity"),
        ]

        margin = 25
        box_width = (WIDTH - margin * 3) / 2
        box_height = 55

        for i, (title, desc) in enumerate(methods):
            col = i % 2
            row = i // 2
            x = margin + col * (box_width + margin)
            y = 105 + row * (box_height + 15)

            # Box
            self.set_fill_color(*WHITE)
            self.set_draw_color(*MID_BLUE)
            self.set_line_width(0.8)
            self.rect(x, y, box_width, box_height, "DF")

            # Title
            self.set_xy(x + 5, y + 12)
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(*MID_BLUE)
            self.cell(box_width - 10, 10, title, align="C")

            # Description
            self.set_xy(x + 5, y + 30)
            self.set_font("Helvetica", "", 13)
            self.set_text_color(*GRAY)
            self.cell(box_width - 10, 10, desc, align="C")

        self.add_connector_graphic("right")
        self.add_footer()

    def slide_code(self):
        """Slide 6: Code Example"""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(30, "Clean, Pythonic API", size=36)

        # Code block
        margin = 25
        code_y = 75
        self.set_fill_color(30, 41, 59)  # Dark slate for code
        self.rect(margin, code_y, WIDTH - margin * 2, 180, "F")

        # Simplified code rendering
        code_text = """from diff_diff import CallawaySantAnna

# Staggered DiD in 5 lines
cs = CallawaySantAnna()
results = cs.fit(
    data,
    outcome='sales',
    unit='firm_id',
    time='year',
    first_treat='first_treat'
)

results.print_summary()"""

        self.set_font("Courier", "", 14)
        self.set_text_color(*WHITE)
        for i, line in enumerate(code_text.split('\n')):
            self.set_xy(margin + 15, code_y + 15 + i * 12)
            self.cell(0, 10, line)

        self.centered_text(280, "sklearn-like fit() + statsmodels-style output", size=16, bold=False, color=GRAY)
        self.add_connector_graphic("left")
        self.add_footer()

    def slide_validated(self):
        """Slide 7: Validated"""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(30, "Validated. Trusted.", size=34)
        self.centered_text(58, "Production-Ready.", size=34, color=MID_BLUE)

        # Table
        margin = 25
        table_y = 100
        row_height = 38

        data = [
            ("Point estimates vs R", "Identical (10+ decimals)", True),
            ("Standard errors", "Within 1-3%", True),
            ("R packages tested", "did, synthdid, fixest", False),
            ("Real-world validation", "MPDTA dataset", True),
        ]

        # Header
        self.set_fill_color(*MID_BLUE)
        self.rect(margin, table_y, WIDTH - margin * 2, row_height, "F")

        col1_width = (WIDTH - margin * 2) * 0.5
        col2_width = (WIDTH - margin * 2) * 0.5

        self.set_xy(margin, table_y + 10)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*WHITE)
        self.cell(col1_width, 14, "Comparison", align="C")
        self.cell(col2_width, 14, "Result", align="C")

        # Rows
        for i, (label, value, is_check) in enumerate(data):
            y = table_y + row_height * (i + 1)

            if i % 2 == 0:
                self.set_fill_color(245, 248, 255)
            else:
                self.set_fill_color(*WHITE)
            self.rect(margin, y, WIDTH - margin * 2, row_height, "F")

            self.set_xy(margin + 10, y + 10)
            self.set_font("Helvetica", "", 15)
            self.set_text_color(*NAVY)
            self.cell(col1_width - 10, 14, label)

            self.set_xy(margin + col1_width, y + 10)
            self.set_font("Helvetica", "B", 15)
            if is_check:
                self.set_text_color(*GREEN)
            else:
                self.set_text_color(*NAVY)
            self.cell(col2_width - 10, 14, value, align="C")

        self.centered_text(295, "Academic-grade accuracy. No compromises.", size=16, bold=False, color=GRAY)
        self.add_connector_graphic("right")
        self.add_footer()

    def slide_features(self):
        """Slide 8: Features"""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(30, "Everything", size=38)
        self.centered_text(63, "Included", size=38, color=MID_BLUE)

        features = [
            "Robust & Cluster SEs",
            "Wild Bootstrap",
            "Event Study Plots",
            "Parallel Trends Tests",
            "Bacon Decomposition",
            "Power Analysis",
            "Pre-trends Power",
            "Placebo Tests",
        ]

        margin = 35
        col_width = (WIDTH - margin * 3) / 2

        for i, feature in enumerate(features):
            col = i % 2
            row = i // 2
            x = margin + col * (col_width + margin)
            y = 110 + row * 42

            # Icon
            self.set_xy(x, y)
            self.set_font("Helvetica", "B", 20)
            self.set_text_color(*MID_BLUE)
            self.cell(20, 14, "*")

            # Text
            self.set_text_color(*NAVY)
            self.set_font("Helvetica", "", 18)
            self.cell(col_width - 20, 14, feature)

            # Underline
            self.set_draw_color(200, 215, 235)
            self.set_line_width(0.5)
            self.line(x, y + 22, x + col_width, y + 22)

        self.centered_text(295, "Publication-ready visualizations included", size=16, bold=False, color=GRAY)
        self.add_connector_graphic("left")
        self.add_footer()

    def slide_audience(self):
        """Slide 9: Audience"""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(30, "Built for", size=38)
        self.centered_text(63, "Researchers", size=38, color=MID_BLUE)

        audiences = [
            ("|||", "Applied Economists"),
            ("^^^", "Policy Researchers"),
            ("oOo", "Data Scientists"),
            (">_", "Python-First Teams"),
        ]

        margin = 30
        box_width = (WIDTH - margin * 3) / 2
        box_height = 80

        for i, (icon, title) in enumerate(audiences):
            col = i % 2
            row = i // 2
            x = margin + col * (box_width + margin)
            y = 100 + row * (box_height + 20)

            # Box
            self.set_fill_color(*WHITE)
            self.set_draw_color(*MID_BLUE)
            self.set_line_width(0.8)
            self.rect(x, y, box_width, box_height, "DF")

            # Icon
            self.set_xy(x, y + 15)
            self.set_font("Courier", "B", 32)
            self.set_text_color(*MID_BLUE)
            self.cell(box_width, 18, icon, align="C")

            # Title
            self.set_xy(x, y + 50)
            self.set_font("Helvetica", "B", 17)
            self.set_text_color(*NAVY)
            self.cell(box_width, 14, title, align="C")

        self.centered_text(300, "Finally escape R for causal inference", size=18, bold=False, color=GRAY)
        self.add_connector_graphic("right")
        self.add_footer()

    def slide_cta(self):
        """Slide 10: Call to Action"""
        self.add_page()
        self.light_gradient_background()

        self.centered_text(45, "Get Started in", size=38)
        self.centered_text(78, "30 Seconds", size=38, color=MID_BLUE)

        # pip install box
        box_width = 220
        box_x = (WIDTH - box_width) / 2
        self.set_fill_color(*MID_BLUE)
        self.rect(box_x, 125, box_width, 45, "F")

        self.set_xy(box_x, 138)
        self.set_font("Courier", "B", 24)
        self.set_text_color(*WHITE)
        self.cell(box_width, 14, "$ pip install diff-diff", align="C")

        # Links
        self.centered_text(200, "github.com/igerber/diff-diff", size=20, color=MID_BLUE)

        self.centered_text(235, "Full documentation & tutorials included", size=16, bold=False, color=GRAY)
        self.centered_text(255, "MIT Licensed  |  Open Source", size=16, bold=False, color=GRAY)

        self.add_connector_graphic("left")

        # Logo
        self.centered_text(290, "diff-diff", size=28, color=NAVY)
        self.centered_text(310, "Difference-in-Differences for Python", size=14, bold=False, color=GRAY)


def main():
    pdf = CarouselPDF()

    # Generate all slides
    pdf.slide_hook()
    pdf.slide_problem()
    pdf.slide_solution()
    pdf.slide_performance()
    pdf.slide_methods()
    pdf.slide_code()
    pdf.slide_validated()
    pdf.slide_features()
    pdf.slide_audience()
    pdf.slide_cta()

    # Save PDF
    output_path = "/Users/igerber/diff-diff/carousel/diff-diff-carousel.pdf"
    pdf.output(output_path)
    print(f"PDF saved to: {output_path}")


if __name__ == "__main__":
    main()
