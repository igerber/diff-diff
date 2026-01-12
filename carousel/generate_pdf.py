#!/usr/bin/env python3
"""Generate LinkedIn carousel PDF for diff-diff library."""

from fpdf import FPDF

# LinkedIn carousel dimensions (4:5 aspect ratio)
WIDTH = 270  # mm (1080px at 72dpi * 25.4 / 96 ≈ 270mm scaled for PDF)
HEIGHT = 337.5  # mm (1350px at 72dpi)

# Colors
NAVY = (30, 58, 95)  # #1e3a5f
DARK = (15, 23, 42)  # #0f172a
BLUE_ACCENT = (96, 165, 250)  # #60a5fa
BRIGHT_BLUE = (59, 130, 246)  # #3b82f6
WHITE = (255, 255, 255)
RED = (239, 68, 68)  # #ef4444
GREEN = (74, 222, 128)  # #4ade80
GRAY = (156, 163, 175)


class CarouselPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format=(WIDTH, HEIGHT))
        self.set_auto_page_break(False)

    def gradient_background(self):
        """Draw gradient background."""
        # Simulate gradient with rectangles
        steps = 50
        for i in range(steps):
            ratio = i / steps
            r = int(NAVY[0] * (1 - ratio) + DARK[0] * ratio)
            g = int(NAVY[1] * (1 - ratio) + DARK[1] * ratio)
            b = int(NAVY[2] * (1 - ratio) + DARK[2] * ratio)
            self.set_fill_color(r, g, b)
            y = i * HEIGHT / steps
            self.rect(0, y, WIDTH, HEIGHT / steps + 1, "F")

    def add_footer(self):
        """Add footer with logo."""
        self.set_xy(0, HEIGHT - 20)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*GRAY)
        self.cell(WIDTH, 10, "diff-diff", align="C")

    def centered_text(self, y, text, size=24, bold=True, color=WHITE):
        """Add centered text."""
        self.set_xy(0, y)
        self.set_font("Helvetica", "B" if bold else "", size)
        self.set_text_color(*color)
        self.cell(WIDTH, size * 0.5, text, align="C")

    def add_list_item(self, y, icon, text, icon_color, text_size=18):
        """Add a list item with icon."""
        margin = 40
        self.set_xy(margin, y)
        self.set_font("Helvetica", "B", text_size)
        self.set_text_color(*icon_color)
        self.cell(20, 10, icon, align="C")
        self.set_text_color(*WHITE)
        self.cell(WIDTH - margin * 2 - 20, 10, text)

    def slide_hook(self):
        """Slide 1: Hook"""
        self.add_page()
        self.gradient_background()

        # Logo
        self.set_xy(0, 40)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*WHITE)
        self.cell(WIDTH, 8, "diff", align="C", new_x="LEFT")
        self.set_xy(WIDTH/2 - 2, 40)
        self.set_text_color(*BLUE_ACCENT)
        self.text(WIDTH/2 + 10, 46, "-")
        self.set_text_color(*WHITE)
        self.text(WIDTH/2 + 14, 46, "diff")

        # Main headline
        self.centered_text(80, "Difference-in-", size=32)
        self.centered_text(100, "Differences.", size=32)
        self.set_text_color(*BLUE_ACCENT)
        self.centered_text(125, "Now in Python.", size=32, color=BLUE_ACCENT)

        # Subheadline
        self.centered_text(165, "Up to 2,000x faster than R", size=16, color=BLUE_ACCENT)

        # pip install box
        box_width = 160
        box_x = (WIDTH - box_width) / 2
        self.set_fill_color(0, 0, 0)
        self.set_draw_color(*BRIGHT_BLUE)
        self.set_line_width(0.8)
        self.rect(box_x, 200, box_width, 25, "DF")

        self.set_xy(box_x, 205)
        self.set_font("Courier", "B", 14)
        self.set_text_color(*BLUE_ACCENT)
        self.cell(20, 10, "$", align="R")
        self.set_text_color(*WHITE)
        self.cell(box_width - 20, 10, " pip install diff-diff")

        self.add_footer()

    def slide_problem(self):
        """Slide 2: The Problem"""
        self.add_page()
        self.gradient_background()

        self.centered_text(50, "DiD in Python", size=32)
        self.centered_text(75, "Was Broken", size=32, color=BLUE_ACCENT)

        items = [
            "Fragmented packages",
            "Missing modern methods",
            "Slow performance",
            "R required for serious work"
        ]

        y_start = 120
        for i, item in enumerate(items):
            self.add_list_item(y_start + i * 25, "X", item, RED)

        self.centered_text(260, "Researchers deserved better.", size=14, bold=False, color=GRAY)
        self.add_footer()

    def slide_solution(self):
        """Slide 3: The Solution"""
        self.add_page()
        self.gradient_background()

        self.centered_text(50, "Introducing", size=32)
        self.centered_text(75, "diff-diff", size=32, color=BLUE_ACCENT)

        items = [
            "Complete DiD toolkit",
            "sklearn-like API",
            "All modern methods",
            "Blazing fast"
        ]

        y_start = 120
        for i, item in enumerate(items):
            self.add_list_item(y_start + i * 25, "+", item, GREEN)

        self.centered_text(260, "One library. Everything you need.", size=14, bold=False, color=GRAY)
        self.add_footer()

    def slide_performance(self):
        """Slide 4: Performance"""
        self.add_page()
        self.gradient_background()

        # Headline
        self.set_xy(0, 40)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*WHITE)
        self.cell(WIDTH, 12, "R: ", align="C", new_x="LEFT")

        self.set_xy(0, 40)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*RED)
        self.cell(WIDTH + 30, 12, "24 minutes", align="C")

        self.set_xy(0, 65)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*WHITE)
        self.cell(WIDTH - 45, 12, "diff-diff: ", align="C")
        self.set_text_color(*BLUE_ACCENT)
        self.cell(60, 12, "2.6 seconds", align="L")

        # Bar chart
        bar_data = [
            ("Synthetic DiD", 100, "2,234x faster"),
            ("Basic DiD", 40, "18x faster"),
            ("Callaway-SA", 35, "14x faster"),
        ]

        margin = 30
        bar_height = 20
        y_start = 120

        for i, (label, pct, speedup) in enumerate(bar_data):
            y = y_start + i * 40

            # Label
            self.set_xy(margin, y)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*WHITE)
            self.cell(55, bar_height, label, align="R")

            # Bar background
            bar_x = margin + 60
            bar_width = WIDTH - margin * 2 - 60
            self.set_fill_color(50, 50, 60)
            self.rect(bar_x, y, bar_width, bar_height, "F")

            # Bar fill
            fill_width = bar_width * pct / 100
            self.set_fill_color(*BRIGHT_BLUE)
            self.rect(bar_x, y, fill_width, bar_height, "F")

            # Speedup text
            self.set_xy(bar_x, y)
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(*WHITE)
            self.cell(fill_width - 5, bar_height, speedup, align="R")

        self.centered_text(270, "Benchmarked at 10K unit scale against R packages", size=12, bold=False, color=GRAY)
        self.add_footer()

    def slide_methods(self):
        """Slide 5: Methods"""
        self.add_page()
        self.gradient_background()

        self.centered_text(40, "Every Method", size=32)
        self.centered_text(65, "You Need", size=32, color=BLUE_ACCENT)

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
        box_height = 45

        for i, (title, desc) in enumerate(methods):
            col = i % 2
            row = i // 2
            x = margin + col * (box_width + margin)
            y = 100 + row * (box_height + 15)

            # Box
            self.set_fill_color(40, 50, 70)
            self.set_draw_color(80, 120, 180)
            self.set_line_width(0.3)
            self.rect(x, y, box_width, box_height, "DF")

            # Title
            self.set_xy(x + 5, y + 8)
            self.set_font("Helvetica", "B", 12)
            self.set_text_color(*BLUE_ACCENT)
            self.cell(box_width - 10, 8, title, align="C")

            # Description
            self.set_xy(x + 5, y + 22)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*GRAY)
            self.cell(box_width - 10, 8, desc, align="C")

        self.add_footer()

    def slide_code(self):
        """Slide 6: Code Example"""
        self.add_page()
        self.gradient_background()

        self.centered_text(35, "Clean, Pythonic API", size=28)

        # Code block
        margin = 25
        code_y = 75
        self.set_fill_color(20, 25, 35)
        self.set_draw_color(60, 100, 150)
        self.set_line_width(0.5)
        self.rect(margin, code_y, WIDTH - margin * 2, 165, "DF")

        code_lines = [
            ("from", "keyword"),
            (" diff_diff ", "normal"),
            ("import", "keyword"),
            (" CallawaySantAnna", "normal"),
            ("", "newline"),
            ("", "newline"),
            ("# Staggered DiD in 5 lines", "comment"),
            ("", "newline"),
            ("cs = ", "normal"),
            ("CallawaySantAnna", "function"),
            ("()", "normal"),
            ("", "newline"),
            ("results = cs.", "normal"),
            ("fit", "function"),
            ("(", "normal"),
            ("", "newline"),
            ("    data,", "normal"),
            ("", "newline"),
            ("    outcome=", "normal"),
            ("'sales'", "string"),
            (",", "normal"),
            ("", "newline"),
            ("    unit=", "normal"),
            ("'firm_id'", "string"),
            (",", "normal"),
            ("", "newline"),
            ("    time=", "normal"),
            ("'year'", "string"),
            (",", "normal"),
            ("", "newline"),
            ("    first_treat=", "normal"),
            ("'first_treat'", "string"),
            ("", "newline"),
            (")", "normal"),
            ("", "newline"),
            ("", "newline"),
            ("results.", "normal"),
            ("print_summary", "function"),
            ("()", "normal"),
        ]

        # Simplified code rendering
        self.set_font("Courier", "", 10)
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

        self.set_xy(margin + 10, code_y + 10)
        self.set_text_color(*WHITE)
        for i, line in enumerate(code_text.split('\n')):
            self.set_xy(margin + 10, code_y + 10 + i * 10)
            self.cell(0, 8, line)

        self.centered_text(260, "sklearn-like fit() + statsmodels-style output", size=12, bold=False, color=GRAY)
        self.add_footer()

    def slide_validated(self):
        """Slide 7: Validated"""
        self.add_page()
        self.gradient_background()

        self.centered_text(35, "Validated. Trusted.", size=28)
        self.centered_text(57, "Production-Ready.", size=28, color=BLUE_ACCENT)

        # Table
        margin = 25
        table_y = 100
        row_height = 30

        data = [
            ("Point estimates vs R", "Identical (10+ decimals)", True),
            ("Standard errors", "Within 1-3%", True),
            ("R packages tested", "did, synthdid, fixest", False),
            ("Real-world validation", "MPDTA dataset", True),
        ]

        # Header
        self.set_fill_color(50, 70, 100)
        self.rect(margin, table_y, WIDTH - margin * 2, row_height, "F")

        col1_width = (WIDTH - margin * 2) * 0.5
        col2_width = (WIDTH - margin * 2) * 0.5

        self.set_xy(margin, table_y + 8)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*WHITE)
        self.cell(col1_width, 12, "Comparison", align="C")
        self.cell(col2_width, 12, "Result", align="C")

        # Rows
        for i, (label, value, is_check) in enumerate(data):
            y = table_y + row_height * (i + 1)

            if i % 2 == 0:
                self.set_fill_color(35, 45, 60)
            else:
                self.set_fill_color(30, 40, 55)
            self.rect(margin, y, WIDTH - margin * 2, row_height, "F")

            self.set_xy(margin + 10, y + 8)
            self.set_font("Helvetica", "", 11)
            self.set_text_color(*WHITE)
            self.cell(col1_width - 10, 12, label)

            self.set_xy(margin + col1_width, y + 8)
            if is_check:
                self.set_text_color(*GREEN)
            else:
                self.set_text_color(*WHITE)
            self.cell(col2_width - 10, 12, value, align="C")

        self.centered_text(280, "Academic-grade accuracy. No compromises.", size=12, bold=False, color=GRAY)
        self.add_footer()

    def slide_features(self):
        """Slide 8: Features"""
        self.add_page()
        self.gradient_background()

        self.centered_text(35, "Everything", size=32)
        self.centered_text(60, "Included", size=32, color=BLUE_ACCENT)

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

        margin = 30
        col_width = (WIDTH - margin * 3) / 2

        for i, feature in enumerate(features):
            col = i % 2
            row = i // 2
            x = margin + col * (col_width + margin)
            y = 105 + row * 35

            # Icon
            self.set_xy(x, y)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*BLUE_ACCENT)
            self.cell(15, 12, "*")

            # Text
            self.set_text_color(*WHITE)
            self.set_font("Helvetica", "", 13)
            self.cell(col_width - 15, 12, feature)

            # Underline
            self.set_draw_color(60, 100, 150)
            self.set_line_width(0.2)
            self.line(x, y + 18, x + col_width, y + 18)

        self.centered_text(280, "Publication-ready visualizations included", size=12, bold=False, color=GRAY)
        self.add_footer()

    def slide_audience(self):
        """Slide 9: Audience"""
        self.add_page()
        self.gradient_background()

        self.centered_text(35, "Built for", size=32)
        self.centered_text(60, "Researchers", size=32, color=BLUE_ACCENT)

        audiences = [
            ("|||", "Applied Economists"),
            ("^^^", "Policy Researchers"),
            ("oOo", "Data Scientists"),
            (">_", "Python-First Teams"),
        ]

        margin = 30
        box_width = (WIDTH - margin * 3) / 2
        box_height = 70

        for i, (icon, title) in enumerate(audiences):
            col = i % 2
            row = i // 2
            x = margin + col * (box_width + margin)
            y = 100 + row * (box_height + 20)

            # Box
            self.set_fill_color(40, 50, 70)
            self.set_draw_color(80, 120, 180)
            self.set_line_width(0.3)
            self.rect(x, y, box_width, box_height, "DF")

            # Icon
            self.set_xy(x, y + 10)
            self.set_font("Courier", "B", 24)
            self.set_text_color(*WHITE)
            self.cell(box_width, 15, icon, align="C")

            # Title
            self.set_xy(x, y + 40)
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(*WHITE)
            self.cell(box_width, 12, title, align="C")

        self.centered_text(290, "Finally escape R for your causal inference work", size=14, bold=False, color=GRAY)
        self.add_footer()

    def slide_cta(self):
        """Slide 10: Call to Action"""
        self.add_page()
        self.gradient_background()

        self.centered_text(50, "Get Started in", size=32)
        self.centered_text(75, "30 Seconds", size=32, color=BLUE_ACCENT)

        # pip install box
        box_width = 180
        box_x = (WIDTH - box_width) / 2
        self.set_fill_color(0, 0, 0)
        self.set_draw_color(*BRIGHT_BLUE)
        self.set_line_width(1)
        self.rect(box_x, 115, box_width, 35, "DF")

        self.set_xy(box_x, 123)
        self.set_font("Courier", "B", 18)
        self.set_text_color(*BLUE_ACCENT)
        self.cell(25, 12, "$", align="R")
        self.set_text_color(*WHITE)
        self.cell(box_width - 25, 12, " pip install diff-diff")

        # Links
        self.centered_text(180, "github.com/igerber/diff-diff", size=14, color=BLUE_ACCENT)

        self.centered_text(210, "Full documentation & tutorials included", size=11, bold=False, color=GRAY)
        self.centered_text(225, "MIT Licensed | Open Source", size=11, bold=False, color=GRAY)

        # Logo
        self.centered_text(270, "diff-diff", size=24)
        self.centered_text(290, "Difference-in-Differences for Python", size=11, bold=False, color=GRAY)


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
