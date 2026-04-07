"""Shared utilities for the visualization subpackage."""

import re


def _require_matplotlib():
    """Lazy import matplotlib with clear error message.

    Returns
    -------
    module
        The ``matplotlib.pyplot`` module.
    """
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. " "Install it with: pip install matplotlib"
        )


def _require_plotly():
    """Lazy import plotly with clear error message.

    Returns
    -------
    module
        The ``plotly.graph_objects`` module.
    """
    try:
        import plotly.graph_objects as go

        return go
    except ImportError:
        raise ImportError(
            "plotly is required for interactive plots. "
            "Install with: pip install diff-diff[plotly]"
        )


def _plotly_default_layout(fig, *, title=None, xlabel=None, ylabel=None, show_legend=True):
    """Apply standard plotly layout settings.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to configure.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    show_legend : bool, default=True
        Whether to show the legend.
    """
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        showlegend=show_legend,
        template="plotly_white",
        font=dict(size=12),
        margin=dict(l=60, r=30, t=50, b=50),
    )


# Complete CSS named color table (all 148 standard CSS colors).
# No matplotlib dependency required for any color used by plotly/CSS.
_CSS_COLORS = {
    "aliceblue": (240, 248, 255),
    "antiquewhite": (250, 235, 215),
    "aqua": (0, 255, 255),
    "aquamarine": (127, 255, 212),
    "azure": (240, 255, 255),
    "beige": (245, 245, 220),
    "bisque": (255, 228, 196),
    "black": (0, 0, 0),
    "blanchedalmond": (255, 235, 205),
    "blue": (0, 0, 255),
    "blueviolet": (138, 43, 226),
    "brown": (165, 42, 42),
    "burlywood": (222, 184, 135),
    "cadetblue": (95, 158, 160),
    "chartreuse": (127, 255, 0),
    "chocolate": (210, 105, 30),
    "coral": (255, 127, 80),
    "cornflowerblue": (100, 149, 237),
    "cornsilk": (255, 248, 220),
    "crimson": (220, 20, 60),
    "cyan": (0, 255, 255),
    "darkblue": (0, 0, 139),
    "darkcyan": (0, 139, 139),
    "darkgoldenrod": (184, 134, 11),
    "darkgray": (169, 169, 169),
    "darkgreen": (0, 100, 0),
    "darkgrey": (169, 169, 169),
    "darkkhaki": (189, 183, 107),
    "darkmagenta": (139, 0, 139),
    "darkolivegreen": (85, 107, 47),
    "darkorange": (255, 140, 0),
    "darkorchid": (153, 50, 204),
    "darkred": (139, 0, 0),
    "darksalmon": (233, 150, 122),
    "darkseagreen": (143, 188, 143),
    "darkslateblue": (72, 61, 139),
    "darkslategray": (47, 79, 79),
    "darkslategrey": (47, 79, 79),
    "darkturquoise": (0, 206, 209),
    "darkviolet": (148, 0, 211),
    "deeppink": (255, 20, 147),
    "deepskyblue": (0, 191, 255),
    "dimgray": (105, 105, 105),
    "dimgrey": (105, 105, 105),
    "dodgerblue": (30, 144, 255),
    "firebrick": (178, 34, 34),
    "floralwhite": (255, 250, 240),
    "forestgreen": (34, 139, 34),
    "fuchsia": (255, 0, 255),
    "gainsboro": (220, 220, 220),
    "ghostwhite": (248, 248, 255),
    "gold": (255, 215, 0),
    "goldenrod": (218, 165, 32),
    "gray": (128, 128, 128),
    "green": (0, 128, 0),
    "greenyellow": (173, 255, 47),
    "grey": (128, 128, 128),
    "honeydew": (240, 255, 240),
    "hotpink": (255, 105, 180),
    "indianred": (205, 92, 92),
    "indigo": (75, 0, 130),
    "ivory": (255, 255, 240),
    "khaki": (240, 230, 140),
    "lavender": (230, 230, 250),
    "lavenderblush": (255, 240, 245),
    "lawngreen": (124, 252, 0),
    "lemonchiffon": (255, 250, 205),
    "lightblue": (173, 216, 230),
    "lightcoral": (240, 128, 128),
    "lightcyan": (224, 255, 255),
    "lightgoldenrodyellow": (250, 250, 210),
    "lightgray": (211, 211, 211),
    "lightgreen": (144, 238, 144),
    "lightgrey": (211, 211, 211),
    "lightpink": (255, 182, 193),
    "lightsalmon": (255, 160, 122),
    "lightseagreen": (32, 178, 170),
    "lightskyblue": (135, 206, 250),
    "lightslategray": (119, 136, 153),
    "lightslategrey": (119, 136, 153),
    "lightsteelblue": (176, 196, 222),
    "lightyellow": (255, 255, 224),
    "lime": (0, 255, 0),
    "limegreen": (50, 205, 50),
    "linen": (250, 240, 230),
    "magenta": (255, 0, 255),
    "maroon": (128, 0, 0),
    "mediumaquamarine": (102, 205, 170),
    "mediumblue": (0, 0, 205),
    "mediumorchid": (186, 85, 211),
    "mediumpurple": (147, 112, 219),
    "mediumseagreen": (60, 179, 113),
    "mediumslateblue": (123, 104, 238),
    "mediumspringgreen": (0, 250, 154),
    "mediumturquoise": (72, 209, 204),
    "mediumvioletred": (199, 21, 133),
    "midnightblue": (25, 25, 112),
    "mintcream": (245, 255, 250),
    "mistyrose": (255, 228, 225),
    "moccasin": (255, 228, 181),
    "navajowhite": (255, 222, 173),
    "navy": (0, 0, 128),
    "oldlace": (253, 245, 230),
    "olive": (128, 128, 0),
    "olivedrab": (107, 142, 35),
    "orange": (255, 165, 0),
    "orangered": (255, 69, 0),
    "orchid": (218, 112, 214),
    "palegoldenrod": (238, 232, 170),
    "palegreen": (152, 251, 152),
    "paleturquoise": (175, 238, 238),
    "palevioletred": (219, 112, 147),
    "papayawhip": (255, 239, 213),
    "peachpuff": (255, 218, 185),
    "peru": (205, 133, 63),
    "pink": (255, 192, 203),
    "plum": (221, 160, 221),
    "powderblue": (176, 224, 230),
    "purple": (128, 0, 128),
    "rebeccapurple": (102, 51, 153),
    "red": (255, 0, 0),
    "rosybrown": (188, 143, 143),
    "royalblue": (65, 105, 225),
    "saddlebrown": (139, 69, 19),
    "salmon": (250, 128, 114),
    "sandybrown": (244, 164, 96),
    "seagreen": (46, 139, 87),
    "seashell": (255, 245, 238),
    "sienna": (160, 82, 45),
    "silver": (192, 192, 192),
    "skyblue": (135, 206, 235),
    "slateblue": (106, 90, 205),
    "slategray": (112, 128, 144),
    "slategrey": (112, 128, 144),
    "snow": (255, 250, 250),
    "springgreen": (0, 255, 127),
    "steelblue": (70, 130, 180),
    "tan": (210, 180, 140),
    "teal": (0, 128, 128),
    "thistle": (216, 191, 216),
    "tomato": (255, 99, 71),
    "turquoise": (64, 224, 208),
    "violet": (238, 130, 238),
    "wheat": (245, 222, 179),
    "white": (255, 255, 255),
    "whitesmoke": (245, 245, 245),
    "yellow": (255, 255, 0),
    "yellowgreen": (154, 205, 50),
}

_RGB_RE = re.compile(r"^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$")
_RGBA_RE = re.compile(r"^rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*\)$")


def _color_to_rgba(color, alpha=1.0):
    """Convert any color to an ``rgba(r, g, b, a)`` string for plotly.

    Accepts hex colors (``#rrggbb``, ``#rgb``), all 148 CSS named colors,
    and ``rgb(r,g,b)`` / ``rgba(r,g,b,a)`` strings. Does **not** require
    matplotlib.

    Parameters
    ----------
    color : str
        Color specification.
    alpha : float, default=1.0
        Opacity value between 0 and 1.

    Returns
    -------
    str
        An ``rgba(r, g, b, a)`` string.
    """
    if not isinstance(color, str):
        raise ValueError(f"Expected a color string, got {type(color).__name__}")

    # 1. Hex colors: #rrggbb or #rgb
    stripped = color.lstrip("#")
    if color.startswith("#") and all(c in "0123456789abcdefABCDEF" for c in stripped):
        if len(stripped) == 6:
            r = int(stripped[0:2], 16)
            g = int(stripped[2:4], 16)
            b = int(stripped[4:6], 16)
            return f"rgba({r}, {g}, {b}, {alpha})"
        if len(stripped) == 3:
            r = int(stripped[0] * 2, 16)
            g = int(stripped[1] * 2, 16)
            b = int(stripped[2] * 2, 16)
            return f"rgba({r}, {g}, {b}, {alpha})"

    # 2. Named CSS colors (complete table — no matplotlib needed)
    if color.lower() in _CSS_COLORS:
        r, g, b = _CSS_COLORS[color.lower()]
        return f"rgba({r}, {g}, {b}, {alpha})"

    # 3. rgb(r, g, b) — parse and apply alpha
    m = _RGB_RE.match(color.strip())
    if m:
        r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"rgba({r}, {g}, {b}, {alpha})"

    # 4. rgba(r, g, b, a) — parse and override alpha
    m = _RGBA_RE.match(color.strip())
    if m:
        r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"rgba({r}, {g}, {b}, {alpha})"

    raise ValueError(
        f"Cannot parse color '{color}'. Use hex (#rrggbb), a CSS color name, "
        "or rgb(r,g,b) / rgba(r,g,b,a) format."
    )


# Matplotlib marker code -> plotly symbol name mapping
_MPL_TO_PLOTLY_SYMBOL = {
    "o": "circle",
    "s": "square",
    "D": "diamond",
    "d": "diamond",
    "^": "triangle-up",
    "v": "triangle-down",
    "<": "triangle-left",
    ">": "triangle-right",
    "p": "pentagon",
    "h": "hexagon",
    "+": "cross",
    "x": "x",
    "*": "star",
    ".": "circle",
}


def _mpl_marker_to_plotly_symbol(marker):
    """Convert a matplotlib marker code to a plotly symbol name.

    Parameters
    ----------
    marker : str
        Matplotlib marker shorthand (e.g., ``"o"``, ``"s"``, ``"D"``).

    Returns
    -------
    str
        Plotly symbol name (e.g., ``"circle"``, ``"square"``, ``"diamond"``).
        Returns ``"circle"`` for unrecognized markers.
    """
    return _MPL_TO_PLOTLY_SYMBOL.get(marker, "circle")


# Default color constants
DEFAULT_BLUE = "#2563eb"
DEFAULT_RED = "#dc2626"
DEFAULT_GREEN = "#22c55e"
DEFAULT_GRAY = "#6b7280"
DEFAULT_DARK = "#1f2937"
DEFAULT_SHADE = "#f0f0f0"
