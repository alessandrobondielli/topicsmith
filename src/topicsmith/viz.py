"""Shared chart style, palette and primitives.

Every figure topicsmith draws is built from these, so the whole set reads as one
system.

**Themes.** The palette is not hard-coded: :data:`THEMES` holds several complete,
validated colour systems and :func:`apply_style` installs one. Switching the
whole pack is a single argument at the top of a notebook::

    viz.apply_style(theme="ink_ember")      # the default -- warm paper, ember
    viz.apply_style(theme="mediterranean")  # teal
    viz.apply_style(theme="ultraviolet")    # violet
    viz.apply_style(theme="classic_blue")   # the 2026 original
    viz.apply_style(theme="forest")         # botanical green
    viz.apply_style(theme="crimson")        # deep red
    viz.apply_style(theme="marigold")       # warm gold
    viz.apply_style(theme="slate")          # muted blue, near-monochrome (print)

``use_theme`` rebinds the module-level colour names, so every primitive below --
and everything in ``network`` and ``figures`` -- follows automatically. Colours
are read at call time, never captured at import.

Colour is assigned by the job it does:

* **sequential** (one hue, light to dark) for magnitude -- counts of papers per
  area, of papers per subtopic. This is the default and covers most charts here.
* **categorical** for the few places where the series *are* the subject: a
  two-way split, a handful of clusters. Capped at three slots, which is the
  number that validates all-pairs; a fourth series folds into "other".
* **emphasis** -- one hue plus grey -- wherever a single group is the point.

Every ramp here passes ``validate_palette.js``: monotone lightness, adjacent
step gaps of at least 0.06 L, a light end clearing 2:1 on its own surface, and
for the categorical slots all-pairs CVD separation. Direct labels are still
mandatory on bars, because they are what lets the lighter steps stay legible.

**Type is sized for a projector, not for a page.** :data:`SCALE` multiplies every
font size in one place; ``apply_style(scale=1.15)`` makes the whole pack bigger
without touching a single chart.

**Saved figures have a transparent background** regardless of theme -- they go
full-bleed onto slides whose colour is unknown here. The theme ``SURFACE`` still
paints the on-screen figure and, opaquely, the knockout halos and segment gaps
that keep labels legible where marks overlap. Dark ink plus white halos assumes
a light-ish slide; a deck on black wants a light ``INK``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

# --- themes ------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    """One complete colour system.

    ``sequential`` runs light to dark and is the magnitude ramp. ``categorical``
    is a fixed order -- assigned in order, never cycled -- whose first three
    slots are validated to separate under every colour-vision simulation with
    all pairs on screen at once.
    """

    name: str
    surface: str
    page: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    baseline: str
    deemphasis: str
    sequential: tuple[str, ...]
    categorical: tuple[str, ...]
    land: str
    land_edge: str
    # Neutral ramp, used where a second magnitude scale has to stay quiet
    # (map land shading, de-emphasised network components).
    neutral: tuple[str, ...] = ()

    @property
    def accent(self) -> str:
        return self.categorical[0]

    @property
    def accent_dark(self) -> str:
        return self.sequential[-3]


GRAPHITE = (
    "#e2ded4", "#cdc8bc", "#b4aea1", "#9a9386", "#82796c", "#6a6155",
    "#534b40", "#3e372e", "#2b251e", "#211c16", "#191410",
)

THEMES: dict[str, Theme] = {
    # Newspaper graphic: near-black type on warm paper, ember carrying the data.
    "ink_ember": Theme(
        name="ink_ember",
        surface="#f7f5f0",
        page="#f2efe8",
        ink="#141210",
        ink_secondary="#4a443d",
        ink_muted="#847c71",
        grid="#e3ded3",
        baseline="#c5bdaf",
        deemphasis="#c2bbae",
        sequential=(
            "#f7ddc9", "#f2ceb2", "#ecbd9a", "#e6aa7f", "#e2955f", "#d47a3d",
            "#c26026", "#a94f1f", "#8d4119", "#713414", "#55270f", "#47210c",
            "#3a1a0a",
        ),
        categorical=("#d4552b", "#00998a", "#7a52c0", "#4a443d"),
        land="#e8e3d8",
        land_edge="#d5cec0",
        neutral=GRAPHITE,
    ),
    # Deep teal with terracotta and violet accents, on warm paper.
    "mediterranean": Theme(
        name="mediterranean",
        surface="#FFFFFF",
        page="#FFFFFF",
        ink="#101614",
        ink_secondary="#41504c",
        ink_muted="#7d8a86",
        grid="#e2e5df",
        baseline="#FFFFFF",
        deemphasis="#c1c8c3",
        sequential=(
            "#d5efe9", "#b6e3da", "#93d5c9", "#5fbcae", "#33a696", "#009182",
            "#007c6e", "#01685c", "#05544a", "#093f38", "#0d2c28", "#0a231f",
            "#071a17",
        ),
        categorical=("#00998a", "#d4552b", "#7a52c0", "#41504c"),
        land="#e9e6de",
        land_edge="#d6d2c7",
        neutral=GRAPHITE,
    ),
    # Violet with gold and teal accents.
    "ultraviolet": Theme(
        name="ultraviolet",
        surface="#faf9fb",
        page="#f3f1f6",
        ink="#14121a",
        ink_secondary="#4b4757",
        ink_muted="#847f92",
        grid="#e5e2ec",
        baseline="#c8c3d3",
        deemphasis="#c6c2d0",
        sequential=(
            "#e3dbf8", "#d0c5f2", "#b8a6ea", "#a18ee0", "#8a72d4", "#7458c4",
            "#5f45ab", "#4b348c", "#3e2a75", "#38266c", "#2f2059", "#26194c",
            "#1d1339",
        ),
        categorical=("#5b3fa8", "#e0982a", "#0f9b8e", "#4b4757"),
        land="#eae7ef",
        land_edge="#d8d3e0",
        neutral=GRAPHITE,
    ),
    # Botanical green carrying the data, with a magenta and a blue accent, on a
    # faintly green-tinted paper.
    "forest": Theme(
        name="forest",
        surface="#f5f7f3",
        page="#eef2ea",
        ink="#0f130f",
        ink_secondary="#494f4a",
        ink_muted="#7c827d",
        grid="#dae0da",
        baseline="#b9c0ba",
        deemphasis="#b5bdb6",
        sequential=(
            "#d2f0d9", "#b4e7c0", "#92dca6", "#65c282", "#46b36c", "#2ca45b",
            "#0f944c", "#00813f", "#006f36", "#005829", "#003e1b", "#023115",
            "#022610",
        ),
        categorical=("#0f8a49", "#c8578a", "#1f8fd0", "#494f4a"),
        land="#e7ebe0",
        land_edge="#d4dbc9",
        neutral=GRAPHITE,
    ),
    # Deep crimson, with a teal and a blue accent, on warm paper.
    "crimson": Theme(
        name="crimson",
        surface="#f9f6f4",
        page="#f4efeb",
        ink="#16100f",
        ink_secondary="#534b49",
        ink_muted="#867e7d",
        grid="#e5dcda",
        baseline="#c5bbb9",
        deemphasis="#c2b8b6",
        sequential=(
            "#ffdedb", "#ffc9c4", "#ffb0a9", "#f28981", "#e6726b", "#d7615b",
            "#c5514c", "#af413e", "#9a3533", "#7b2927", "#561d1c", "#441716",
            "#351110",
        ),
        categorical=("#c0403f", "#0f9a7f", "#2f6fb8", "#534b49"),
        land="#ece5df",
        land_edge="#dcd2c8",
        neutral=GRAPHITE,
    ),
    # Warm marigold, with a berry and a teal accent, on cream paper.
    "marigold": Theme(
        name="marigold",
        surface="#faf8f1",
        page="#f4f0e4",
        ink="#14110d",
        ink_secondary="#514c46",
        ink_muted="#847f7a",
        grid="#e2ddd7",
        baseline="#c2bdb5",
        deemphasis="#bfbab2",
        sequential=(
            "#f7e5c4", "#f2d49d", "#ebc171", "#d6a132", "#c59000", "#b28100",
            "#9f7300", "#8a6300", "#785500", "#5f4300", "#442f00", "#372400",
            "#2b1b00",
        ),
        categorical=("#a87b12", "#a6395f", "#0b93a0", "#514c46"),
        land="#ece6d5",
        land_edge="#ddd4bd",
        neutral=GRAPHITE,
    ),
    # Muted slate blue, near-monochrome -- for a formal or printed deck. Berry and
    # amber accents for the few categorical charts.
    "slate": Theme(
        name="slate",
        surface="#f6f8f9",
        page="#eef2f4",
        ink="#0e1216",
        ink_secondary="#484e54",
        ink_muted="#7c8186",
        grid="#d9dfe5",
        baseline="#b8bec5",
        deemphasis="#b5bbc2",
        sequential=(
            "#d8eaf9", "#bfdcf5", "#a5cdf0", "#80b1db", "#69a0ce", "#5891c0",
            "#4981af", "#3a709b", "#2f6188", "#244c6d", "#1a354c", "#142a3c",
            "#0e202e",
        ),
        categorical=("#2f74b4", "#a6395f", "#b07414", "#484e54"),
        land="#e6eaed",
        land_edge="#d2d9de",
        neutral=GRAPHITE,
    ),
    # The palette the pack shipped with, kept so nothing is lost.
    "classic_blue": Theme(
        name="classic_blue",
        surface="#fcfcfb",
        page="#f9f9f7",
        ink="#0b0b0b",
        ink_secondary="#52514e",
        ink_muted="#898781",
        grid="#e1e0d9",
        baseline="#c3c2b7",
        deemphasis="#c3c2b7",
        sequential=(
            "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b",
        ),
        categorical=("#2a78d6", "#eb6834", "#1baf7a", "#52514e"),
        land="#eeede7",
        land_edge="#dcdbd3",
        neutral=GRAPHITE,
    ),
}

DEFAULT_THEME = "ink_ember"

# Reserved, never reused as a series colour.
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

# --- live palette ------------------------------------------------------------
# These names are what every chart reads. `use_theme` rebinds them; because the
# primitives look them up at call time, one call re-colours the whole pack.

THEME: Theme
SURFACE: str
PAGE: str
INK: str
INK_SECONDARY: str
INK_MUTED: str
GRID: str
BASELINE: str
DEEMPHASIS: str
SEQUENTIAL: list[str]
CATEGORICAL: list[str]
CATEGORICAL_ALL_PAIRS: list[str]
ORDINAL: list[str]
NEUTRAL: list[str]
ACCENT: str
ACCENT_DARK: str
SEQ_CMAP: LinearSegmentedColormap
NEUTRAL_CMAP: LinearSegmentedColormap
LAND: str
LAND_EDGE: str

# Type scale. Every size below is a multiple of SCALE, so one number moves the
# whole pack up for a big room or down for a printed page.
SCALE = 1.0
_BASE = {
    "base": 13.0,      # tick labels, body
    "title": 20.0,     # chart title
    "subtitle": 14.0,  # the line under it
    "label": 12.5,     # direct value labels on marks
    "legend": 12.5,
    "note": 10.5,      # methodology / source foot
    "tile_value": 40.0,
    "tile_label": 13.0,
}


def size(role: str) -> float:
    """A font size in points for *role*, after the global :data:`SCALE`."""
    return _BASE[role] * SCALE


def use_theme(theme: str | Theme = DEFAULT_THEME) -> Theme:
    """Install *theme* as the live palette and return it.

    Rebinding module globals rather than threading a palette object through every
    signature is what keeps ``network`` and ``figures`` in step for free: they
    all read ``viz.SURFACE``, ``viz.ACCENT`` and friends at call time.
    """
    global THEME, SURFACE, PAGE, INK, INK_SECONDARY, INK_MUTED, GRID, BASELINE
    global DEEMPHASIS, SEQUENTIAL, CATEGORICAL, CATEGORICAL_ALL_PAIRS, ORDINAL
    global NEUTRAL, ACCENT, ACCENT_DARK, SEQ_CMAP, NEUTRAL_CMAP, LAND, LAND_EDGE
    global BLUE, BLUE_DARK

    if isinstance(theme, str):
        if theme not in THEMES:
            raise KeyError(f"unknown theme {theme!r}; have {', '.join(THEMES)}")
        theme = THEMES[theme]

    THEME = theme
    SURFACE, PAGE = theme.surface, theme.page
    INK, INK_SECONDARY, INK_MUTED = theme.ink, theme.ink_secondary, theme.ink_muted
    GRID, BASELINE, DEEMPHASIS = theme.grid, theme.baseline, theme.deemphasis
    SEQUENTIAL = list(theme.sequential)
    CATEGORICAL = list(theme.categorical)
    # Forms where every pair is on screen at once (scatter, network, map) cap at
    # the three slots that validate all-pairs.
    CATEGORICAL_ALL_PAIRS = CATEGORICAL[:3]
    # Ordinal ramps must stay above 2:1 on the surface: start at step 4.
    ORDINAL = SEQUENTIAL[4:]
    NEUTRAL = list(theme.neutral or GRAPHITE)
    ACCENT, ACCENT_DARK = theme.accent, theme.accent_dark
    SEQ_CMAP = LinearSegmentedColormap.from_list("topicsmith_seq", SEQUENTIAL)
    NEUTRAL_CMAP = LinearSegmentedColormap.from_list("topicsmith_neutral", NEUTRAL)
    LAND, LAND_EDGE = theme.land, theme.land_edge
    # Back-compat aliases from when the pack was blue-only.
    BLUE, BLUE_DARK = ACCENT, ACCENT_DARK
    return theme


use_theme(DEFAULT_THEME)


def apply_style(theme: str | Theme | None = None, *, scale: float | None = None) -> Theme:
    """Install the house matplotlib style. Call once per notebook.

    *theme* names one of :data:`THEMES`; *scale* multiplies every font size (use
    ~1.15 for a large room, ~0.85 for a printed page).

    Saved figures are written with a **transparent** background (``figure`` and
    ``axes`` patches both), because the pack is dropped full-bleed onto slides
    whose background colour is not known here. The themed ``SURFACE`` colour is
    still used on screen and, opaquely, for the knockout halos and segment gaps
    that keep labels legible where marks overlap.
    """
    global SCALE

    active = use_theme(theme) if theme is not None else THEME
    if scale is not None:
        SCALE = float(scale)

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.transparent": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.22,
        "savefig.dpi": 200,
        "figure.dpi": 110,
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": size("base"),
        "text.color": INK,
        "axes.labelsize": size("base"),
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.9,
        "axes.titlesize": size("title"),
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": size("base"),
        "ytick.labelsize": size("base"),
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": size("legend"),
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
    })
    return active


def sequential_colors(n: int, *, dark_first: bool = True) -> list[str]:
    """*n* evenly spaced steps of the magnitude ramp, dark to light."""
    if n <= 0:
        return []
    idx = np.linspace(0.88, 0.34, n) if dark_first else np.linspace(0.34, 0.88, n)
    return [mpl.colors.to_hex(SEQ_CMAP(v)) for v in idx]


def shade_by_value(values: Sequence[float], *, floor: float = 0.34, ceil: float = 0.92) -> list[str]:
    """Map each value onto the magnitude ramp by size, not by position.

    Colour has to follow the quantity: shading a sorted bar chart by rank gives
    two areas with the same paper count visibly different shades, which reads as
    a difference that is not there.
    """
    values = np.asarray(values, dtype=float)
    if not values.size:
        return []
    top = values.max()
    scaled = np.full(values.shape, ceil) if top <= 0 else floor + (ceil - floor) * (values / top)
    return [mpl.colors.to_hex(SEQ_CMAP(v)) for v in scaled]


def emphasis_colors(labels: Sequence[str], highlight: Iterable[str]) -> list[str]:
    """One hue for the highlighted labels, de-emphasis grey for the rest."""
    hot = set(highlight)
    return [ACCENT if label in hot else DEEMPHASIS for label in labels]


def fade(color: str, alpha: float) -> tuple[float, float, float, float]:
    """RGBA of *color* at *alpha*, for network edges and map halos."""
    return to_rgba(color, alpha)


def mix(color: str, other: str, t: float) -> str:
    """*color* blended *t* of the way towards *other*. For quiet tints."""
    a = np.array(to_rgba(color)[:3])
    b = np.array(to_rgba(other)[:3])
    return mpl.colors.to_hex(tuple(a + (b - a) * float(t)))


# --- chart chrome ------------------------------------------------------------

def figure_title(fig, title: str, subtitle: str | None = None, *, compact: bool = True) -> None:
    """Flush-left title above the whole figure, clear of the axes.

    Anchoring to the figure rather than the axes is the fix for a horizontal bar
    chart: an axes-anchored title starts where the *plot area* starts, which on
    those charts is well to the right of the institution names, so the heading
    floated in the middle of the image. ``savefig.bbox = "tight"`` grows the
    canvas to include text placed above ``y = 1``, so nothing has to be reserved.

    ``compact`` first pulls the default subplot margins in. Those margins reserve
    about 12% of the canvas above the axes and 11% below for a title and an x
    label that a chart built this way does not have, and on a tall figure that
    is two inches of blank paper between the heading and the first bar.
    """
    from matplotlib.transforms import ScaledTranslation

    if compact:
        try:
            fig.tight_layout(pad=0.35)
        except Exception:  # noqa: BLE001 - legends outside the axes can defeat it
            pass

    def at(dy_inches: float, text: str, **kwargs):
        transform = fig.transFigure + ScaledTranslation(0, dy_inches, fig.dpi_scale_trans)
        return fig.text(0, 1, text, transform=transform, ha="left", va="bottom", **kwargs)

    offset = 0.09
    if subtitle:
        at(offset, subtitle, fontsize=size("subtitle"), color=INK_SECONDARY)
        offset += size("subtitle") / 72 * 1.75
    at(offset, title, fontsize=size("title"), fontweight="bold", color=INK)


def titles(ax, title: str, subtitle: str | None = None, *, legend: bool = False) -> None:
    """Panel title on *ax*, with an optional quieter subtitle beneath it.

    For a figure holding a single chart prefer :func:`figure_title`, which is not
    pushed right by the y tick labels. This one is for the panels of a
    multi-panel figure, where a per-axes anchor is what keeps each title over its
    own panel.

    Pass ``legend=True`` when the axes carries a legend above the plot area, so
    the title is lifted clear of it instead of overlapping.
    """
    pad = 14
    if legend:
        pad += size("legend") * 2.1
    if subtitle:
        ax.set_title(title, pad=pad + size("subtitle") * 1.3, fontsize=size("title"))
        ax.annotate(
            subtitle, xy=(0, 1), xytext=(0, pad - 4), xycoords="axes fraction",
            textcoords="offset points", ha="left", va="bottom",
            fontsize=size("subtitle"), color=INK_SECONDARY,
        )
    else:
        ax.set_title(title, pad=pad, fontsize=size("title"))


def note(fig, text: str, *, pad: float = 0.10) -> None:
    """A methodology or source line at the foot of the figure.

    ``pad`` is inches below the axes; raise it when the figure already carries a
    legend under the plot area.
    """
    from matplotlib.transforms import ScaledTranslation

    transform = fig.transFigure + ScaledTranslation(0, -pad, fig.dpi_scale_trans)
    fig.text(0, 0, text, transform=transform, ha="left", va="top",
             fontsize=size("note"), color=INK_MUTED, wrap=True)


def save(fig, name: str, cfg=None, *, subdir: str | None = None) -> Path:
    """Write a figure to ``figures/`` as PNG and SVG, returning the PNG path."""
    from .config import load_config

    cfg = cfg or load_config()
    directory = cfg.paths.figures / subdir if subdir else cfg.paths.figures
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{name}.png"
    fig.savefig(png)
    fig.savefig(directory / f"{name}.svg")
    return png


# --- forms -------------------------------------------------------------------

def hbar(
    ax,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    colors: Sequence[str] | str | None = None,
    fmt: str = "{:,.0f}",
    label_offset: float = 0.01,
    bar_height: float = 0.7,
    label_size: float | None = None,
) -> None:
    """Horizontal magnitude bars, largest first, every bar directly labelled.

    Horizontal because institution and country names are long, and reading them
    left-to-right beats rotating them. Direct labels are not decoration: they are
    the relief that lets the lighter ramp steps stay legible, and they remove the
    need for x gridlines entirely.
    """
    labels = [str(v) for v in labels]
    values = np.asarray(values, dtype=float)
    y = np.arange(len(labels))
    if colors is None:
        colors = shade_by_value(values)
    bars = ax.barh(y, values, height=bar_height, color=colors, zorder=3)
    for bar in bars:
        bar.set_capstyle("round")

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)

    span = float(values.max()) if values.size else 1.0
    for yi, value in zip(y, values):
        ax.annotate(
            fmt.format(value), xy=(value, yi), xytext=(6, 0),
            textcoords="offset points", va="center", ha="left",
            fontsize=label_size or size("label"), color=INK_SECONDARY,
            fontweight="bold",
        )
    ax.set_xlim(0, span * (1.12 + label_offset))
    ax.margins(y=0.5 / max(len(labels), 1))


def column(
    ax,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    colors: Sequence[str] | str | None = None,
    fmt: str = "{:,.0f}",
    xlabel: str | None = None,
    label_size: float | None = None,
) -> None:
    """Vertical columns for an ordered scale (a distribution), directly labelled."""
    values = np.asarray(values, dtype=float)
    x = np.arange(len(labels))
    if colors is None:
        colors = shade_by_value(values)
    ax.bar(x, values, width=0.7, color=colors, zorder=3)
    ax.set_xticks(x, [str(v) for v in labels])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    for xi, value in zip(x, values):
        if value:
            ax.annotate(
                fmt.format(value), xy=(xi, value), xytext=(0, 5),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=label_size or size("label"), color=INK_SECONDARY,
                fontweight="bold",
            )
    ax.set_ylim(0, float(values.max()) * 1.18 if values.size else 1)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=10)


def stacked_share(
    ax,
    rows: Sequence[str],
    series: dict[str, Sequence[float]],
    *,
    colors: Sequence[str] | None = None,
    min_label_pct: float = 8.0,
    reference: float | None = None,
    bar_height: float = 0.66,
) -> None:
    """Part-to-whole across several groupings, as 100% horizontal stacked bars.

    A 2px surface-coloured gap separates adjacent segments so the boundary reads
    without a border, and segments wide enough to hold text are labelled in place.
    Pass ``reference=50`` to draw the parity line a share chart is read against --
    without it the eye has to estimate where half-way falls on every row.
    """
    keys = list(series)
    colors = list(colors or CATEGORICAL[: len(keys)])
    # Coerce first: callers pass pandas Series as often as lists, and a Series
    # indexed by category name cannot be subscripted by position.
    values = {k: np.asarray(v, dtype=float) for k, v in series.items()}
    totals = np.sum([values[k] for k in keys], axis=0).astype(float)
    totals[totals == 0] = 1.0

    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for key, color in zip(keys, colors):
        widths = values[key] / totals * 100
        ax.barh(y, widths, left=left, height=bar_height, color=color, label=key, zorder=3,
                edgecolor=SURFACE, linewidth=2)
        for yi, (w, l) in enumerate(zip(widths, left)):
            if w >= min_label_pct:
                ax.annotate(
                    f"{w:.0f}%", xy=(l + w / 2, yi), ha="center", va="center",
                    fontsize=size("label"), color="white", fontweight="bold", zorder=5,
                )
        left += widths

    if reference is not None:
        ax.axvline(reference, color=INK, linewidth=1.1, linestyle=(0, (4, 3)),
                   alpha=0.55, zorder=6)

    ax.set_yticks(y, [str(r) for r in rows])
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_xlim(0, 100)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncols=len(keys),
              fontsize=size("legend"), handlelength=1.1, handleheight=1.1,
              columnspacing=1.6, borderpad=0)


def heatmap(
    ax,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    fmt: str = "{:.0f}",
    hide_zero: bool = True,
    normalise_rows: bool = False,
) -> None:
    """Sequential grid for magnitude across two dimensions.

    Cell text flips to white on the dark end of the ramp so every value stays
    readable -- the table view the contrast rule asks for is the chart itself.

    ``normalise_rows`` shades each row against its own maximum instead of the
    grid's. Use it when one column dwarfs the rest (an Italian conference
    cross-tabbed by country): shading globally there paints every other cell the
    same near-white and throws away the comparison the chart exists to make.
    """
    values = np.asarray(matrix, dtype=float)
    if normalise_rows:
        row_max = values.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        shade = values / row_max
    else:
        shade = values / (values.max() or 1)

    ax.imshow(shade, cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=32, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_xticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if hide_zero and not value:
                continue
            ax.text(j, i, fmt.format(value), ha="center", va="center",
                    fontsize=size("label"), fontweight="bold",
                    color="white" if shade[i, j] > 0.58 else INK_SECONDARY)


def stat_tiles(
    ax,
    tiles: Sequence[tuple[str, str]],
    *,
    columns: int = 4,
    value_size: float | None = None,
    label_size: float | None = None,
    accent_every: int = 0,
) -> None:
    """A KPI row of headline numbers -- the right form for a handful of scalars.

    Each tile is ``(value, label)``. A bar chart of unrelated totals would invite
    comparisons between quantities that share no scale; separate tiles do not.
    ``accent_every=2`` tints alternate tiles with the accent hue, which gives a
    slide-sized grid some rhythm without implying a grouping.
    """
    ax.set_axis_off()
    value_size = value_size or size("tile_value")
    label_size = label_size or size("tile_label")
    rows = int(np.ceil(len(tiles) / columns))
    # Value sits in the upper part of its cell, label hangs beneath it, and the
    # gap between them scales with the row height so multi-line labels in a tall
    # grid do not run into the next row's number.
    for i, (value, label) in enumerate(tiles):
        col, row = i % columns, i // columns
        x = (col + 0.5) / columns
        y = 1 - (row + 0.40) / rows
        color = ACCENT if accent_every and i % accent_every == 0 else INK
        ax.text(x, y, value, ha="center", va="center", fontsize=value_size,
                fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x, y - 0.21 / rows, label, ha="center", va="top", fontsize=label_size,
                color=INK_SECONDARY, transform=ax.transAxes, linespacing=1.35)


def legend_swatches(ax, entries: Sequence[tuple[str, str]], **kwargs) -> None:
    """A legend built from (label, colour) pairs, for charts drawn without labels."""
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=color, label=label) for label, color in entries]
    ax.legend(handles=handles, **{
        "loc": "upper right", "frameon": False, "fontsize": size("legend"),
        "handlelength": 1.1, "handleheight": 1.1, "labelspacing": 0.7, **kwargs,
    })
