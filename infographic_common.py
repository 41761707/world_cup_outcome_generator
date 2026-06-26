"""Shared constants, data paths and drawing helpers for infographic scripts."""

from __future__ import annotations

import contextlib
import io
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw

from main import run_monte_carlo
from wc_logic import load_schedule_presets

BASE_DIR = Path(__file__).resolve().parent
COUNTRIES_FILE = BASE_DIR / "countries.txt"
GROUPS_FILE = BASE_DIR / "groups.txt"
SCHEDULE_GROUPS_FILE = BASE_DIR / "schedule_groups.txt"
SCHEDULE_KNOCKOUT_FILE = BASE_DIR / "schedule_knockout.txt"
OUTPUT_DIR = BASE_DIR / "infographic"
GROUPS_OUTPUT_DIR = OUTPUT_DIR / "groups"
FLAG_CACHE_DIR = OUTPUT_DIR / ".flag_cache"
FLAG_CANVAS_WIDTH = 80
FLAG_CANVAS_HEIGHT = 53
FLAG_BORDER_COLOR = (173, 181, 189, 255)
FLAG_BORDER_WIDTH = 1
FLAG_ZOOM_HERO = 1.0
FLAG_ZOOM_PANEL = 0.72
FLAG_ZOOM_COMPACT = 0.55

N_SIMULATIONS = 100000
LAMBDA_BASE = 1.3
K_FACTOR = 0.3

PAGE_BG = "#eef2f7"
HEADER_GRADIENT = ("#0b1d3a", "#1e3a6e", "#2d5aa0")
CARD_FACE = "#ffffff"
CARD_EDGE = "#dee2e6"

COUNTRY_FLAG_CODES = {
    "Czechy": "cz",
    "Meksyk": "mx",
    "Republika Południowej Afryki": "za",
    "Korea Południowa": "kr",
    "Szwajcaria": "ch",
    "Bośnia i Hercegowina": "ba",
    "Kanada": "ca",
    "Katar": "qa",
    "Szkocja": "gb-sct",
    "Brazylia": "br",
    "Haiti": "ht",
    "Maroko": "ma",
    "Turcja": "tr",
    "Paragwaj": "py",
    "USA": "us",
    "Australia": "au",
    "Niemcy": "de",
    "Ekwador": "ec",
    "Wybrzeże Kości Słoniowej": "ci",
    "Curacao": "cw",
    "Szwecja": "se",
    "Holandia": "nl",
    "Tunezja": "tn",
    "Japonia": "jp",
    "Belgia": "be",
    "Egipt": "eg",
    "Iran": "ir",
    "Nowa Zelandia": "nz",
    "Hiszpania": "es",
    "Urugwaj": "uy",
    "Republika Zielonego Przylądka": "cv",
    "Arabia Saudyjska": "sa",
    "Francja": "fr",
    "Norwegia": "no",
    "Senegal": "sn",
    "Irak": "iq",
    "Austria": "at",
    "Argentyna": "ar",
    "Algieria": "dz",
    "Jordania": "jo",
    "Portugalia": "pt",
    "Kolumbia": "co",
    "DR Konga": "cd",
    "Uzbekistan": "uz",
    "Chorwacja": "hr",
    "Anglia": "gb-eng",
    "Ghana": "gh",
    "Panama": "pa",
}


def load_groups_data() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return group members and team-to-group mapping."""
    members: dict[str, list[str]] = {}
    team_group: dict[str, str] = {}
    with open(GROUPS_FILE, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            group_name, rest = line.split(":", 1)
            names = [name.strip() for name in rest.split(",")]
            members[group_name.strip()] = names
            for name in names:
                team_group[name] = group_name.strip()
    return members, team_group


def team_name_label(
    name: str,
    *,
    base_fontsize: float = 14,
) -> tuple[str, float, float]:
    """Return wrapped team label, font size and line spacing."""
    if len(name) <= 20:
        return name, base_fontsize, 1.0
    if len(name) <= 28:
        return name, base_fontsize * 0.86, 1.0
    words = name.split()
    if len(words) >= 2:
        mid = (len(words) + 1) // 2
        line1 = " ".join(words[:mid])
        line2 = " ".join(words[mid:])
        return f"{line1}\n{line2}", base_fontsize * 0.78, 1.35
    return name, base_fontsize * 0.78, 1.0


def load_schedule_presets_from_file() -> dict[tuple[str, str], tuple[int, int]] | None:
    """Return fixed group scores embedded in schedule_groups.txt."""
    presets = load_schedule_presets(str(SCHEDULE_GROUPS_FILE))
    return presets or None


def format_preset_notice(
    fixed_group_results: dict[tuple[str, str], tuple[int, int]] | None,
) -> str:
    """Return a short log line describing applied schedule presets."""
    if not fixed_group_results:
        return "Bez znanych wynikow z schedule_groups.txt."
    return (
        f"Uwzgledniam {len(fixed_group_results)} znanych wynikow "
        "z schedule_groups.txt."
    )


def resolve_fixed_group_results(
    fixed_group_results: dict[tuple[str, str], tuple[int, int]] | None,
    *,
    use_schedule_presets: bool,
) -> dict[tuple[str, str], tuple[int, int]] | None:
    """Merge explicit overrides with schedule file presets when enabled."""
    if fixed_group_results is not None:
        return fixed_group_results
    if use_schedule_presets:
        return load_schedule_presets_from_file()
    return None


def run_simulation(
    n: int = N_SIMULATIONS,
    *,
    lambda_base: float = LAMBDA_BASE,
    k: float = K_FACTOR,
    fixed_group_results: dict[tuple[str, str], tuple[int, int]] | None = None,
    fixed_knockout_results: dict[str, tuple[int, int, str | None]] | None = None,
    use_schedule_presets: bool = True,
    quiet: bool = True,
) -> dict[str, Any]:
    """Run Monte Carlo with project default data paths."""
    effective_fixed = resolve_fixed_group_results(
        fixed_group_results,
        use_schedule_presets=use_schedule_presets,
    )

    def _run() -> dict[str, Any]:
        return run_monte_carlo(
            str(COUNTRIES_FILE),
            str(GROUPS_FILE),
            str(SCHEDULE_GROUPS_FILE),
            str(SCHEDULE_KNOCKOUT_FILE),
            n=n,
            lambda_base=lambda_base,
            k=k,
            fixed_group_results=effective_fixed,
            fixed_knockout_results=fixed_knockout_results,
        )

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            return _run()
    return _run()


def collect_match_score_counts(
    stats: dict[str, Any],
    team_a: str,
    team_b: str,
) -> dict[tuple[int, int], int]:
    """Merge score counts from both home/away orientations."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    score_counts = stats.get("group_match_score_counts", {})
    for (s1, s2), count in score_counts.get((team_a, team_b), {}).items():
        counts[(s1, s2)] += count
    for (s1, s2), count in score_counts.get((team_b, team_a), {}).items():
        counts[(s2, s1)] += count
    return dict(counts)


def expected_match_points(
    stats: dict[str, Any],
    team: str,
    opponent: str,
) -> float:
    """Return expected group-stage points from one fixture."""
    n = stats["n_simulations"]
    counts = collect_match_score_counts(stats, team, opponent)
    if not counts:
        return 0.0
    total = 0.0
    for (s1, s2), count in counts.items():
        if s1 > s2:
            points = 3
        elif s1 == s2:
            points = 1
        else:
            points = 0
        total += points * count / n
    return total


def _normalize_flag_image(image: Image.Image) -> Image.Image:
    """Crop/scale source art to a fixed canvas with a visible edge."""
    target_w, target_h = FLAG_CANVAS_WIDTH, FLAG_CANVAS_HEIGHT
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = image.resize(
        (round(src_w * scale), round(src_h * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    normalized = resized.crop((left, top, left + target_w, top + target_h))
    draw = ImageDraw.Draw(normalized)
    for inset in range(FLAG_BORDER_WIDTH):
        draw.rectangle(
            (
                inset,
                inset,
                target_w - 1 - inset,
                target_h - 1 - inset,
            ),
            outline=FLAG_BORDER_COLOR,
        )
    return normalized


def load_flag_image(country: str, zoom: float = 0.35) -> OffsetImage | None:
    """Load a cached flag image for matplotlib embedding."""
    code = COUNTRY_FLAG_CODES.get(country)
    if not code:
        return None
    FLAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FLAG_CACHE_DIR / f"{code.replace('-', '_')}.png"
    if not cache_path.exists():
        url = f"https://flagcdn.com/w80/{code}.png"
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                cache_path.write_bytes(response.read())
        except (urllib.error.URLError, TimeoutError, OSError):
            return None
    try:
        image = _normalize_flag_image(Image.open(cache_path).convert("RGBA"))
    except OSError:
        return None
    return OffsetImage(np.asarray(image), zoom=zoom)


def flag_slot_size(
    fig: plt.Figure,
    ax: plt.Axes,
    width: float,
) -> tuple[float, float]:
    """Return axes-coordinate size for a visually proportional flag slot."""
    pos = ax.get_position()
    axis_width = pos.width * fig.get_figwidth()
    axis_height = pos.height * fig.get_figheight()
    if axis_height == 0:
        return width, width * FLAG_CANVAS_HEIGHT / FLAG_CANVAS_WIDTH
    height = (
        width
        * FLAG_CANVAS_HEIGHT
        / FLAG_CANVAS_WIDTH
        * axis_width
        / axis_height
    )
    return width, height


def draw_flag_slot(
    ax: plt.Axes,
    fig: plt.Figure,
    xy: tuple[float, float],
    width: float,
    *,
    on_dark: bool = False,
    zorder: float = 1.5,
) -> None:
    """Draw a fixed-size frame so every flag occupies the same template area."""
    slot_w, slot_h = flag_slot_size(fig, ax, width)
    left = xy[0] - slot_w / 2
    bottom = xy[1] - slot_h / 2
    if on_dark:
        facecolor = (1, 1, 1, 0.12)
        edgecolor = (1, 1, 1, 0.35)
    else:
        facecolor = "#f8f9fa"
        edgecolor = "#dee2e6"
    add_rounded_rect(
        ax,
        (left, bottom),
        slot_w,
        slot_h,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
        zorder=zorder,
    )


def flag_zoom_for_slot(
    fig: plt.Figure,
    ax: plt.Axes,
    slot_width: float,
    *,
    pad: float = 0.90,
) -> float:
    """Return OffsetImage zoom so a normalized flag fits the slot width."""
    pos = ax.get_position()
    slot_w, slot_h = flag_slot_size(fig, ax, slot_width)
    slot_w_pt = slot_w * pos.width * fig.get_figwidth() * fig.dpi
    slot_h_pt = slot_h * pos.height * fig.get_figheight() * fig.dpi
    zoom_w = slot_w_pt / FLAG_CANVAS_WIDTH
    zoom_h = slot_h_pt / FLAG_CANVAS_HEIGHT
    return min(zoom_w, zoom_h) * pad


def add_flag_in_slot(
    ax: plt.Axes,
    fig: plt.Figure,
    country: str,
    xy: tuple[float, float],
    slot_width: float,
    *,
    on_dark: bool = False,
    pad: float = 0.90,
    draw_slot: bool = True,
    slot_zorder: float = 1.5,
    flag_zorder: float = 5,
) -> AnnotationBbox | None:
    """Draw a fixed slot and place a uniformly sized flag inside it."""
    if draw_slot:
        draw_flag_slot(
            ax, fig, xy, slot_width, on_dark=on_dark, zorder=slot_zorder,
        )
    zoom = flag_zoom_for_slot(fig, ax, slot_width, pad=pad)
    return add_flag(
        ax, country, xy, zoom, zorder=flag_zorder,
    )


def add_flag(
    ax: plt.Axes,
    country: str,
    xy: tuple[float, float],
    zoom: float,
    box_alignment: tuple[float, float] = (0.5, 0.5),
    *,
    zorder: float = 5,
) -> AnnotationBbox | None:
    """Place a flag image on an axes."""
    flag = load_flag_image(country, zoom=zoom)
    if flag is None:
        return None
    artist = AnnotationBbox(
        flag, xy, frameon=False, box_alignment=box_alignment, zorder=zorder,
    )
    ax.add_artist(artist)
    return artist


def add_rounded_rect(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = "none",
    linewidth: float = 0,
    alpha: float = 1.0,
    corner_radius: float | None = None,
    zorder: float = 2,
) -> FancyBboxPatch:
    """Draw a rounded rectangle with a modest corner radius."""
    radius = corner_radius if corner_radius is not None else min(height * 0.14, 0.015)
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_white_card(
    ax: plt.Axes,
    rect: tuple[float, float, float, float],
    *,
    margin: float = 0.03,
) -> None:
    """Draw a rounded white panel inside normalized axes coordinates."""
    left, bottom, width, height = rect
    ax.add_patch(FancyBboxPatch(
        (left + margin, bottom + margin * 0.5),
        width - 2 * margin,
        height - margin,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=CARD_FACE,
        edgecolor=CARD_EDGE,
        linewidth=1.2,
        zorder=0,
    ))


def draw_gradient_band(ax: plt.Axes) -> None:
    """Fill axes with the standard infographic header gradient."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "hdr", list(HEADER_GRADIENT),
    )
    ax.imshow(gradient, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1], zorder=0)


def draw_score_dots(
    ax: plt.Axes,
    x: float,
    y: float,
    pct: float,
    color: str,
    n_dots: int = 10,
) -> None:
    """Render a compact tick matrix for match-outcome frequency."""
    fill_units = min(pct, 100) / 100 * n_dots
    cols = 5
    tick_w = 0.024
    tick_h = 0.018
    gap_x = 0.028
    gap_y = 0.034
    for index in range(n_dots):
        row, col = divmod(index, cols)
        cx = x + col * gap_x
        cy = y - row * gap_y
        left = cx - tick_w / 2
        bottom = cy - tick_h / 2
        add_rounded_rect(
            ax,
            (left, bottom),
            tick_w,
            tick_h,
            facecolor="#dee2e6",
            corner_radius=0.003,
            zorder=3,
        )
        segment_fill = min(1.0, max(0.0, fill_units - index))
        if segment_fill <= 0:
            continue
        add_rounded_rect(
            ax,
            (left, bottom),
            tick_w * segment_fill,
            tick_h,
            facecolor=color,
            corner_radius=0.003,
            zorder=4,
        )


def _artist_right_edge_x(
    ax: plt.Axes,
    fig: plt.Figure,
    artist: plt.Artist,
) -> float:
    """Return the data-x coordinate at the right edge of a drawn artist."""
    fig.canvas.draw()
    bbox = artist.get_window_extent(fig.canvas.get_renderer())
    return ax.transData.inverted().transform((bbox.x1, bbox.y0))[0]


def text_right_edge_x(
    ax: plt.Axes,
    fig: plt.Figure,
    text_obj: plt.Text,
) -> float:
    """Return the data-x coordinate at the right edge of a text object."""
    return _artist_right_edge_x(ax, fig, text_obj)


def save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Save a figure as PNG and close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=PAGE_BG)
    plt.close(fig)
    return output_path
