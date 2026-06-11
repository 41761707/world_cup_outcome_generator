"""Generate per-team tournament infographics from one Monte Carlo run."""

from __future__ import annotations

import argparse
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
from PIL import Image

from main import run_monte_carlo

BASE_DIR = Path(__file__).resolve().parent
COUNTRIES_FILE = BASE_DIR / "countries.txt"
GROUPS_FILE = BASE_DIR / "groups.txt"
SCHEDULE_GROUPS_FILE = BASE_DIR / "schedule_groups.txt"
SCHEDULE_KNOCKOUT_FILE = BASE_DIR / "schedule_knockout.txt"
OUTPUT_DIR = BASE_DIR / "infographic"
FLAG_CACHE_DIR = OUTPUT_DIR / ".flag_cache"

N_SIMULATIONS = 1_000
LAMBDA_BASE = 1.3
K_FACTOR = 0.3
REACH_HIDE_THRESHOLD = 95.0
EXIT_SLICE_MIN_PCT = 1.5

STAGES_ORDER = [
    "Grupa", "R_32", "R_16", "QF", "SF", "3RD", "F", "Zwycięzca",
]
STAGES_LABELS = {
    "Grupa": "Faza grupowa",
    "R_32": "1/16 finału",
    "R_16": "1/8 finału",
    "QF": "Ćwierćfinał",
    "SF": "Półfinał",
    "3RD": "3. miejsce",
    "F": "Finał",
    "Zwycięzca": "Zwycięzca",
}
STAGE_COLORS = {
    "Faza grupowa": "#868e96",
    "1/16 finału": "#4dabf7",
    "1/8 finału": "#228be6",
    "Ćwierćfinał": "#fab005",
    "Półfinał": "#f76707",
    "3. miejsce": "#be4bdb",
    "Finał": "#fa5252",
    "Zwycięzca": "#40c057",
    "Inne": "#ced4da",
}
KO_STAGES = ["R_32", "R_16", "QF", "SF", "3RD", "F"]
ROUND_LABELS = {
    "R_32": "1/16 finału",
    "R_16": "1/8 finału",
    "QF": "Ćwierćfinały",
    "SF": "Półfinały",
    "3RD": "Mecz o 3. miejsce",
    "F": "Finał",
}
CUMULATIVE_STAGES = [
    ("1/16 finału", ("R_32", "R_16", "QF", "SF", "3RD", "F", "Zwycięzca")),
    ("1/8 finału", ("R_16", "QF", "SF", "3RD", "F", "Zwycięzca")),
    ("Ćwierćfinał", ("QF", "SF", "3RD", "F", "Zwycięzca")),
    ("Półfinał", ("SF", "3RD", "F", "Zwycięzca")),
    ("Finał", ("F", "Zwycięzca")),
    ("Zwycięzca", ("Zwycięzca",)),
]
KO_REACH_LABEL = {
    "R_32": "1/16 finału",
    "R_16": "1/8 finału",
    "QF": "Ćwierćfinał",
    "SF": "Półfinał",
    "3RD": "Półfinał",
    "F": "Finał",
}
REACH_BAR_COLORS = ["#12b886", "#20c997", "#22b8cf", "#339af0", "#7950f2", "#e64980"]
PAGE_BG = "#eef2f7"
HEADER_GRADIENT = ("#0b1d3a", "#1e3a6e", "#2d5aa0")
TIER_EXIT_STAGES: dict[str, tuple[str, ...]] = {
    "Faza grupowa": ("Grupa",),
    "1/16 finału": ("R_32",),
    "1/8 finału": ("R_16",),
    "Ćwierćfinał": ("QF",),
    "Półfinał": ("SF", "3RD"),
    "Finał": ("F",),
    "Zwycięzca": ("Zwycięzca",),
}

COUNTRY_FLAG_CODES: dict[str, str] = {
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
            names = [n.strip() for n in rest.split(",")]
            members[group_name.strip()] = names
            for name in names:
                team_group[name] = group_name.strip()
    return members, team_group


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
        image = Image.open(cache_path).convert("RGBA")
    except OSError:
        return None
    return OffsetImage(np.asarray(image), zoom=zoom)


def add_flag(
    ax: plt.Axes,
    country: str,
    xy: tuple[float, float],
    zoom: float,
    *,
    box_alignment: tuple[float, float] = (0.5, 0.5),
) -> AnnotationBbox | None:
    """Place a flag image on an axes."""
    flag = load_flag_image(country, zoom=zoom)
    if flag is None:
        return None
    artist = AnnotationBbox(
        flag, xy, frameon=False, box_alignment=box_alignment,
    )
    ax.add_artist(artist)
    return artist


def _artist_right_edge_x(
    ax: plt.Axes,
    fig: plt.Figure,
    artist: plt.Artist,
) -> float:
    """Return the data-x coordinate at the right edge of a drawn artist."""
    fig.canvas.draw()
    bbox = artist.get_window_extent(fig.canvas.get_renderer())
    return ax.transData.inverted().transform((bbox.x1, bbox.y0))[0]


def _text_right_edge_x(
    ax: plt.Axes,
    fig: plt.Figure,
    text_obj: plt.Text,
) -> float:
    """Return the data-x coordinate at the right edge of a text object."""
    return _artist_right_edge_x(ax, fig, text_obj)


GROUP_CARD_TEAM_FLAG_X = 0.075
GROUP_CARD_TEXT_X = 0.12
GROUP_CARD_TEXT_FLAG_GAP = 0.03
GROUP_CARD_SCORE_DOTS_GAP = 0.06
GROUP_CARD_SCORE_W = 0.13
GROUP_CARD_DOTS_W = 0.19
GROUP_CARD_RIGHT_MARGIN = 0.95
WR_BADGE_FILLS = {"#2f9e44": "#b2f2bb", "#e03131": "#ffc9c9"}
SCORE_BADGE_FILLS = {
    "#339af0": "#a5d8ff",
    "#37b24d": "#8ce99a",
    "#f59f00": "#ffe066",
}


def _add_rounded_rect(
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


def _opponent_matchup_label(opponent: str) -> tuple[str, float, float]:
    """Return matchup text, font size and line spacing for group cards."""
    if len(opponent) <= 20:
        return f"vs {opponent}", 21, 1.0
    if len(opponent) <= 28:
        return f"vs {opponent}", 17, 1.0
    words = opponent.split()
    if len(words) >= 2:
        mid = (len(words) + 1) // 2
        line1 = "vs " + " ".join(words[:mid])
        line2 = " ".join(words[mid:])
        return f"{line1}\n{line2}", 14, 1.35
    return f"vs {opponent}", 14, 1.0


def _group_card_score_layout() -> tuple[float, float, float]:
    """Return fixed right-aligned score-box and dots x positions."""
    dots_x = GROUP_CARD_RIGHT_MARGIN - GROUP_CARD_DOTS_W
    score_left = dots_x - GROUP_CARD_SCORE_DOTS_GAP - GROUP_CARD_SCORE_W
    score_cx = score_left + GROUP_CARD_SCORE_W / 2
    return score_left, score_cx, dots_x


def reach_probability(stages: list[str], stage_set: tuple[str, ...], n: int) -> float:
    """Return cumulative reach percentage for a stage set."""
    return 100 * sum(1 for s in stages if s in stage_set) / n


def meaningful_cumulative_stages(
    stages: list[str],
    n: int,
    threshold: float = REACH_HIDE_THRESHOLD,
) -> list[tuple[str, tuple[str, ...]]]:
    """Skip early stages that are nearly guaranteed for strong teams."""
    visible = []
    for label, stage_set in CUMULATIVE_STAGES:
        pct = reach_probability(stages, stage_set, n)
        if pct < threshold:
            visible.append((label, stage_set))
    return visible or [CUMULATIVE_STAGES[-1]]


def meaningful_ko_stages(
    stages: list[str],
    n: int,
    threshold: float = REACH_HIDE_THRESHOLD,
) -> list[str]:
    """Return knockout rounds worth showing for this team."""
    visible = []
    seen_labels: set[str] = set()
    for stage in KO_STAGES:
        reach_label = KO_REACH_LABEL[stage]
        if reach_label in seen_labels:
            continue
        seen_labels.add(reach_label)
        stage_set = next(s for lbl, s in CUMULATIVE_STAGES if lbl == reach_label)
        if reach_probability(stages, stage_set, n) < threshold:
            visible.append(stage)
    return visible


def exit_stage_distribution(
    stages: list[str],
    min_pct: float = EXIT_SLICE_MIN_PCT,
) -> list[tuple[str, int, float]]:
    """Return exit-stage counts; tiny slices are merged into 'Inne'."""
    n = len(stages)
    rows: list[tuple[str, int, float]] = []
    other_count = 0
    for stage in STAGES_ORDER:
        count = stages.count(stage)
        if count == 0:
            continue
        pct = 100 * count / n
        label = STAGES_LABELS[stage]
        if pct < min_pct:
            other_count += count
        else:
            rows.append((label, count, pct))
    if other_count:
        rows.append(("Inne", other_count, 100 * other_count / n))
    return rows


def compute_winner_ranks(stats: dict[str, Any]) -> dict[str, int]:
    """Rank teams by simulated tournament win rate (1 = most likely winner)."""
    n = stats["n_simulations"]
    rates = {
        team: stages.count("Zwycięzca") / n
        for team, stages in stats["team_exit_stages"].items()
    }
    ordered = sorted(rates.keys(), key=lambda name: (-rates[name], name))
    return {team: index + 1 for index, team in enumerate(ordered)}


def tier_exit_label(rank: int) -> str:
    """Map global favorite rank to the expected exit stage shown in the header."""
    if rank <= 2:
        return "Finał"
    if rank <= 4:
        return "Półfinał"
    if rank <= 8:
        return "Ćwierćfinał"
    if rank <= 16:
        return "1/8 finału"
    if rank <= 32:
        return "1/16 finału"
    return "Faza grupowa"


def exit_pct_at_label(stages: list[str], n: int, label: str) -> float:
    """Return share of simulations where the team exits at the given tier."""
    codes = TIER_EXIT_STAGES[label]
    return 100 * sum(stages.count(code) for code in codes) / n


def team_headline(
    team: str,
    stages: list[str],
    n: int,
    winner_ranks: dict[str, int],
) -> tuple[str, str, str]:
    """Return title line, highlight text and accent color for the header."""
    winner_pct = 100 * stages.count("Zwycięzca") / n
    rank = winner_ranks.get(team, 48)
    tier_label = tier_exit_label(rank)
    tier_pct = exit_pct_at_label(stages, n, tier_label)

    if winner_pct >= 12:
        return (
            "Faworyt do mistrzostwa",
            f"{winner_pct:.1f}% szans na złoto",
            "#ffd43b",
        )
    if rank <= 16:
        return (
            "Oczekiwany etap zakończenia",
            f"{tier_label}  ·  {tier_pct:.1f}%",
            STAGE_COLORS.get(tier_label, "#dee2e6"),
        )
    dist = exit_stage_distribution(stages, min_pct=0.0)
    modal_label, _, modal_pct = max(dist, key=lambda row: row[1])
    return (
        "Najczęstszy etap zakończenia",
        f"{modal_label}  ·  {modal_pct:.1f}%",
        STAGE_COLORS.get(modal_label, "#dee2e6"),
    )


def cumulative_reach(
    stages: list[str],
    n: int,
) -> list[tuple[str, float]]:
    """Return filtered cumulative reach probabilities."""
    return [
        (label, reach_probability(stages, stage_set, n))
        for label, stage_set in meaningful_cumulative_stages(stages, n)
    ]


def top_opponents_by_stage(
    team: str,
    stats: dict[str, Any],
    n: int,
) -> list[tuple[str, str, float, float]]:
    """Return the most likely opponent at each knockout round with data."""
    ko_counts = stats["knockout_meeting_counts"]
    ko_wins = stats.get("knockout_meeting_wins", {})
    by_stage: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for (pair, stage), count in ko_counts.items():
        if team not in pair or stage not in KO_STAGES:
            continue
        opponent = next(iter(pair - {team}))
        wins = ko_wins.get((pair, stage), {}).get(team, 0)
        by_stage[stage].append((opponent, count, wins))

    rows: list[tuple[str, str, float, float]] = []
    for stage in KO_STAGES:
        stage_rows = by_stage.get(stage)
        if not stage_rows:
            continue
        opponent, count, wins = max(stage_rows, key=lambda row: row[1])
        meet_pct = 100 * count / n
        win_pct = 100 * wins / count if count else 0.0
        rows.append((ROUND_LABELS[stage], opponent, meet_pct, win_pct))
    return rows


def group_opponent_scores(
    team: str,
    opponents: list[str],
    stats: dict[str, Any],
    n: int,
) -> list[tuple[str, str, float, int]]:
    """Return the modal group-stage score vs each opponent."""
    score_counts = stats.get("group_match_score_counts", {})
    rows: list[tuple[str, str, float, int]] = []
    for opponent in opponents:
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for (s1, s2), count in score_counts.get((team, opponent), {}).items():
            counts[(s1, s2)] += count
        for (s1, s2), count in score_counts.get((opponent, team), {}).items():
            counts[(s2, s1)] += count
        if not counts:
            rows.append((opponent, "—", 0.0, 0))
            continue
        (gs, os), best = max(counts.items(), key=lambda item: item[1])
        rows.append((opponent, f"{gs}:{os}", 100 * best / n, best))
    return rows


def _draw_charts_background(fig: plt.Figure, rect: list[float]) -> None:
    """Draw a white card behind the top chart row."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#ffffff", edgecolor="#dee2e6", linewidth=1.2,
        zorder=0,
    ))


def _draw_gradient_header(
    ax: plt.Axes,
    team: str,
    stages: list[str],
    n: int,
    winner_ranks: dict[str, int],
) -> None:
    """Render a colorful header band with team identity."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "hdr", list(HEADER_GRADIENT),
    )
    ax.imshow(gradient, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1], zorder=0)
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.12), 0.96, 0.76,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=(1, 1, 1, 0.08), edgecolor=(1, 1, 1, 0.25), linewidth=1.5,
        zorder=1,
    ))
    add_flag(ax, team, (0.09, 0.5), zoom=1.5)
    ax.text(0.19, 0.62, team, fontsize=30, fontweight="bold", color="white", zorder=2)
    title, highlight, accent = team_headline(team, stages, n, winner_ranks)
    ax.text(0.19, 0.38, title, fontsize=16, color="#c8d6e5", zorder=2)
    ax.text(0.19, 0.22, highlight, fontsize=22, fontweight="bold", color=accent, zorder=2)
    ax.text(
        0.97, 0.1,
        f"{n:,} sym.  ·  k={K_FACTOR}  ·  λ={LAMBDA_BASE}",
        fontsize=25, color="#a4b0be", ha="right", zorder=2,
    )


def draw_exit_donut(fig: plt.Figure, rect: list[float], stages: list[str]) -> None:
    """Draw a donut chart of exit-stage distribution."""
    ax = fig.add_axes(rect)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.axis("off")
    ax.set_title("Gdzie kończy turniej?", fontsize=18, fontweight="bold", pad=6, y=0.96)
    dist = exit_stage_distribution(stages)
    labels = [row[0] for row in dist]
    sizes = [row[1] for row in dist]
    colors = [STAGE_COLORS.get(label, "#adb5bd") for label in labels]
    top_idx = int(np.argmax(sizes))
    explode = [0.06 if i == top_idx else 0 for i in range(len(sizes))]
    pie_ax = ax.inset_axes([0.0, 0.22, 1.0, 0.76])
    pie_ax.set_aspect("equal")
    wedges, _, autotexts = pie_ax.pie(
        sizes,
        labels=None,
        colors=colors,
        explode=explode,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        startangle=120,
        pctdistance=0.78,
        wedgeprops=dict(width=0.40, edgecolor="white", linewidth=2.5),
        textprops={"fontsize": 16, "fontweight": "bold", "color": "#212529"},
    )
    for autotext in autotexts:
        autotext.set_visible(autotext.get_text() != "")
    ncol = 2 if len(labels) > 4 else 1
    leg_ax = ax.inset_axes([0.0, 0.0, 1.0, 0.20])
    leg_ax.axis("off")
    leg_ax.legend(
        wedges,
        labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize=12,
        handlelength=1.0,
        columnspacing=1.2,
        handletextpad=0.4,
    )


def draw_reach_bars(fig: plt.Figure, rect: list[float], stages: list[str], n: int) -> None:
    """Draw a vibrant horizontal bar chart of stage-reach probabilities."""
    ax = fig.add_axes(rect)
    reach = cumulative_reach(stages, n)
    if not reach:
        ax.axis("off")
        return
    labels = [row[0] for row in reach]
    values = [row[1] for row in reach]
    y_pos = np.arange(len(labels))
    colors = REACH_BAR_COLORS[:len(labels)][::-1]
    bars = ax.barh(
        y_pos, values, color=colors, height=0.62,
        edgecolor="white", linewidth=2,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=15, fontweight="bold")
    ax.set_xlim(0, 108)
    ax.set_xlabel("Prawdopodobieństwo (%)", fontsize=16, color="#495057")
    ax.set_title(
        "Szanse na dotarcie do danej fazy turnieju",
        fontsize=16, fontweight="bold", pad=14, loc="left", y=1.02,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_facecolor("#f8f9fa")
    for bar, value in zip(bars, values):
        ax.text(
            value + 1.2, bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center", fontsize=15, fontweight="bold", color="#212529",
        )


def draw_opponents_panel(
    fig: plt.Figure,
    rect: list[float],
    team: str,
    stats: dict[str, Any],
    stages: list[str],
    n: int,
) -> None:
    """Draw opponent cards for meaningful knockout rounds."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rows = top_opponents_by_stage(team, stats, n)
    ax.text(
        0.03, 0.97, "Najbardziej prawdopodobni rywale",
        fontsize=19, fontweight="bold", color="#212529", va="top",
    )
    if not rows:
        ax.text(
            0.03, 0.5,
            "Brak istotnych danych o rywalach w fazie pucharowej.",
            fontsize=17, color="#868e96",
        )
        return
    card_h = min(0.17, 0.82 / len(rows))
    y = 0.88
    for idx, (stage_label, opponent, meet_pct, win_pct) in enumerate(rows):
        bg = "#f1f3f5" if idx % 2 == 0 else "#ffffff"
        ax.add_patch(FancyBboxPatch(
            (0.03, y - card_h + 0.02), 0.94, card_h - 0.04,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor=bg, edgecolor="#dee2e6", linewidth=1,
        ))
        ax.text(0.06, y - 0.03, stage_label, fontsize=14, color="#868e96", fontweight="bold")
        add_flag(ax, opponent, (0.08, y - card_h / 2 - 0.01), zoom=0.45)
        ax.text(
            0.13, y - card_h / 2 - 0.02, opponent,
            fontsize=22, fontweight="bold", va="center",
        )
        ax.text(
            0.55, y - card_h / 2 - 0.02,
            f"{meet_pct:.1f}% spotkań",
            fontsize=20, color="#495057", va="center",
        )
        wr_color = "#2f9e44" if win_pct >= 50 else "#e03131"
        _add_rounded_rect(
            ax, (0.78, y - card_h / 2 - 0.045), 0.16, 0.09,
            facecolor=WR_BADGE_FILLS[wr_color], zorder=3,
        )
        ax.text(
            0.86, y - card_h / 2 - 0.02,
            f"{win_pct:.0f}% WR",
            fontsize=18, fontweight="bold", color=wr_color,
            ha="center", va="center",
        )
        y -= card_h


def _draw_score_dots(
    ax: plt.Axes,
    x: float,
    y: float,
    pct: float,
    color: str,
    n_dots: int = 12,
) -> None:
    """Render a compact tick matrix for match-outcome frequency."""
    filled = int(round(min(pct, 100) / 100 * n_dots))
    cols = 6
    tick_w = 0.024
    tick_h = 0.018
    gap_x = 0.028
    gap_y = 0.034
    for index in range(n_dots):
        row, col = divmod(index, cols)
        cx = x + col * gap_x
        cy = y - row * gap_y
        face = color if index < filled else "#dee2e6"
        _add_rounded_rect(
            ax,
            (cx - tick_w / 2, cy - tick_h / 2),
            tick_w,
            tick_h,
            facecolor=face,
            corner_radius=0.003,
            zorder=3,
        )


def draw_group_panel(
    fig: plt.Figure,
    rect: list[float],
    team: str,
    group_name: str,
    opponents: list[str],
    stats: dict[str, Any],
    n: int,
) -> None:
    """Draw group-stage score cards with score badges and dot frequency."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.03, 0.97, f"Grupa {group_name} — najczęstsze wyniki",
        fontsize=20, fontweight="bold", color="#212529", va="top",
    )
    scores = group_opponent_scores(team, opponents, stats, n)
    base_card_h = min(0.25, 0.86 / max(len(scores), 1))
    y = 0.86
    palette = ["#339af0", "#37b24d", "#f59f00"]
    for idx, (opponent, score, pct, count) in enumerate(scores):
        accent = palette[idx % 3]
        matchup_label, matchup_fontsize, line_spacing = _opponent_matchup_label(opponent)
        line_count = matchup_label.count("\n") + 1
        card_h = base_card_h * (1.12 if line_count > 1 else 1.0)
        mid_y = y - card_h / 2
        ax.add_patch(FancyBboxPatch(
            (0.03, y - card_h + 0.02), 0.94, card_h - 0.04,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor="#ffffff", edgecolor="#dee2e6", linewidth=1,
        ))
        ax.add_patch(FancyBboxPatch(
            (0.03, y - card_h + 0.02), 0.012, card_h - 0.04,
            boxstyle="round,pad=0,rounding_size=0.01",
            facecolor=accent, edgecolor="none",
        ))
        add_flag(ax, team, (GROUP_CARD_TEAM_FLAG_X, mid_y), zoom=0.62)
        vs_text = ax.text(
            GROUP_CARD_TEXT_X, mid_y, matchup_label,
            fontsize=matchup_fontsize, fontweight="bold",
            va="center", linespacing=line_spacing,
        )
        opponent_flag_x = _text_right_edge_x(ax, fig, vs_text) + GROUP_CARD_TEXT_FLAG_GAP
        add_flag(
            ax, opponent, (opponent_flag_x, mid_y),
            zoom=0.62, box_alignment=(0.0, 0.5),
        )
        score_left, score_cx, dots_x = _group_card_score_layout()
        score_h = 0.13
        _add_rounded_rect(
            ax,
            (score_left, mid_y - score_h / 2),
            GROUP_CARD_SCORE_W,
            score_h,
            facecolor=SCORE_BADGE_FILLS[accent],
            zorder=3,
        )
        ax.text(
            score_cx, mid_y, score,
            fontsize=23, fontweight="bold", color=accent,
            ha="center", va="center",
        )
        _draw_score_dots(ax, dots_x, mid_y + 0.045, pct, accent)
        ax.text(
            dots_x + GROUP_CARD_DOTS_W / 2, mid_y - 0.095,
            f"{pct:.1f}%",
            fontsize=15, fontweight="bold", color="#495057",
            ha="center",
        )
        y -= card_h + 0.01


def generate_team_infographic(
    team: str,
    stats: dict[str, Any],
    team_group: dict[str, str],
    group_members: dict[str, list[str]],
    output_dir: Path,
    winner_ranks: dict[str, int],
) -> Path:
    """Build and save one team's infographic PNG."""
    stages = stats["team_exit_stages"][team]
    n = stats["n_simulations"]
    group_name = team_group[team]
    opponents = [name for name in group_members[group_name] if name != team]

    fig = plt.figure(figsize=(14, 20), facecolor=PAGE_BG)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.97, bottom=0.02)

    _draw_gradient_header(
        fig.add_axes([0.04, 0.905, 0.92, 0.085]),
        team, stages, n, winner_ranks,
    )
    _draw_charts_background(fig, [0.03, 0.51, 0.94, 0.37])
    draw_exit_donut(fig, [0.05, 0.52, 0.43, 0.35], stages)
    draw_reach_bars(fig, [0.51, 0.54, 0.44, 0.33], stages, n)
    draw_opponents_panel(
        fig, [0.04, 0.27, 0.92, 0.22], team, stats, stages, n,
    )
    draw_group_panel(
        fig, [0.04, 0.03, 0.92, 0.23], team, group_name, opponents, stats, n,
    )

    team_dir = output_dir / team
    team_dir.mkdir(parents=True, exist_ok=True)
    out_path = team_dir / "infographic.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=PAGE_BG)
    plt.close(fig)
    return out_path


def run_infographic_generation(
    n_simulations: int = N_SIMULATIONS,
    teams: list[str] | None = None,
) -> list[Path]:
    """Run one simulation and export infographics."""
    print(
        f"Uruchamiam {n_simulations:,} symulacji "
        f"(k={K_FACTOR}, lambda_base={LAMBDA_BASE})..."
    )
    with contextlib.redirect_stdout(io.StringIO()):
        stats = run_monte_carlo(
            str(COUNTRIES_FILE),
            str(GROUPS_FILE),
            str(SCHEDULE_GROUPS_FILE),
            str(SCHEDULE_KNOCKOUT_FILE),
            n=n_simulations,
            lambda_base=LAMBDA_BASE,
            k=K_FACTOR,
        )
    print("Symulacja zakończona. Generuję infografiki...")
    group_members, team_group = load_groups_data()
    winner_ranks = compute_winner_ranks(stats)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_teams = sorted(stats["team_exit_stages"].keys())
    target_teams = [t for t in (teams or all_teams) if t in stats["team_exit_stages"]]
    paths: list[Path] = []
    for team in target_teams:
        path = generate_team_infographic(
            team, stats, team_group, group_members, OUTPUT_DIR, winner_ranks,
        )
        paths.append(path)
        print(f"  Zapisano: {path}")
    print(f"\nGotowe — {len(paths)} infografik w folderze {OUTPUT_DIR}")
    return paths


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generuj infografiki MŚ")
    parser.add_argument(
        "-n", "--simulations", type=int, default=N_SIMULATIONS,
        help="Liczba symulacji (domyślnie 1000)",
    )
    parser.add_argument(
        "--teams", nargs="*", help="Opcjonalnie: tylko wybrane drużyny",
    )
    args = parser.parse_args()
    run_infographic_generation(n_simulations=args.simulations, teams=args.teams)


if __name__ == "__main__":
    main()
