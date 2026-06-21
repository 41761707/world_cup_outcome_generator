"""Generate head-to-head match forecast infographic for two group rivals."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from infographic_common import (
    K_FACTOR,
    LAMBDA_BASE,
    N_SIMULATIONS,
    OUTPUT_DIR,
    PAGE_BG,
    add_flag_in_slot,
    add_rounded_rect,
    collect_match_score_counts,
    draw_gradient_band,
    draw_score_dots,
    draw_white_card,
    load_groups_data,
    run_simulation,
    save_figure,
    team_name_label,
)

TOP_SCORES_LIMIT = 6
MIN_SCORE_PCT = 2.0
SCORE_PALETTE = ["#339af0", "#37b24d", "#f59f00", "#7950f2", "#e64980", "#22b8cf"]
H2H_SCORE_BADGE_FILLS = {
    "#339af0": "#a5d8ff",
    "#37b24d": "#8ce99a",
    "#f59f00": "#ffe066",
    "#7950f2": "#d0bfff",
    "#e64980": "#ffdeeb",
    "#22b8cf": "#99e9f2",
}
TEAM_A_COLOR = "#339af0"
TEAM_B_COLOR = "#f76707"
DRAW_COLOR = "#868e96"


@dataclass(frozen=True)
class HeadToHeadData:
    """Aggregated simulation stats for one group-stage fixture."""

    team_a: str
    team_b: str
    group_name: str
    n_simulations: int
    team_a_win_pct: float
    draw_pct: float
    team_b_win_pct: float
    top_scores: list[tuple[str, float, int]]
    total_goals: list[tuple[int, float, int]]
    btts_yes_pct: float
    btts_no_pct: float
    modal_score: str
    modal_pct: float


def validate_same_group(team_a: str, team_b: str) -> str:
    """Ensure both teams exist and share a group."""
    if team_a == team_b:
        raise ValueError("Podaj dwie rozne druzyny.")
    _, team_group = load_groups_data()
    if team_a not in team_group:
        raise ValueError(f"Nieznana druzyna: {team_a}")
    if team_b not in team_group:
        raise ValueError(f"Nieznana druzyna: {team_b}")
    group_name = team_group[team_a]
    if team_group[team_b] != group_name:
        raise ValueError(
            f"{team_a} (gr. {group_name}) i {team_b} (gr. {team_group[team_b]}) "
            "nie graja ze soba w fazie grupowej."
        )
    return group_name


def build_head_to_head_data(
    stats: dict,
    team_a: str,
    team_b: str,
    group_name: str,
) -> HeadToHeadData:
    """Build headline and distribution rows from simulation stats."""
    n = stats["n_simulations"]
    counts = collect_match_score_counts(stats, team_a, team_b)
    if not counts:
        raise ValueError(f"Brak danych o meczu {team_a} vs {team_b}.")
    total = sum(counts.values())
    team_a_wins = sum(c for (s1, s2), c in counts.items() if s1 > s2)
    draws = sum(c for (s1, s2), c in counts.items() if s1 == s2)
    team_b_wins = sum(c for (s1, s2), c in counts.items() if s1 < s2)
    btts_yes = sum(c for (s1, s2), c in counts.items() if s1 > 0 and s2 > 0)
    total_goal_counts: dict[int, int] = {}
    for (s1, s2), count in counts.items():
        total_goals = s1 + s2
        total_goal_counts[total_goals] = total_goal_counts.get(total_goals, 0) + count
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -item[0][0], item[0][1]))
    (modal_s1, modal_s2), modal_count = ranked[0]
    top_scores: list[tuple[str, float, int]] = []
    for (s1, s2), count in ranked[:TOP_SCORES_LIMIT]:
        pct = 100 * count / n
        if pct < MIN_SCORE_PCT and top_scores:
            break
        top_scores.append((f"{s1}:{s2}", pct, count))
    total_goals = [
        (goals, 100 * count / total, count)
        for goals, count in sorted(total_goal_counts.items())
    ]
    return HeadToHeadData(
        team_a=team_a,
        team_b=team_b,
        group_name=group_name,
        n_simulations=n,
        team_a_win_pct=100 * team_a_wins / total,
        draw_pct=100 * draws / total,
        team_b_win_pct=100 * team_b_wins / total,
        top_scores=top_scores,
        total_goals=total_goals,
        btts_yes_pct=100 * btts_yes / total,
        btts_no_pct=100 * (total - btts_yes) / total,
        modal_score=f"{modal_s1}:{modal_s2}",
        modal_pct=100 * modal_count / n,
    )


def run_head_to_head_simulation(
    team_a: str,
    team_b: str,
    n_simulations: int = N_SIMULATIONS,
    lambda_base: float = LAMBDA_BASE,
    k: float = K_FACTOR,
) -> HeadToHeadData:
    """Run Monte Carlo and aggregate head-to-head stats."""
    group_name = validate_same_group(team_a, team_b)
    stats = run_simulation(
        n=n_simulations,
        lambda_base=lambda_base,
        k=k,
    )
    return build_head_to_head_data(stats, team_a, team_b, group_name)


def _draw_header(ax: plt.Axes, fig: plt.Figure, data: HeadToHeadData) -> None:
    """Render gradient header with both teams."""
    draw_gradient_band(ax)
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.08), 0.96, 0.86,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=(1, 1, 1, 0.08), edgecolor=(1, 1, 1, 0.25), linewidth=1.5,
        zorder=1,
    ))
    ax.text(
        0.5, 0.94, "Prognoza meczu",
        fontsize=22, fontweight="bold", color="white", ha="center", va="top", zorder=2,
    )
    ax.text(
        0.5, 0.80, f"Grupa {data.group_name}  ·  faza grupowa",
        fontsize=15, color="#c8d6e5", ha="center", va="top", zorder=2,
    )
    add_flag_in_slot(ax, fig, data.team_a, (0.24, 0.48), 0.085, on_dark=True)
    add_flag_in_slot(ax, fig, data.team_b, (0.76, 0.48), 0.085, on_dark=True)
    ax.text(
        0.5, 0.46, "vs",
        fontsize=22, fontweight="bold", color="#ffd43b", ha="center", va="center", zorder=2,
    )
    team_a_label, team_a_fs, team_a_ls = team_name_label(data.team_a, base_fontsize=20)
    team_b_label, team_b_fs, team_b_ls = team_name_label(data.team_b, base_fontsize=20)
    name_y = 0.20 if "\n" in team_a_label or "\n" in team_b_label else 0.22
    ax.text(
        0.24, name_y, team_a_label,
        fontsize=team_a_fs, fontweight="bold", color="white",
        ha="center", va="top", linespacing=team_a_ls, zorder=2,
    )
    ax.text(
        0.76, name_y, team_b_label,
        fontsize=team_b_fs, fontweight="bold", color="white",
        ha="center", va="top", linespacing=team_b_ls, zorder=2,
    )


def _draw_outcome_panel(ax: plt.Axes, data: HeadToHeadData) -> None:
    """Draw win/draw/win probability bars."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_white_card(ax, (0, 0, 1, 1))
    ax.text(
        0.05, 0.92, "Szanse na wynik meczu",
        fontsize=19, fontweight="bold", color="#212529", va="top",
    )
    ax.text(
        0.05, 0.82,
        f"Najbardziej prawdopodobny wynik: {data.modal_score} ({data.modal_pct:.1f}%)",
        fontsize=14, color="#495057", va="top",
    )
    bar_left = 0.08
    bar_track_width = 0.54
    pct_gap = 0.07
    pct_x = bar_left + bar_track_width + pct_gap
    bar_h = 0.10
    rows = [
        (f"Wygrana {data.team_a}", data.team_a_win_pct, TEAM_A_COLOR),
        ("Remis", data.draw_pct, DRAW_COLOR),
        (f"Wygrana {data.team_b}", data.team_b_win_pct, TEAM_B_COLOR),
    ]
    y_positions = [0.50, 0.27, 0.04]
    for (label, value, color), y in zip(rows, y_positions):
        ax.text(
            bar_left, y + bar_h + 0.055, label,
            fontsize=14, color="#868e96", fontweight="bold",
        )
        fill_w = bar_track_width * min(value, 100) / 100
        add_rounded_rect(
            ax, (bar_left, y), bar_track_width, bar_h,
            facecolor="#e9ecef", corner_radius=0.012, zorder=1,
        )
        if fill_w > 0:
            add_rounded_rect(
                ax, (bar_left, y), fill_w, bar_h,
                facecolor=color, corner_radius=0.012, zorder=2,
            )
        ax.text(
            pct_x, y + bar_h / 2,
            f"{value:.1f}%",
            fontsize=18, fontweight="bold", color="#212529",
            ha="left", va="center",
        )


def _draw_goals_panel(ax: plt.Axes, data: HeadToHeadData) -> None:
    """Draw total-goals distribution and BTTS percentages."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_white_card(ax, (0, 0, 1, 1))
    ax.text(
        0.05, 0.91, "Bramki w symulowanych meczach",
        fontsize=19, fontweight="bold", color="#212529", va="top",
    )
    if not data.total_goals:
        ax.text(0.05, 0.5, "Brak danych o bramkach.", fontsize=16, color="#868e96")
        return

    chart_left = 0.06
    chart_bottom = 0.16
    chart_width = 0.58
    chart_height = 0.54
    max_pct = max(pct for _goals, pct, _count in data.total_goals) or 1
    bar_count = len(data.total_goals)
    gap = 0.012
    bar_w = (chart_width - gap * (bar_count - 1)) / bar_count
    for index, (goals, pct, _count) in enumerate(data.total_goals):
        x = chart_left + index * (bar_w + gap)
        fill_h = chart_height * pct / max_pct
        add_rounded_rect(
            ax,
            (x, chart_bottom),
            bar_w,
            chart_height,
            facecolor="#e9ecef",
            corner_radius=0.006,
            zorder=1,
        )
        add_rounded_rect(
            ax,
            (x, chart_bottom),
            bar_w,
            fill_h,
            facecolor="#339af0",
            corner_radius=0.006,
            zorder=2,
        )
        ax.text(
            x + bar_w / 2,
            chart_bottom - 0.04,
            str(goals),
            fontsize=11,
            fontweight="bold",
            color="#495057",
            ha="center",
            va="top",
        )
        if pct >= 3:
            ax.text(
                x + bar_w / 2,
                chart_bottom + fill_h + 0.025,
                f"{pct:.0f}%",
                fontsize=10,
                fontweight="bold",
                color="#212529",
                ha="center",
                va="bottom",
            )
    ax.text(
        chart_left,
        0.08,
        "Łączna liczba bramek",
        fontsize=12,
        color="#868e96",
        ha="left",
    )

    btts_left = 0.72
    btts_rows = [
        ("Obie strzelą", data.btts_yes_pct, "#40c057"),
        ("Nie obie strzelą", data.btts_no_pct, "#fa5252"),
    ]
    ax.text(
        btts_left, 0.70, "BTTS",
        fontsize=16, fontweight="bold", color="#212529", va="center",
    )
    for idx, (label, pct, color) in enumerate(btts_rows):
        y = 0.48 - idx * 0.22
        add_rounded_rect(
            ax,
            (btts_left, y),
            0.20,
            0.12,
            facecolor="#e9ecef",
            corner_radius=0.014,
            zorder=1,
        )
        add_rounded_rect(
            ax,
            (btts_left, y),
            0.20 * pct / 100,
            0.12,
            facecolor=color,
            corner_radius=0.014,
            zorder=2,
        )
        ax.text(
            btts_left,
            y + 0.16,
            label,
            fontsize=12,
            fontweight="bold",
            color="#495057",
            va="bottom",
        )
        ax.text(
            btts_left + 0.24,
            y + 0.06,
            f"{pct:.1f}%",
            fontsize=18,
            fontweight="bold",
            color=color,
            va="center",
        )


def _draw_top_scores_panel(ax: plt.Axes, data: HeadToHeadData) -> None:
    """Draw ranked list of the most common simulated scores."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_white_card(ax, (0, 0, 1, 1))
    ax.text(
        0.05, 0.96,
        f"Najczestsze wyniki ({data.team_a} : {data.team_b})",
        fontsize=19, fontweight="bold", color="#212529", va="top",
    )
    if not data.top_scores:
        ax.text(0.05, 0.5, "Brak danych o wynikach.", fontsize=16, color="#868e96")
        return
    row_count = len(data.top_scores)
    card_h = min(0.15, 0.82 / row_count)
    y = 0.84
    for idx, (score, pct, _count) in enumerate(data.top_scores):
        accent = SCORE_PALETTE[idx % len(SCORE_PALETTE)]
        mid_y = y - card_h / 2
        ax.add_patch(FancyBboxPatch(
            (0.05, y - card_h + 0.015), 0.90, card_h - 0.03,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            facecolor="#f8f9fa" if idx % 2 == 0 else "#ffffff",
            edgecolor="#e9ecef", linewidth=0.8, zorder=1,
        ))
        ax.text(
            0.08, mid_y, f"{idx + 1}.",
            fontsize=16, fontweight="bold", color="#868e96",
            va="center", ha="left",
        )
        score_w, score_h = 0.11, 0.075
        score_left = 0.12
        add_rounded_rect(
            ax, (score_left, mid_y - score_h / 2), score_w, score_h,
            facecolor=H2H_SCORE_BADGE_FILLS[accent], zorder=2,
        )
        ax.text(
            score_left + score_w / 2, mid_y, score,
            fontsize=20, fontweight="bold", color=accent,
            ha="center", va="center",
        )
        bar_left = 0.27
        bar_width = 0.30
        fill_w = bar_width * min(pct, 100) / 100
        add_rounded_rect(
            ax, (bar_left, mid_y - 0.03), bar_width, 0.06,
            facecolor="#e9ecef", corner_radius=0.008, zorder=2,
        )
        if fill_w > 0:
            add_rounded_rect(
                ax, (bar_left, mid_y - 0.03), fill_w, 0.06,
                facecolor=accent, corner_radius=0.008, zorder=3,
            )
        draw_score_dots(ax, 0.66, mid_y + 0.02, pct, accent)
        ax.text(
            0.92, mid_y,
            f"{pct:.1f}%",
            fontsize=17, fontweight="bold", color="#212529",
            ha="right", va="center",
        )
        y -= card_h + 0.01


def generate_head_to_head_infographic(
    data: HeadToHeadData,
    output_path: Path,
) -> Path:
    """Build and save the head-to-head forecast PNG."""
    score_rows = max(len(data.top_scores), 1)
    fig_h = 10.0 + score_rows * 0.85
    fig = plt.figure(figsize=(12, fig_h), facecolor=PAGE_BG)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.03)
    _draw_header(fig.add_axes([0.05, 0.80, 0.9, 0.17]), fig, data)
    _draw_outcome_panel(fig.add_axes([0.05, 0.58, 0.9, 0.20]), data)
    _draw_goals_panel(fig.add_axes([0.05, 0.37, 0.9, 0.19]), data)
    scores_bottom = max(0.05, 0.37 - score_rows * 0.06)
    _draw_top_scores_panel(
        fig.add_axes([0.05, scores_bottom, 0.9, 0.35 - scores_bottom]),
        data,
    )
    return save_figure(fig, output_path)


def run_head_to_head_generation(
    team_a: str,
    team_b: str,
    n_simulations: int = N_SIMULATIONS,
    output_path: Path | None = None,
) -> Path:
    """Simulate, render and save one head-to-head infographic."""
    print(
        f"Uruchamiam {n_simulations:,} symulacji dla meczu "
        f"{team_a} vs {team_b} (k={K_FACTOR}, lambda_base={LAMBDA_BASE})..."
    )
    data = run_head_to_head_simulation(team_a, team_b, n_simulations=n_simulations)
    print(
        f"  Wygrana {data.team_a}: {data.team_a_win_pct:.1f}%  |  "
        f"Remis: {data.draw_pct:.1f}%  |  "
        f"Wygrana {data.team_b}: {data.team_b_win_pct:.1f}%"
    )
    print(f"  Najczestszy wynik: {data.modal_score} ({data.modal_pct:.1f}%)")
    print(
        f"  Obie strzela: {data.btts_yes_pct:.1f}%  |  "
        f"Nie obie strzela: {data.btts_no_pct:.1f}%"
    )
    if output_path is None:
        slug = f"{team_a}_vs_{team_b}".replace(" ", "_")
        output_path = OUTPUT_DIR / "head_to_head" / f"{slug}.png"
    path = generate_head_to_head_infographic(data, output_path)
    print(f"Zapisano: {path}")
    return path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generuj infografike prognozy meczu miedzy dwiema druzynami z grupy",
    )
    parser.add_argument("team_a", help="Pierwsza druzyna (np. Brazylia)")
    parser.add_argument("team_b", help="Druga druzyna (np. Maroko)")
    parser.add_argument(
        "-n", "--simulations", type=int, default=N_SIMULATIONS,
        help="Liczba symulacji (domyslnie 1000)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Sciezka pliku PNG (domyslnie infographic/head_to_head/...)",
    )
    args = parser.parse_args()
    run_head_to_head_generation(
        args.team_a,
        args.team_b,
        n_simulations=args.simulations,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
