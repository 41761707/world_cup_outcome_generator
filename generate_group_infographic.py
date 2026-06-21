from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from group_stats import GroupInfographicData, TeamGroupStats, build_group_infographic_data
from infographic_common import (
    GROUPS_OUTPUT_DIR,
    K_FACTOR,
    LAMBDA_BASE,
    N_SIMULATIONS,
    PAGE_BG,
    add_flag,
    add_rounded_rect,
    draw_gradient_band,
    draw_white_card,
    format_preset_notice,
    load_groups_data,
    resolve_fixed_group_results,
    run_simulation,
    save_figure,
    team_name_label,
)

MAX_GROUP_POINTS = 9.0
TEAM_BAR_COLORS = ["#339af0", "#37b24d", "#f59f00", "#7950f2"]
ADVANCE_DIRECT_COLOR = "#40c057"
ADVANCE_THIRD_COLOR = "#339af0"
ELIMINATED_COLOR = "#fa5252"
POSITION_COLORS = ("#ffd43b", "#adb5bd", "#e07b39", "#868e96")
POSITION_LABELS = ("1.", "2.", "3.", "4.")
LABEL_FLAG_X = 0.08
LABEL_TEXT_X = 0.14
BAR_LEFT = 0.40
BAR_WIDTH = 0.42
VALUE_X = BAR_LEFT + BAR_WIDTH + 0.02


def _row_height(base_row_h: float, label: str) -> float:
    """Expand row height when a wrapped label spans two lines."""
    line_count = label.count("\n") + 1
    return base_row_h * (1.15 if line_count > 1 else 1.0)


def _draw_header(ax: plt.Axes, data: GroupInfographicData) -> None:
    """Render gradient header with group identity."""
    draw_gradient_band(ax)
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.10), 0.96, 0.80,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=(1, 1, 1, 0.08), edgecolor=(1, 1, 1, 0.25), linewidth=1.5,
        zorder=1,
    ))
    ax.text(
        0.5, 0.72, f"Grupa {data.group_name}",
        fontsize=34, fontweight="bold", color="white",
        ha="center", va="center", zorder=2,
    )
    ax.text(
        0.5, 0.38, "Prognoza fazy grupowej",
        fontsize=18, color="#c8d6e5", ha="center", va="center", zorder=2,
    )
    ax.text(
        0.97, 0.12,
        f"{data.n_simulations:,} sym.  ·  k={K_FACTOR}  ·  λ={LAMBDA_BASE}",
        fontsize=14, color="#a4b0be", ha="right", zorder=2,
    )


def _draw_points_panel(ax: plt.Axes, teams: tuple[TeamGroupStats, ...]) -> None:
    """Draw expected points bars sorted from strongest to weakest."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_white_card(ax, (0, 0, 1, 1))
    ax.text(
        0.05, 0.94, "Oczekiwana liczba punktów",
        fontsize=18, fontweight="bold", color="#212529", va="top",
    )
    row_h = 0.18
    y = 0.78
    for index, team in enumerate(teams):
        color = TEAM_BAR_COLORS[index % len(TEAM_BAR_COLORS)]
        label, fontsize, line_spacing = team_name_label(team.name)
        row_h = _row_height(0.18, label)
        mid_y = y - row_h / 2
        add_flag(ax, team.name, (LABEL_FLAG_X, mid_y), zoom=0.55)
        ax.text(
            LABEL_TEXT_X, mid_y, label,
            fontsize=fontsize, fontweight="bold", color="#212529",
            va="center", linespacing=line_spacing,
        )
        fill_w = BAR_WIDTH * min(team.expected_points, MAX_GROUP_POINTS) / MAX_GROUP_POINTS
        add_rounded_rect(
            ax, (BAR_LEFT, mid_y - 0.045), BAR_WIDTH, 0.09,
            facecolor="#e9ecef", corner_radius=0.01, zorder=1,
        )
        if fill_w > 0:
            add_rounded_rect(
                ax, (BAR_LEFT, mid_y - 0.045), fill_w, 0.09,
                facecolor=color, corner_radius=0.01, zorder=2,
            )
        ax.text(
            VALUE_X, mid_y,
            f"{team.expected_points:.1f}",
            fontsize=16, fontweight="bold", color=color,
            va="center", ha="left",
        )
        y -= row_h + 0.02


def _draw_advancement_panel(ax: plt.Axes, teams: tuple[TeamGroupStats, ...]) -> None:
    """Draw stacked advancement probability bars."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_white_card(ax, (0, 0, 1, 1))
    ax.text(
        0.05, 0.94, "Szanse na awans",
        fontsize=18, fontweight="bold", color="#212529", va="top",
    )
    legend_y = 0.78
    legend_items = [
        (ADVANCE_DIRECT_COLOR, "Awans z 1./2. miejsca"),
        (ADVANCE_THIRD_COLOR, "Awans jako najlepsza trzecia"),
        (ELIMINATED_COLOR, "Odpadnięcie"),
    ]
    legend_x = 0.05
    for color, label in legend_items:
        add_rounded_rect(
            ax, (legend_x, legend_y), 0.025, 0.035,
            facecolor=color, corner_radius=0.004, zorder=2,
        )
        ax.text(
            legend_x + 0.035, legend_y + 0.017, label,
            fontsize=10, color="#495057", va="center",
        )
        legend_x += 0.31
    y = 0.73
    for team in teams:
        label, fontsize, line_spacing = team_name_label(team.name)
        row_h = _row_height(0.16, label)
        mid_y = y - row_h / 2
        add_flag(ax, team.name, (LABEL_FLAG_X, mid_y), zoom=0.50)
        ax.text(
            LABEL_TEXT_X, mid_y, label,
            fontsize=fontsize, fontweight="bold", color="#212529",
            va="center", linespacing=line_spacing,
        )
        segments = [
            (team.direct_advance_pct, ADVANCE_DIRECT_COLOR),
            (team.third_advance_pct, ADVANCE_THIRD_COLOR),
            (team.eliminated_pct, ELIMINATED_COLOR),
        ]
        cursor = BAR_LEFT
        for value, color in segments:
            fill_w = BAR_WIDTH * min(value, 100) / 100
            if fill_w > 0:
                add_rounded_rect(
                    ax, (cursor, mid_y - 0.04), fill_w, 0.08,
                    facecolor=color, corner_radius=0.006, zorder=2,
                )
            cursor += fill_w
        total_advance = team.direct_advance_pct + team.third_advance_pct
        ax.text(
            VALUE_X, mid_y,
            f"{total_advance:.0f}%",
            fontsize=14, fontweight="bold", color="#212529",
            va="center", ha="left",
        )
        y -= row_h + 0.02


def _draw_position_panel(ax: plt.Axes, teams: tuple[TeamGroupStats, ...]) -> None:
    """Draw position distribution heatmap for all teams."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_white_card(ax, (0, 0, 1, 1))
    ax.text(
        0.05, 0.94, "Rozkład miejsc w grupie",
        fontsize=18, fontweight="bold", color="#212529", va="top",
    )
    col_left = BAR_LEFT
    col_width = 0.10
    for col_index, label in enumerate(POSITION_LABELS):
        ax.text(
            col_left + col_index * col_width + col_width / 2,
            0.86, label,
            fontsize=13, fontweight="bold", color="#868e96",
            ha="center", va="center",
        )
    y = 0.74
    for team in teams:
        label, fontsize, line_spacing = team_name_label(team.name)
        row_h = _row_height(0.16, label)
        mid_y = y - row_h / 2
        add_flag(ax, team.name, (LABEL_FLAG_X, mid_y), zoom=0.48)
        ax.text(
            LABEL_TEXT_X, mid_y, label,
            fontsize=fontsize, fontweight="bold", color="#212529",
            va="center", linespacing=line_spacing,
        )
        for col_index, pct in enumerate(team.position_pcts):
            color = POSITION_COLORS[col_index]
            cell_left = col_left + col_index * col_width
            intensity = 0.25 + 0.75 * min(pct, 100) / 100
            add_rounded_rect(
                ax, (cell_left + 0.01, mid_y - 0.045), col_width - 0.02, 0.09,
                facecolor=color, alpha=intensity, corner_radius=0.01, zorder=2,
            )
            ax.text(
                cell_left + col_width / 2, mid_y,
                f"{pct:.0f}%",
                fontsize=12, fontweight="bold", color="#212529",
                ha="center", va="center",
            )
        y -= row_h + 0.02


def generate_group_infographic(
    data: GroupInfographicData,
    output_path: Path,
) -> Path:
    """Build and save one group-stage infographic PNG."""
    fig = plt.figure(figsize=(12, 16), facecolor=PAGE_BG)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.03)
    _draw_header(fig.add_axes([0.05, 0.90, 0.9, 0.08]), data)
    _draw_points_panel(fig.add_axes([0.05, 0.63, 0.9, 0.25]), data.teams)
    _draw_advancement_panel(fig.add_axes([0.05, 0.33, 0.9, 0.28]), data.teams)
    _draw_position_panel(fig.add_axes([0.05, 0.03, 0.9, 0.28]), data.teams)
    return save_figure(fig, output_path)


def run_group_infographic_generation(
    n_simulations: int = N_SIMULATIONS,
    groups: list[str] | None = None,
    output_dir: Path | None = None,
    use_schedule_presets: bool = True,
) -> list[Path]:
    """Simulate once and export group infographics."""
    fixed_results = resolve_fixed_group_results(
        None,
        use_schedule_presets=use_schedule_presets,
    )
    print(
        f"Uruchamiam {n_simulations:,} symulacji "
        f"(k={K_FACTOR}, lambda_base={LAMBDA_BASE})..."
    )
    print(format_preset_notice(fixed_results))
    stats = run_simulation(
        n=n_simulations,
        use_schedule_presets=use_schedule_presets,
    )
    print("Symulacja zakonczona. Generuje infografiki grupowe...")
    group_members, _ = load_groups_data()
    target_dir = output_dir or GROUPS_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    group_names = sorted(group_members.keys())
    if groups:
        group_names = [name for name in group_names if name in groups]
        missing = set(groups) - set(group_names)
        if missing:
            raise ValueError(f"Nieznane grupy: {', '.join(sorted(missing))}")
    paths: list[Path] = []
    for group_name in group_names:
        teams = group_members[group_name]
        data = build_group_infographic_data(group_name, teams, stats)
        output_path = target_dir / f"Grupa_{group_name}.png"
        paths.append(generate_group_infographic(data, output_path))
        leader = data.teams[0]
        print(
            f"  Grupa {group_name}: lider {leader.name} "
            f"({leader.expected_points:.1f} pkt, "
            f"{leader.direct_advance_pct + leader.third_advance_pct:.0f}% awansu) "
            f"-> {output_path}"
        )
    print(f"\nGotowe — {len(paths)} infografik w folderze {target_dir}")
    return paths


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generuj infografiki fazy grupowej (punkty, awans, miejsca)",
    )
    parser.add_argument(
        "-n", "--simulations", type=int, default=N_SIMULATIONS,
        help="Liczba symulacji (domyslnie 100000)",
    )
    parser.add_argument(
        "--groups", nargs="*", help="Opcjonalnie: tylko wybrane grupy (np. A C)",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Folder wyjsciowy (domyslnie infographic/groups/)",
    )
    parser.add_argument(
        "--ignore-presets",
        action="store_true",
        help="Ignoruj znane wyniki z schedule_groups.txt",
    )
    args = parser.parse_args()
    run_group_infographic_generation(
        n_simulations=args.simulations,
        groups=args.groups,
        output_dir=args.output_dir,
        use_schedule_presets=not args.ignore_presets,
    )


if __name__ == "__main__":
    main()
