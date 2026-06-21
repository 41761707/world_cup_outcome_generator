"""Generate before/after infographic for a fixed group-match result."""

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
    SCHEDULE_GROUPS_FILE,
    add_flag_in_slot,
    add_rounded_rect,
    draw_gradient_band,
    draw_white_card,
    run_simulation,
    save_figure,
    team_name_label,
)
from wc_logic import load_schedule_presets

R32_REACH_STAGES = ("R_32", "R_16", "QF", "SF", "3RD", "F", "Zwycięzca")
STAGE_LABEL = "1/16 finału"
UP_COLOR = "#2f9e44"
DOWN_COLOR = "#e03131"
BEFORE_BAR = "#adb5bd"
AFTER_UP_BAR = "#339af0"
AFTER_DOWN_BAR = "#fa5252"


@dataclass(frozen=True)
class TeamImpact:
    """Advancement probability shift for one team."""

    name: str
    before_pct: float
    after_pct: float

    @property
    def delta_pp(self) -> float:
        return self.after_pct - self.before_pct


@dataclass(frozen=True)
class MatchImpact:
    """Before/after probabilities for both sides of a fixed result."""

    home: str
    away: str
    home_score: int
    away_score: int
    stage_label: str
    n_simulations: int
    home_impact: TeamImpact
    away_impact: TeamImpact


def _match_keys(home: str, away: str) -> tuple[tuple[str, str], tuple[str, str]]:
    return (home, away), (away, home)


def _presets_without_match(
    presets: dict[tuple[str, str], tuple[int, int]],
    home: str,
    away: str,
) -> dict[tuple[str, str], tuple[int, int]] | None:
    """Return schedule presets excluding the target fixture."""
    home_key, away_key = _match_keys(home, away)
    filtered = {
        key: value
        for key, value in presets.items()
        if key not in (home_key, away_key)
    }
    return filtered or None


def _presets_with_match(
    presets: dict[tuple[str, str], tuple[int, int]],
    home: str,
    away: str,
    home_score: int,
    away_score: int,
) -> dict[tuple[str, str], tuple[int, int]]:
    """Return schedule presets with the target fixture forced."""
    merged = dict(presets)
    merged[(home, away)] = (home_score, away_score)
    return merged


def reach_probability(
    stats: dict,
    team: str,
    stage_set: tuple[str, ...] = R32_REACH_STAGES,
) -> float:
    """Return cumulative reach percentage for a team."""
    stages = stats["team_exit_stages"][team]
    n = stats["n_simulations"]
    return 100 * sum(1 for stage in stages if stage in stage_set) / n


def run_match_impact_simulation(
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    n_simulations: int = N_SIMULATIONS,
    lambda_base: float = LAMBDA_BASE,
    k: float = K_FACTOR,
) -> MatchImpact:
    """Run before/after Monte Carlo runs for one fixed group result."""
    presets = load_schedule_presets(str(SCHEDULE_GROUPS_FILE))
    before_fixed = _presets_without_match(presets, home, away)
    after_fixed = _presets_with_match(
        presets, home, away, home_score, away_score,
    )
    before_stats = run_simulation(
        n=n_simulations,
        lambda_base=lambda_base,
        k=k,
        fixed_group_results=before_fixed,
    )
    after_stats = run_simulation(
        n=n_simulations,
        lambda_base=lambda_base,
        k=k,
        fixed_group_results=after_fixed,
    )
    return MatchImpact(
        home=home,
        away=away,
        home_score=home_score,
        away_score=away_score,
        stage_label=STAGE_LABEL,
        n_simulations=n_simulations,
        home_impact=TeamImpact(
            name=home,
            before_pct=reach_probability(before_stats, home),
            after_pct=reach_probability(after_stats, home),
        ),
        away_impact=TeamImpact(
            name=away,
            before_pct=reach_probability(before_stats, away),
            after_pct=reach_probability(after_stats, away),
        ),
    )


def _draw_header(ax: plt.Axes, fig: plt.Figure, impact: MatchImpact) -> None:
    """Render gradient header with match identity."""
    draw_gradient_band(ax)
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.08), 0.96, 0.84,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=(1, 1, 1, 0.08), edgecolor=(1, 1, 1, 0.25), linewidth=1.5,
        zorder=1,
    ))
    ax.text(
        0.5, 0.92, "Wpływ wyniku meczu",
        fontsize=20, fontweight="bold", color="white", ha="center", va="top", zorder=2,
    )
    ax.text(
        0.5, 0.79, "na szanse awansu do",
        fontsize=15, color="#c8d6e5", ha="center", va="top", zorder=2,
    )
    ax.text(
        0.5, 0.65, impact.stage_label,
        fontsize=24, fontweight="bold", color="#ffd43b", ha="center", va="top", zorder=2,
    )
    add_flag_in_slot(ax, fig, impact.home, (0.22, 0.40), 0.085, on_dark=True)
    add_flag_in_slot(ax, fig, impact.away, (0.78, 0.40), 0.085, on_dark=True)
    ax.text(
        0.5, 0.38,
        f"{impact.home_score} : {impact.away_score}",
        fontsize=34, fontweight="bold", color="white", ha="center", va="center", zorder=3,
    )
    home_label, home_fs, home_ls = team_name_label(impact.home, base_fontsize=22)
    away_label, away_fs, away_ls = team_name_label(impact.away, base_fontsize=22)
    name_y = 0.20 if "\n" in home_label or "\n" in away_label else 0.25
    ax.text(
        0.22, name_y, home_label,
        fontsize=home_fs, fontweight="bold", color="white",
        ha="center", va="top", linespacing=home_ls, zorder=2,
    )
    ax.text(
        0.78, name_y, away_label,
        fontsize=away_fs, fontweight="bold", color="white",
        ha="center", va="top", linespacing=away_ls, zorder=2,
    )


def _draw_team_panel(
    ax: plt.Axes,
    fig: plt.Figure,
    team: TeamImpact,
    *,
    accent_after: str,
) -> None:
    """Draw before/after bars and delta annotation for one team."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    card_left, card_width = 0.03, 0.94
    draw_white_card(ax, (card_left, 0.04, card_width, 0.92))
    header_y = 0.86
    flag_x = 0.095
    add_flag_in_slot(ax, fig, team.name, (flag_x, header_y), 0.055)
    label, fontsize, line_spacing = team_name_label(team.name, base_fontsize=20)
    ax.text(
        0.155, header_y, label,
        fontsize=fontsize, fontweight="bold", color="#212529",
        va="center", linespacing=line_spacing,
    )
    delta = team.delta_pp
    delta_color = UP_COLOR if delta >= 0 else DOWN_COLOR
    arrow = "▲" if delta >= 0 else "▼"
    ax.text(
        card_left + card_width - 0.04, header_y,
        f"{arrow} {delta:+.1f} p.p.",
        fontsize=17, fontweight="bold", color=delta_color,
        ha="right", va="center",
    )
    bar_left = 0.08
    bar_track_width = 0.60
    pct_gap = 0.05
    pct_x = bar_left + bar_track_width + pct_gap
    bar_h = 0.12
    rows = [
        ("Przed meczem", team.before_pct, BEFORE_BAR),
        ("Po meczu", team.after_pct, accent_after),
    ]
    y_positions = [0.5, 0.2]
    for (label, value, color), y in zip(rows, y_positions):
        ax.text(
            bar_left, y + bar_h + 0.05, label,
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


def generate_match_impact_infographic(
    impact: MatchImpact,
    output_path: Path,
) -> Path:
    """Build and save the before/after match-impact PNG."""
    fig = plt.figure(figsize=(12, 11), facecolor=PAGE_BG)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.04)
    _draw_header(fig.add_axes([0.05, 0.72, 0.9, 0.26]), fig, impact)
    _draw_team_panel(
        fig.add_axes([0.05, 0.39, 0.9, 0.30]),
        fig,
        impact.home_impact,
        accent_after=AFTER_UP_BAR if impact.home_impact.delta_pp >= 0 else AFTER_DOWN_BAR,
    )
    _draw_team_panel(
        fig.add_axes([0.05, 0.05, 0.9, 0.30]),
        fig,
        impact.away_impact,
        accent_after=AFTER_UP_BAR if impact.away_impact.delta_pp >= 0 else AFTER_DOWN_BAR,
    )
    return save_figure(fig, output_path)


def run_match_impact_generation(
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    n_simulations: int = N_SIMULATIONS,
    output_path: Path | None = None,
    home_before: float | None = None,
    home_after: float | None = None,
    away_before: float | None = None,
    away_after: float | None = None,
) -> Path:
    """Simulate, render and save one match-impact infographic."""
    manual_values = (home_before, home_after, away_before, away_after)
    if any(value is not None for value in manual_values):
        if any(value is None for value in manual_values):
            raise ValueError(
                "Pass all four overrides together: "
                "--home-before, --home-after, --away-before, --away-after"
            )
        impact = MatchImpact(
            home=home,
            away=away,
            home_score=home_score,
            away_score=away_score,
            stage_label=STAGE_LABEL,
            n_simulations=n_simulations,
            home_impact=TeamImpact(home, home_before, home_after),
            away_impact=TeamImpact(away, away_before, away_after),
        )
        print("Uzyto recznie podanych wartosci (bez symulacji).")
    else:
        print(
            f"Uruchamiam {n_simulations:,} symulacji przed/po meczu "
            f"{home} {home_score}:{away_score} {away} "
            f"(k={K_FACTOR}, lambda_base={LAMBDA_BASE})..."
        )
        impact = run_match_impact_simulation(
            home, away, home_score, away_score, n_simulations=n_simulations,
        )
    for team in (impact.home_impact, impact.away_impact):
        print(
            f"  {team.name}: {team.before_pct:.1f}% -> {team.after_pct:.1f}% "
            f"({team.delta_pp:+.1f} p.p.)"
        )
    if output_path is None:
        slug = f"{home}_vs_{away}_{home_score}-{away_score}".replace(" ", "_")
        output_path = OUTPUT_DIR / "match_impact" / f"{slug}.png"
    path = generate_match_impact_infographic(impact, output_path)
    print(f"Zapisano: {path}")
    return path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generuj infografikę wpływu wyniku meczu na awans do 1/16",
    )
    parser.add_argument("home", help="Gospodarz (np. USA)")
    parser.add_argument("away", help="Gość (np. Paragwaj)")
    parser.add_argument("home_score", type=int, help="Bramki gospodarza")
    parser.add_argument("away_score", type=int, help="Bramki gościa")
    parser.add_argument(
        "-n", "--simulations", type=int, default=N_SIMULATIONS,
        help="Liczba symulacji na scenariusz (domyślnie 1000)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Ścieżka pliku PNG (domyślnie infographic/match_impact/...)",
    )
    parser.add_argument("--home-before", type=float, default=None)
    parser.add_argument("--home-after", type=float, default=None)
    parser.add_argument("--away-before", type=float, default=None)
    parser.add_argument("--away-after", type=float, default=None)
    args = parser.parse_args()
    run_match_impact_generation(
        args.home,
        args.away,
        args.home_score,
        args.away_score,
        n_simulations=args.simulations,
        output_path=args.output,
        home_before=args.home_before,
        home_after=args.home_after,
        away_before=args.away_before,
        away_after=args.away_after,
    )


if __name__ == "__main__":
    main()
