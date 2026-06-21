"""Pure data aggregation for group-stage infographic charts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infographic_common import expected_match_points


@dataclass(frozen=True)
class TeamGroupStats:
    """Simulation summary for one team inside a group."""

    name: str
    expected_points: float
    position_pcts: tuple[float, float, float, float]
    direct_advance_pct: float
    third_advance_pct: float
    eliminated_pct: float


@dataclass(frozen=True)
class GroupInfographicData:
    """Aggregated group-stage metrics for one group."""

    group_name: str
    n_simulations: int
    teams: tuple[TeamGroupStats, ...]


def position_distribution(
    group_name: str,
    team: str,
    stats: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Return finish-place percentages (1st through 4th)."""
    n = stats["n_simulations"]
    counts = stats["group_standings_counts"].get(group_name, {})
    place_counts = [0, 0, 0, 0]
    for order, count in counts.items():
        for index, name in enumerate(order):
            if name == team:
                place_counts[index] += count
                break
    return tuple(100 * count / n for count in place_counts)


def advancement_rates(
    team: str,
    position_pcts: tuple[float, float, float, float],
    stats: dict[str, Any],
) -> tuple[float, float, float]:
    """Return direct, best-third and elimination percentages."""
    direct = position_pcts[0] + position_pcts[1]
    n = stats["n_simulations"]
    third = 100 * stats.get("qualified_thirds_team_counts", {}).get(team, 0) / n
    eliminated = max(0.0, 100.0 - direct - third)
    return direct, third, eliminated


def expected_group_points(
    team: str,
    opponents: list[str],
    stats: dict[str, Any],
) -> float:
    """Return expected group-stage points across all rivals."""
    return sum(
        expected_match_points(stats, team, opponent)
        for opponent in opponents
    )


def build_team_group_stats(
    group_name: str,
    team: str,
    opponents: list[str],
    stats: dict[str, Any],
) -> TeamGroupStats:
    """Build one team's group-stage summary from simulation stats."""
    positions = position_distribution(group_name, team, stats)
    direct, third, eliminated = advancement_rates(team, positions, stats)
    return TeamGroupStats(
        name=team,
        expected_points=expected_group_points(team, opponents, stats),
        position_pcts=positions,
        direct_advance_pct=direct,
        third_advance_pct=third,
        eliminated_pct=eliminated,
    )


def build_group_infographic_data(
    group_name: str,
    teams: list[str],
    stats: dict[str, Any],
) -> GroupInfographicData:
    """Build sorted group infographic payload for all teams."""
    rows = [
        build_team_group_stats(
            group_name,
            team,
            [rival for rival in teams if rival != team],
            stats,
        )
        for team in teams
    ]
    rows.sort(key=lambda row: (-row.expected_points, row.name))
    return GroupInfographicData(
        group_name=group_name,
        n_simulations=stats["n_simulations"],
        teams=tuple(rows),
    )
