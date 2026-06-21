"""Monte Carlo tournament simulation entry point."""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Any

from wc_logic import (
    create_groups,
    get_countries,
    load_knockout_schedule,
    load_schedule,
    simulate_tournament_once,
)

# TODO: Stałe do konfiga
STAGES_ORDER = [
    "Grupa", "R_32", "R_16", "QF", "SF", "3RD", "F", "Zwycięzca",
]


def _accumulate_simulation_stats(
    result: dict[str, Any],
    team_exit_stages: dict[str, list[str]],
    group_meeting_counts: dict[frozenset[str], int],
    knockout_meeting_counts: dict[tuple[frozenset[str], str], int],
    knockout_meeting_wins: dict[tuple[frozenset[str], str], dict[str, int]],
    match_slot_pairs: dict[str, dict[tuple[str, str], int]],
    match_slot_winners: dict[str, dict[str, int]],
    match_slot_pair_winners: dict[str, dict[tuple[str, str], dict[str, int]]],
    group_standings_counts: dict[str, dict[tuple[str, ...], int]],
    qualified_thirds_counts: dict[tuple[tuple[str, str], ...], int],
    qualified_thirds_team_counts: dict[str, int],
    group_match_score_counts: dict[tuple[str, str], dict[tuple[int, int], int]],
    group_phase_counts: dict[tuple[tuple[str, tuple[tuple[str, int, int, int], ...]], ...], int],
) -> None:
    """Merge one tournament run into aggregate counters."""
    for country, stage in result["exit_stages"].items():
        team_exit_stages[country].append(stage)
    for t1, t2 in result["group_match_pairs"]:
        group_meeting_counts[frozenset({t1, t2})] += 1
    for matches in result["group_match_details"].values():
        for match in matches:
            key = (match["team1"], match["team2"])
            score_key = (match["score1"], match["score2"])
            group_match_score_counts[key][score_key] += 1
    for t1, t2, stage in result["knockout_match_pairs"]:
        knockout_meeting_counts[(frozenset({t1, t2}), stage)] += 1
    for detail in result["knockout_match_details"]:
        key = (frozenset({detail["team1"], detail["team2"]}), detail["stage"])
        knockout_meeting_wins[key][detail["winner"]] += 1
    for detail in result["knockout_match_details"]:
        mid = detail["match_id"]
        pair = (detail["team1"], detail["team2"])
        match_slot_pairs[mid][pair] += 1
        match_slot_winners[mid][detail["winner"]] += 1
        match_slot_pair_winners[mid][pair][detail["winner"]] += 1
    for gname, standings in result["group_standings"].items():
        order = tuple(name for name, _ in standings)
        group_standings_counts[gname][order] += 1
    qualified_thirds_counts[result["qualified_thirds_ranked"]] += 1
    for _, team_name in result["qualified_thirds_ranked"]:
        qualified_thirds_team_counts[team_name] += 1
    phase_key = tuple(
        (
            gname,
            tuple(
                (
                    name,
                    stats["points"],
                    stats["goal_diff"],
                    stats["goals_scored"],
                )
                for name, stats in standings
            ),
        )
        for gname, standings in sorted(result["group_standings"].items())
    )
    group_phase_counts[phase_key] += 1


def run_monte_carlo(
    countries_file: str,
    groups_file: str,
    schedule_file: str,
    knockout_file: str,
    n: int = 1000,
    lambda_base: float = 1.3,
    k: float = 0.25,
    fixed_group_results: dict[tuple[str, str], tuple[int, int]] | None = None,
    fixed_knockout_results: dict[str, tuple[int, int, str | None]] | None = None,
) -> dict[str, Any]:
    """Run n tournament simulations and return aggregated statistics."""
    countries = get_countries(countries_file)
    countries_by_name = {c.name: c for c in countries}
    original_elos = {c.name: c.elo for c in countries}
    groups = create_groups(groups_file, countries)
    schedules = load_schedule(schedule_file, countries_by_name)
    for group in groups:
        group.schedule = schedules.get(group.name)
        if group.schedule is None:
            raise ValueError(f"Missing schedule for group {group.name}")
    knockout_raw = load_knockout_schedule(knockout_file)
    team_exit_stages: dict[str, list[str]] = defaultdict(list)
    group_meeting_counts: dict[frozenset[str], int] = defaultdict(int)
    knockout_meeting_counts: dict[tuple[frozenset[str], str], int] = defaultdict(int)
    knockout_meeting_wins: dict[tuple[frozenset[str], str], dict[str, int]] = (
        defaultdict(lambda: defaultdict(int))
    )
    match_slot_pairs: dict[str, dict[tuple[str, str], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    match_slot_winners: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    match_slot_pair_winners: dict[str, dict[tuple[str, str], dict[str, int]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )
    group_standings_counts: dict[str, dict[tuple[str, ...], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    qualified_thirds_counts: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)
    qualified_thirds_team_counts: dict[str, int] = defaultdict(int)
    group_match_score_counts: dict[tuple[str, str], dict[tuple[int, int], int]] = (
        defaultdict(lambda: defaultdict(int))
    )
    group_phase_counts: dict[
        tuple[tuple[str, tuple[tuple[str, int, int, int], ...]], ...],
        int,
    ] = defaultdict(int)
    last_bracket = None
    # TODO: Wizualizacja w tqdm
    for i in range(n):
        if (i + 1) % 100 == 0:
            print(f"Symulacja {i + 1}/{n}...")
        result = simulate_tournament_once(
            original_elos,
            countries_by_name,
            groups,
            knockout_raw,
            lambda_base,
            k,
            fixed_group_results=fixed_group_results,
            fixed_knockout_results=fixed_knockout_results,
        )
        _accumulate_simulation_stats(
            result,
            team_exit_stages,
            group_meeting_counts,
            knockout_meeting_counts,
            knockout_meeting_wins,
            match_slot_pairs,
            match_slot_winners,
            match_slot_pair_winners,
            group_standings_counts,
            qualified_thirds_counts,
            qualified_thirds_team_counts,
            group_match_score_counts,
            group_phase_counts,
        )
        last_bracket = {
            "knockout": result["knockout_match_details"],
            "groups": result["group_standings"],
            "group_matches": result["group_match_details"],
        }
    return {
        "team_exit_stages": dict(team_exit_stages),
        "group_meeting_counts": dict(group_meeting_counts),
        "knockout_meeting_counts": dict(knockout_meeting_counts),
        "knockout_meeting_wins": {
            key: dict(wins) for key, wins in knockout_meeting_wins.items()
        },
        "match_slot_pairs": {
            mid: dict(counts) for mid, counts in match_slot_pairs.items()
        },
        "match_slot_winners": {
            mid: dict(counts) for mid, counts in match_slot_winners.items()
        },
        "match_slot_pair_winners": {
            mid: {pair: dict(wins) for pair, wins in pair_wins.items()}
            for mid, pair_wins in match_slot_pair_winners.items()
        },
        "group_standings_counts": {
            gname: dict(counts) for gname, counts in group_standings_counts.items()
        },
        "qualified_thirds_counts": dict(qualified_thirds_counts),
        "qualified_thirds_team_counts": dict(qualified_thirds_team_counts),
        "group_match_score_counts": {
            key: dict(scores) for key, scores in group_match_score_counts.items()
        },
        "group_phase_counts": dict(group_phase_counts),
        "n_simulations": n,
        "last_bracket": last_bracket
    }


def print_stats(stats: dict[str, Any]) -> None:
    """Print summary statistics from a Monte Carlo run."""
    # Stylizacja output przez AI bo mi sie nie chce
    n = stats["n_simulations"]
    team_exit = stats["team_exit_stages"]
    print(f"\n=== Statystyki z {n} symulacji ===\n")
    print("--- Najczestsze finaly (lacznie F + Zwyciezca) ---")
    final_counts = {
        team: sum(1 for s in stages if s in ("F", "Zwycięzca"))
        for team, stages in team_exit.items()
    }
    for team, count in sorted(final_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {team}: {count}/{n} ({100 * count / n:.1f}%)")
    print("\n--- Najczestsi zwyciezcy turnieju ---")
    winner_counts = {
        team: sum(1 for s in stages if s == "Zwycięzca")
        for team, stages in team_exit.items()
    }
    for team, count in sorted(winner_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {team}: {count}/{n} ({100 * count / n:.1f}%)")
    print("\n--- Rozklad etapow dla kazdej druzyny ---")
    for team in sorted(team_exit.keys()):
        stages = team_exit[team]
        dist = {s: stages.count(s) for s in STAGES_ORDER if stages.count(s) > 0}
        dist_str = ", ".join(f"{s}: {c}" for s, c in dist.items())
        print(f"  {team}: {dist_str}")
    print("\n--- Najczestsze spotkania w fazie pucharowej (top 20) ---")
    top_ko = sorted(
        stats["knockout_meeting_counts"].items(), key=lambda x: -x[1]
    )[:20]
    for (pair, stage), count in top_ko:
        teams = list(pair)
        pct = 100 * count / n
        print(f"  [{stage}] {teams[0]} vs {teams[1]}: {count}/{n} ({pct:.1f}%)")


def print_team_stats(stats: dict[str, Any], team_name: str) -> None:
    """Print per-team stage distribution for one team."""
    n = stats["n_simulations"]
    team_exit = stats["team_exit_stages"]
    if team_name not in team_exit:
        print(f"Nie znaleziono druzyny: {team_name}")
        return
    stages = team_exit[team_name]
    print(f"\n--- {team_name} ({n} symulacji) ---")
    for stage in STAGES_ORDER:
        count = stages.count(stage)
        if count > 0:
            bar = "#" * (count * 40 // n)
            pct = 100 * count / n
            print(f"  {stage:12s}: {count:5d} ({pct:5.1f}%)  {bar}")


def main() -> None:
    """CLI entry point for batch simulations."""
    if len(sys.argv) < 5:
        print(
            "Uzycie: python main.py <countries> <groups> "
            "<schedule_groups> <schedule_knockout> [n_symulacji] [druzyna]"
        )
        sys.exit(1)
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    stats = run_monte_carlo(
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], n=n
    )
    print_stats(stats)
    if len(sys.argv) > 6:
        print_team_stats(stats, sys.argv[6])


if __name__ == "__main__":
    main()
