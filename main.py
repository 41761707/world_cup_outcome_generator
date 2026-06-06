import sys
from collections import defaultdict
from wc_logic import (
    get_countries, create_groups, load_schedule, load_knockout_schedule,
    simulate_tournament_once
)
#TODO: Stałe do konfiga
STAGES_ORDER = ["Grupa", "R_32", "R_16", "QF", "SF", "3RD", "F", "Zwycięzca"]

def run_monte_carlo(countries_file, groups_file, schedule_file, knockout_file, n=1000, lambda_base=1.3, k=0.25, fixed_group_results=None):
    countries = get_countries(countries_file)
    countries_by_name = {c.name: c for c in countries}
    original_elos = {c.name: c.elo for c in countries}
    groups = create_groups(groups_file, countries)
    schedules = load_schedule(schedule_file, countries_by_name)
    for group in groups:
        group.schedule = schedules.get(group.name)
        if group.schedule is None:
            raise ValueError(f"Brak harmonogramu dla grupy {group.name}")
    knockout_raw = load_knockout_schedule(knockout_file)
    team_exit_stages = defaultdict(list)
    group_meeting_counts = defaultdict(int)
    knockout_meeting_counts = defaultdict(int)
    last_bracket = None
    for i in range(n):
        if (i + 1) % 100 == 0:
            print(f"Symulacja {i + 1}/{n}...")
        result = simulate_tournament_once(
            original_elos, countries_by_name, groups, knockout_raw, lambda_base, k,
            fixed_group_results=fixed_group_results
        )
        for country, stage in result["exit_stages"].items():
            team_exit_stages[country].append(stage)
        for t1, t2 in result["group_match_pairs"]:
            group_meeting_counts[frozenset({t1, t2})] += 1
        for t1, t2, stage in result["knockout_match_pairs"]:
            knockout_meeting_counts[(frozenset({t1, t2}), stage)] += 1
        last_bracket = {
            "knockout": result["knockout_match_details"],
            "groups": result["group_standings"],
            "group_matches": result["group_match_details"],
        }
    return {
        "team_exit_stages": dict(team_exit_stages),
        "group_meeting_counts": dict(group_meeting_counts),
        "knockout_meeting_counts": dict(knockout_meeting_counts),
        "n_simulations": n,
        "last_bracket": last_bracket,
    }

def print_stats(stats):
    #Stylizacja output przez AI bo mi sie nie chce
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
    for (pair, stage), count in sorted(stats["knockout_meeting_counts"].items(), key=lambda x: -x[1])[:20]:
        teams = list(pair)
        print(f"  [{stage}] {teams[0]} vs {teams[1]}: {count}/{n} ({100 * count / n:.1f}%)")

def print_team_stats(stats, team_name):
    n = stats["n_simulations"]
    team_exit = stats["team_exit_stages"]
    if team_name not in team_exit:
        print(f"Nie znaleziono druzyny: {team_name}")
        return
    stages = team_exit[team_name]
    print(f"\n--- {team_name} ({n} symulacji) ---")
    for s in STAGES_ORDER:
        count = stages.count(s)
        if count > 0:
            bar = "#" * (count * 40 // n)
            print(f"  {s:12s}: {count:5d} ({100 * count / n:5.1f}%)  {bar}")

def main():
    if len(sys.argv) < 5:
        print("Uzycie: python main.py <countries> <groups> <schedule_groups> <schedule_knockout> [n_symulacji] [druzyna]")
        sys.exit(1)
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    stats = run_monte_carlo(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], n=n)
    print_stats(stats)
    if len(sys.argv) > 6:
        print_team_stats(stats, sys.argv[6])

if __name__ == "__main__":
    main()
