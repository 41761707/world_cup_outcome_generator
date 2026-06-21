from wc_logic import Country, Match, get_countries, load_schedule_presets

def main():
    by_name = {c.name: c.elo for c in get_countries("countries.txt")}
    presets = load_schedule_presets("schedule_groups.txt")
    for (home, away), (s1, s2) in presets.items():
        c1 = Country(home, by_name[home])
        c2 = Country(away, by_name[away])
        e1, e2 = c1.elo, c2.elo
        Match(c1, c2).set_fixed_result(s1, s2)
        print(f"{home} {s1}-{s2} {away}")
        print(f"  {home}: {e1} → {c1.elo:.1f} ({c1.elo - e1:+.1f})")
        print(f"  {away}: {e2} → {c2.elo:.1f} ({c2.elo - e2:+.1f})")

if __name__ == "__main__":
    main()