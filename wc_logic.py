"""Core World Cup tournament simulation logic."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np


class Country:
    """National team with an ELO rating."""

    def __init__(self, name: str, elo: int) -> None:
        self.name = name
        self.elo = elo

    def evaluate_const(self, goal_diff: int) -> float:
        """Return ELO K-factor adjusted for goal difference."""
        constant = 64.0
        if goal_diff == 2:
            constant = constant * 3 / 2
        elif goal_diff == 3:
            constant = constant * 7 / 4
        elif goal_diff > 3:
            constant = constant * (7 / 4 + (goal_diff - 3) / 8)
        return constant

    def recalc_elo(self, opponent_elo: float, outcome: str, goal_diff: int) -> None:
        """Update ELO after a match result."""
        k_factor = self.evaluate_const(goal_diff)
        expected_score = 1 / (1 + 10 ** ((opponent_elo - self.elo) / 400))
        if outcome == "win":
            score = 1
        elif outcome == "draw":
            score = 0.5
        else:
            score = 0
        self.elo += k_factor * (score - expected_score)


class Group:
    """World Cup group with teams and a match schedule."""

    def __init__(
        self,
        name: str,
        countries: list[Country],
        schedule: list[tuple[Country, Country]] | None,
    ) -> None:
        self.name = name
        self.countries = countries
        self.schedule = schedule
        self.standings: list[tuple[Country, dict[str, int]]] = []

    def simulate_group_stage(
        self,
        lambda_base: float,
        k: float,
        fixed_results: dict[tuple[str, str], tuple[int, int]] | None = None,
    ) -> list[Match]:
        """Play all group matches and return Match objects."""
        matches = []
        for country1, country2 in self.schedule:
            match = Match(country1, country2)
            key = (country1.name, country2.name)
            key_rev = (country2.name, country1.name)
            if fixed_results and key in fixed_results:
                match.set_fixed_result(*fixed_results[key])
            elif fixed_results and key_rev in fixed_results:
                s1, s2 = fixed_results[key_rev]
                match.set_fixed_result(s2, s1)
            else:
                match.simulate_match(lambda_base, k)
            matches.append(match)
        return matches

    def calculate_standings(
        self, matches: list[Match]
    ) -> list[tuple[Country, dict[str, int]]]:
        """Compute final group table from played matches."""
        name_to_country = {c.name: c for c in self.countries}
        standings = {
            c.name: {"points": 0, "goal_diff": 0, "goals_scored": 0}
            for c in self.countries
        }
        for match in matches:
            if match.outcome == "1":
                standings[match.country1.name]["points"] += 3
            elif match.outcome == "2":
                standings[match.country2.name]["points"] += 3
            else:
                standings[match.country1.name]["points"] += 1
                standings[match.country2.name]["points"] += 1
            standings[match.country1.name]["goal_diff"] += match.score1 - match.score2
            standings[match.country2.name]["goal_diff"] += match.score2 - match.score1
            standings[match.country1.name]["goals_scored"] += match.score1
            standings[match.country2.name]["goals_scored"] += match.score2
        sorted_items = sorted(
            standings.items(),
            key=lambda x: (
                x[1]["points"],
                x[1]["goal_diff"],
                x[1]["goals_scored"],
            ),
            reverse=True,
        )
        self.standings = [
            (name_to_country[name], stats) for name, stats in sorted_items
        ]
        return self.standings


class Match:
    """A single football match between two countries."""

    def __init__(self, country1: Country, country2: Country) -> None:
        self.country1 = country1
        self.country2 = country2
        self.outcome = "0"
        self.lambda1 = 0.0
        self.lambda2 = 0.0
        self.elo_prev1 = 0.0
        self.elo_prev2 = 0.0
        self.elo_after1 = 0.0
        self.elo_after2 = 0.0
        self.score1 = 0
        self.score2 = 0

    def set_fixed_result(self, score1: int, score2: int) -> None:
        """Apply a predetermined score and update ELO."""
        self.score1 = score1
        self.score2 = score2
        goal_diff = abs(score1 - score2)
        if score1 > score2:
            self.outcome = "1"
            self.country1.recalc_elo(self.country2.elo, "win", goal_diff)
            self.country2.recalc_elo(self.country1.elo, "loss", goal_diff)
        elif score2 > score1:
            self.outcome = "2"
            self.country1.recalc_elo(self.country2.elo, "loss", goal_diff)
            self.country2.recalc_elo(self.country1.elo, "win", goal_diff)
        else:
            self.outcome = "X"
            self.country1.recalc_elo(self.country2.elo, "draw", goal_diff)
            self.country2.recalc_elo(self.country1.elo, "draw", goal_diff)
        print(self)

    def simulate_match(self, lambda_base: float, k: float) -> None:
        """Simulate match goals via Poisson model and update ELO."""
        diff = (self.country1.elo - self.country2.elo) / 400.0
        self.lambda1 = lambda_base * np.exp(diff * k * np.log(10))
        self.lambda2 = lambda_base * np.exp(-diff * k * np.log(10))
        self.score1 = int(np.random.poisson(self.lambda1))
        self.score2 = int(np.random.poisson(self.lambda2))
        goal_diff = abs(self.score1 - self.score2)
        if self.score1 > self.score2:
            self.outcome = "1"
            self.country1.recalc_elo(self.country2.elo, "win", goal_diff)
            self.country2.recalc_elo(self.country1.elo, "loss", goal_diff)
        elif self.score2 > self.score1:
            self.outcome = "2"
            self.country1.recalc_elo(self.country2.elo, "loss", goal_diff)
            self.country2.recalc_elo(self.country1.elo, "win", goal_diff)
        else:
            self.outcome = "X"
            self.country1.recalc_elo(self.country2.elo, "draw", goal_diff)
            self.country2.recalc_elo(self.country1.elo, "draw", goal_diff)
        print(self)

    def __repr__(self) -> str:
        return (
            f"{self.country1.name} {self.score1} - "
            f"{self.score2} {self.country2.name}"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def get_match_info(self) -> dict[str, Any]:
        """Return serializable match details."""
        return {
            "country1": self.country1.name,
            "country2": self.country2.name,
            "score1": self.score1,
            "score2": self.score2,
            "outcome": self.outcome,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "elo_prev1": self.elo_prev1,
            "elo_prev2": self.elo_prev2,
            "elo_after1": self.elo_after1,
            "elo_after2": self.elo_after2,
        }


class KnockoutMatch:
    """Knockout-stage match resolved from bracket slots."""

    def __init__(self, match_id: str, slot1: tuple, slot2: tuple) -> None:
        self.match_id = match_id
        self.slot1 = slot1
        self.slot2 = slot2
        self.country1: Country | None = None
        self.country2: Country | None = None
        self.score1: int | None = None
        self.score2: int | None = None
        self.winner: Country | None = None
        self.loser: Country | None = None

    def set_fixed_result(
        self,
        score1: int,
        score2: int,
        penalties_winner: str | None = None,
    ) -> None:
        """Apply a predetermined knockout result, with optional penalties."""
        self.score1 = score1
        self.score2 = score2
        goal_diff = abs(score1 - score2)
        if score1 > score2:
            self.winner = self.country1
            self.loser = self.country2
            self.country1.recalc_elo(self.country2.elo, "win", goal_diff)
            self.country2.recalc_elo(self.country1.elo, "loss", goal_diff)
        elif score2 > score1:
            self.winner = self.country2
            self.loser = self.country1
            self.country2.recalc_elo(self.country1.elo, "win", goal_diff)
            self.country1.recalc_elo(self.country2.elo, "loss", goal_diff)
        else:
            # Remis — wynik po karnych
            if penalties_winner == "slot1":
                self.winner = self.country1
                self.loser = self.country2
            else:
                self.winner = self.country2
                self.loser = self.country1
            self.country1.recalc_elo(self.country2.elo, "draw", 0)
            self.country2.recalc_elo(self.country1.elo, "draw", 0)

    def simulate(self, lambda_base: float, k: float) -> None:
        """Simulate knockout match; penalties on draw."""
        diff = (self.country1.elo - self.country2.elo) / 400.0
        lambda1 = lambda_base * np.exp(diff * k * np.log(10))
        lambda2 = lambda_base * np.exp(-diff * k * np.log(10))
        self.score1 = int(np.random.poisson(lambda1))
        self.score2 = int(np.random.poisson(lambda2))
        goal_diff = abs(self.score1 - self.score2)
        if self.score1 > self.score2:
            self.winner = self.country1
            self.loser = self.country2
            self.country1.recalc_elo(self.country2.elo, "win", goal_diff)
            self.country2.recalc_elo(self.country1.elo, "loss", goal_diff)
        elif self.score2 > self.score1:
            self.winner = self.country2
            self.loser = self.country1
            self.country2.recalc_elo(self.country1.elo, "win", goal_diff)
            self.country1.recalc_elo(self.country2.elo, "loss", goal_diff)
        else:
            # Rzuty karne — 50/50, ELO aktualizowane jak remis
            if np.random.random() < 0.5:
                self.winner = self.country1
                self.loser = self.country2
            else:
                self.winner = self.country2
                self.loser = self.country1
            self.country1.recalc_elo(self.country2.elo, "draw", 0)
            self.country2.recalc_elo(self.country1.elo, "draw", 0)

    def __str__(self) -> str:
        score_str = f"{self.score1} - {self.score2}"
        pen_str = (
            f" (karne: {self.winner.name})" if self.score1 == self.score2 else ""
        )
        return (
            f"{self.country1.name} {score_str} "
            f"{self.country2.name}{pen_str}"
        )


def get_countries(countries_file: str) -> list[Country]:
    """Load countries and ELO ratings from a text file."""
    with open(countries_file, "r", encoding="utf-8") as f:
        countries = []
        for line in f:
            name, elo = line.strip().split(":")
            countries.append(Country(name, int(elo)))
        return countries


def create_groups(groups_file: str, countries: list[Country]) -> list[Group]:
    """Build Group objects from a groups definition file."""
    with open(groups_file, "r", encoding="utf-8") as f:
        groups = []
        for line in f:
            parts = line.strip().split(":")
            group_name = parts[0]
            country_names = [n.strip() for n in parts[1].split(",")]
            group_countries = [
                country for country in countries if country.name in country_names
            ]
            groups.append(Group(group_name, group_countries, schedule=None))
        return groups


def is_score(s: str) -> bool:
    """Return True if string looks like a score (e.g. '2-1')."""
    parts = s.split("-")
    if len(parts) != 2:
        return False
    try:
        int(parts[0])
        int(parts[1])
        return True
    except ValueError:
        return False


def parse_group_fixture_line(
    pair_str: str,
) -> tuple[str, str, tuple[int, int] | None]:
    """Parse 'Home - Away' or 'Home - Away 2-0' into teams and optional score."""
    pair_str = pair_str.strip()
    preset: tuple[int, int] | None = None
    parts = pair_str.rsplit(None, 1)
    if len(parts) == 2 and is_score(parts[1]):
        pair_str = parts[0].strip()
        s1, s2 = parts[1].split("-")
        preset = (int(s1), int(s2))
    home, away = [n.strip() for n in pair_str.split(" - ", 1)]
    return home, away, preset


def load_group_schedule_pairs(
    schedule_file: str,
) -> dict[str, list[tuple[str, str]]]:
    """Load group fixtures as team-name pairs, stripping embedded scores."""
    schedules: dict[str, list[tuple[str, str]]] = {}
    with open(schedule_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            group_name, pair_str = line.split(":", 1)
            home, away, _ = parse_group_fixture_line(pair_str)
            schedules.setdefault(group_name.strip(), []).append((home, away))
    return schedules


def load_schedule(
    schedule_file: str,
    countries_by_name: dict[str, Country],
) -> dict[str, list[tuple[Country, Country]]]:
    """Load group-stage fixtures, optionally stripping preset scores."""
    schedules: dict[str, list[tuple[Country, Country]]] = {}
    with open(schedule_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            group_name, pair_str = line.split(":", 1)
            home, away, _ = parse_group_fixture_line(pair_str)
            pair = (countries_by_name[home], countries_by_name[away])
            schedules.setdefault(group_name.strip(), []).append(pair)
    return schedules


def load_schedule_presets(
    schedule_file: str,
) -> dict[tuple[str, str], tuple[int, int]]:
    """Extract preset scores embedded in a schedule file."""
    presets: dict[tuple[str, str], tuple[int, int]] = {}
    with open(schedule_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, pair_str = line.split(":", 1)
            home, away, preset = parse_group_fixture_line(pair_str)
            if preset is not None:
                presets[(home, away)] = preset
    return presets


def parse_slot(slot_str: str) -> tuple:
    """Parse a bracket slot string into a typed tuple."""
    slot_str = slot_str.strip()
    if slot_str.startswith("W_") or slot_str.startswith("L_"):
        rest = slot_str[2:]
        try:
            int(rest)
        except ValueError:
            kind = "winner" if slot_str.startswith("W_") else "loser"
            return (kind, rest)
    last_underscore = slot_str.rfind("_")
    pos = int(slot_str[last_underscore + 1:])
    groups = slot_str[:last_underscore].split("/")
    return ("group_pos", groups, pos)


def load_knockout_schedule(
    schedule_file: str,
) -> list[tuple[str, tuple, tuple]]:
    """Load knockout bracket fixture list from file."""
    raw = []
    with open(schedule_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match_id, pair_str = line.split(":", 1)
            match_id = match_id.strip()
            slot1_str, slot2_str = pair_str.split(" - ", 1)
            raw.append((match_id, parse_slot(slot1_str), parse_slot(slot2_str)))
    return raw


def match_id_to_stage(match_id: str) -> str:
    """Map a match id to its knockout stage code."""
    if match_id.startswith("R_32"):
        return "R_32"
    if match_id.startswith("R_16"):
        return "R_16"
    if match_id.startswith("QF"):
        return "QF"
    if match_id.startswith("SF"):
        return "SF"
    return match_id


def simulate_tournament_once(
    original_elos: dict[str, int],
    countries_by_name: dict[str, Country],
    groups: list[Group],
    knockout_raw: list[tuple[str, tuple, tuple]],
    lambda_base: float,
    k: float,
    fixed_group_results: dict[tuple[str, str], tuple[int, int]] | None = None,
    fixed_knockout_results: dict[str, tuple[int, int, str | None]] | None = None,
) -> dict[str, Any]:
    """Run one full tournament simulation and return detailed results."""
    for name, elo in original_elos.items():
        countries_by_name[name].elo = elo
    for group in groups:
        group.standings = []
    group_standings_by_name: dict[str, list] = {}
    group_match_pairs: list[tuple[str, str]] = []
    group_match_details: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        matches = group.simulate_group_stage(
            lambda_base, k, fixed_results=fixed_group_results
        )
        standings = group.calculate_standings(matches)
        group_standings_by_name[group.name] = standings
        for match in matches:
            group_match_pairs.append((match.country1.name, match.country2.name))
        group_match_details[group.name] = [
            {
                "team1": match.country1.name,
                "team2": match.country2.name,
                "score1": match.score1,
                "score2": match.score2,
            }
            for match in matches
        ]
    exit_stage = {name: "Grupa" for name in countries_by_name}
    qualified_thirds = select_qualified_thirds(groups)
    thirds_assignment = assign_thirds_to_slots(qualified_thirds, knockout_raw)
    knockout_matches = [
        KnockoutMatch(mid, s1, s2) for mid, s1, s2 in knockout_raw
    ]
    knockout_match_pairs: list[tuple[str, str, str]] = []
    knockout_match_details: list[dict[str, Any]] = []
    match_results: dict[str, KnockoutMatch] = {}
    for km in knockout_matches:
        km.country1 = resolve_slot(
            km.slot1,
            group_standings_by_name,
            qualified_thirds,
            match_results,
            thirds_assignment,
        )
        km.country2 = resolve_slot(
            km.slot2,
            group_standings_by_name,
            qualified_thirds,
            match_results,
            thirds_assignment,
        )
        stage = match_id_to_stage(km.match_id)
        if fixed_knockout_results and km.match_id in fixed_knockout_results:
            s1, s2, pen = fixed_knockout_results[km.match_id]
            km.set_fixed_result(s1, s2, pen)
        else:
            km.simulate(lambda_base, k)
        match_results[km.match_id] = km
        knockout_match_pairs.append(
            (km.country1.name, km.country2.name, stage)
        )
        knockout_match_details.append({
            "match_id": km.match_id,
            "team1": km.country1.name,
            "team2": km.country2.name,
            "score1": km.score1,
            "score2": km.score2,
            "winner": km.winner.name,
            "stage": stage,
            "penalties": km.score1 == km.score2,
        })
        exit_stage[km.loser.name] = stage
    if "F" in match_results:
        exit_stage[match_results["F"].winner.name] = "Zwycięzca"
    if "3RD" in match_results:
        exit_stage[match_results["3RD"].winner.name] = "3RD"
    group_standings_snapshot = {
        gname: [(c.name, dict(stats)) for c, stats in standings]
        for gname, standings in group_standings_by_name.items()
    }
    thirds_all = [
        (gname, standings[2][0].name, standings[2][1])
        for gname, standings in group_standings_by_name.items()
        if len(standings) > 2
    ]
    thirds_sorted = sorted(
        thirds_all,
        key=lambda x: (x[2]["points"], x[2]["goal_diff"], x[2]["goals_scored"]),
        reverse=True,
    )
    qualified_thirds_ranked = tuple(
        (gname, name) for gname, name, _ in thirds_sorted[:8]
    )
    return {
        "exit_stages": exit_stage,
        "group_match_pairs": group_match_pairs,
        "knockout_match_pairs": knockout_match_pairs,
        "knockout_match_details": knockout_match_details,
        "group_standings": group_standings_snapshot,
        "group_match_details": group_match_details,
        "qualified_thirds_ranked": qualified_thirds_ranked,
    }


def select_qualified_thirds(
    groups: list[Group], n: int = 8
) -> dict[str, Country]:
    """Return the best n third-place teams keyed by group name."""
    thirds = [
        (group.name, group.standings[2][0], group.standings[2][1])
        for group in groups
    ]
    thirds_sorted = sorted(
        thirds,
        key=lambda x: (x[2]["points"], x[2]["goal_diff"], x[2]["goals_scored"]),
        reverse=True,
    )
    return {group_name: country for group_name, country, _ in thirds_sorted[:n]}


def assign_thirds_to_slots(
    qualified_thirds: dict[str, Country],
    knockout_raw: list[tuple[str, tuple, tuple]],
) -> dict[frozenset[str], str]:
    """Assign qualified third-place teams to bracket slots via matching."""
    slot_sets = []
    seen: set[frozenset[str]] = set()
    for _, s1, s2 in knockout_raw:
        for slot in (s1, s2):
            if slot[0] == "group_pos" and len(slot[1]) > 1 and slot[2] == 3:
                key = frozenset(slot[1])
                if key not in seen:
                    seen.add(key)
                    slot_sets.append(key)

    qualified_set = set(qualified_thirds.keys())
    group_to_slots = {
        group: [idx for idx, slot in enumerate(slot_sets) if group in slot]
        for group in qualified_set
    }
    match_slot: dict[int, str] = {}

    def augment(group: str, visited: set[int]) -> bool:
        for slot_idx in group_to_slots.get(group, []):
            if slot_idx in visited:
                continue
            visited.add(slot_idx)
            prev = match_slot.get(slot_idx)
            if prev is None or augment(prev, visited):
                match_slot[slot_idx] = group
                return True
        return False

    for group in qualified_set:
        augment(group, set())

    return {slot_sets[i]: group for i, group in match_slot.items()}


def resolve_slot(
    slot: tuple,
    group_standings_by_name: dict[str, list],
    qualified_thirds: dict[str, Country],
    match_results: dict[str, KnockoutMatch],
    thirds_assignment: dict[frozenset[str], str] | None = None,
) -> Country:
    """Resolve a bracket slot to a Country instance."""
    kind = slot[0]
    if kind == "winner":
        return match_results[slot[1]].winner
    if kind == "loser":
        return match_results[slot[1]].loser
    _, groups, pos = slot
    if len(groups) == 1:
        return group_standings_by_name[groups[0]][pos - 1][0]
    # Slot trzeciego miejsca — używamy bipartytowego przypisania
    if thirds_assignment is not None:
        key = frozenset(groups)
        group = thirds_assignment.get(key)
        if group and group in qualified_thirds:
            return qualified_thirds[group]
    # Fallback (nie powinno być potrzebne)
    for group in groups:
        if group in qualified_thirds:
            return qualified_thirds[group]
    raise ValueError(
        f"No qualifying third-place team found for groups {groups}"
    )


def simulate_knockout_stage(
    knockout_raw: list[tuple[str, tuple, tuple]],
    group_standings_by_name: dict[str, list],
    qualified_thirds: dict[str, Country],
    lambda_base: float,
    k: float,
) -> dict[str, KnockoutMatch]:
    """Simulate knockout stage and print each match."""
    match_results: dict[str, KnockoutMatch] = {}
    thirds_assignment = assign_thirds_to_slots(qualified_thirds, knockout_raw)
    for mid, s1, s2 in knockout_raw:
        km = KnockoutMatch(mid, s1, s2)
        km.country1 = resolve_slot(
            km.slot1,
            group_standings_by_name,
            qualified_thirds,
            match_results,
            thirds_assignment,
        )
        km.country2 = resolve_slot(
            km.slot2,
            group_standings_by_name,
            qualified_thirds,
            match_results,
            thirds_assignment,
        )
        km.simulate(lambda_base, k)
        match_results[km.match_id] = km
        print(f"[{km.match_id}] {km}")
    return match_results


def main() -> None:
    """CLI demo: simulate groups then knockout stage."""
    # Countries with their ELO, https://eloratings.net/
    # TODO: Enable parametrization of those constants
    lambda_base = 1.3
    k = 0.25
    countries_file = sys.argv[1]
    groups_file = sys.argv[2]
    schedule_file = sys.argv[3]
    knockout_file = sys.argv[4]
    countries = get_countries(countries_file)
    countries_by_name = {c.name: c for c in countries}
    groups = create_groups(groups_file, countries)
    schedules = load_schedule(schedule_file, countries_by_name)
    for group in groups:
        group.schedule = schedules.get(group.name)
        if group.schedule is None:
            raise ValueError(f"Brak harmonogramu dla grupy {group.name}")
    group_standings_by_name: dict[str, list] = {}
    for group in groups:
        matches = group.simulate_group_stage(lambda_base, k)
        standings = group.calculate_standings(matches)
        group_standings_by_name[group.name] = standings
        print(f"Group {group.name} Standings:")
        for country, stats in standings:
            print(
                f"{country.name}: {stats['points']} pts, "
                f"GD: {stats['goal_diff']}, GF: {stats['goals_scored']}"
            )
        print()
    qualified_thirds = select_qualified_thirds(groups)
    knockout_raw = load_knockout_schedule(knockout_file)
    print("=== Faza Pucharowa ===")
    simulate_knockout_stage(
        knockout_raw, group_standings_by_name, qualified_thirds, lambda_base, k
    )


if __name__ == "__main__":
    main()
