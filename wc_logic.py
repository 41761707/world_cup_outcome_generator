import numpy as np
import sys

class Country:
    def __init__(self, name, elo):
        self.name = name
        self.elo = elo

    def evaluate_const(self, goal_diff):
        constant = 64
        if goal_diff == 2:
            constant = constant * 3/2
        elif goal_diff == 3:
            constant = constant * 7/4
        elif goal_diff > 3:
            constant = constant * (7/4 + (goal_diff-3)/8)
        return constant

    def recalc_elo(self, opponent_elo, outcome, goal_diff):
        k = self.evaluate_const(goal_diff)
        expected_score = 1 / (1 + 10 ** ((opponent_elo - self.elo) / 400))
        if outcome == "win":
            score = 1
        elif outcome == "draw":
            score = 0.5
        else:
            score = 0
        self.elo += k * (score - expected_score)
    
class Group:
    def __init__(self, name, countries, schedule):
        self.name = name
        self.countries = countries
        # schedule: list of rounds, each round is list of (Country, Country) pairs
        self.schedule = schedule
        self.standings = []

    def simulate_group_stage(self, lambda_base, k, fixed_results=None):
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
    
    def calculate_standings(self, matches):
        name_to_country = {c.name: c for c in self.countries}
        standings = {c.name: {"points": 0, "goal_diff": 0, "goals_scored": 0} for c in self.countries}
        for match in matches:
            if match.outcome == "1":
                standings[match.country1.name]["points"] += 3
            elif match.outcome == "2":
                standings[match.country2.name]["points"] += 3
            else:
                standings[match.country1.name]["points"] += 1
                standings[match.country2.name]["points"] += 1
            standings[match.country1.name]["goal_diff"] += (match.score1 - match.score2)
            standings[match.country2.name]["goal_diff"] += (match.score2 - match.score1)
            standings[match.country1.name]["goals_scored"] += match.score1
            standings[match.country2.name]["goals_scored"] += match.score2
        sorted_items = sorted(standings.items(), key=lambda x: (x[1]["points"], x[1]["goal_diff"], x[1]["goals_scored"]), reverse=True)
        self.standings = [(name_to_country[name], stats) for name, stats in sorted_items]
        return self.standings

class Match:
    def __init__(self, country1, country2):
        self.country1 = country1
        self.country2 = country2
        self.outcome = "0"
        self.lambda1 = 0
        self.lambda2 = 0
        self.elo_prev1 = 0
        self.elo_prev2 = 0
        self.elo_after1 = 0
        self.elo_after2 = 0
    
    def set_fixed_result(self, score1, score2):
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

    def simulate_match(self, lambda_base, k):
        diff = (self.country1.elo - self.country2.elo) / 400.0
        self.lambda1 = lambda_base * np.exp(diff * k * np.log(10))
        self.lambda2 = lambda_base * np.exp(-diff * k * np.log(10))
        self.score1 = np.random.poisson(self.lambda1)
        self.score2 = np.random.poisson(self.lambda2)
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

    def __repr__(self):
        return f"{self.country1.name} {self.score1} - {self.score2} {self.country2.name}"
    
    def __str__(self):
        return f"{self.country1.name} {self.score1} - {self.score2} {self.country2.name}"
    
    def get_match_info(self):
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
            "elo_after2": self.elo_after2
        }

class KnockoutMatch:
    def __init__(self, match_id, slot1, slot2):
        self.match_id = match_id
        self.slot1 = slot1
        self.slot2 = slot2
        self.country1 = None
        self.country2 = None
        self.score1 = None
        self.score2 = None
        self.winner = None
        self.loser = None

    def simulate(self, lambda_base, k):
        diff = (self.country1.elo - self.country2.elo) / 400.0
        lambda1 = lambda_base * np.exp(diff * k * np.log(10))
        lambda2 = lambda_base * np.exp(-diff * k * np.log(10))
        self.score1 = np.random.poisson(lambda1)
        self.score2 = np.random.poisson(lambda2)
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

    def __str__(self):
        score_str = f"{self.score1} - {self.score2}"
        pen_str = f" (karne: {self.winner.name})" if self.score1 == self.score2 else ""
        return f"{self.country1.name} {score_str} {self.country2.name}{pen_str}"

def get_countries(countries_file):
    with open(countries_file, 'r', encoding='utf-8') as f:
        countries = []
        for line in f:
            #Brazil:1900
            name, elo = line.strip().split(':')
            countries.append(Country(name, int(elo)))
        return countries
    
def create_groups(groups_file, countries):
    with open(groups_file, 'r', encoding='utf-8') as f:
        groups = []
        for line in f:
            #A:Brazil,Argentina,France,Germany
            parts = line.strip().split(':')
            group_name = parts[0]
            country_names = [n.strip() for n in parts[1].split(',')]
            group_countries = [country for country in countries if country.name in country_names]
            groups.append(Group(group_name, group_countries, schedule=None))
        return groups

def load_schedule(schedule_file, countries_by_name):
    schedules = {}
    with open(schedule_file, 'r', encoding='utf-8') as f:
        for line in f:
            #A: Brazil - Argentina
            #A: Brazil - Argentina 2-1   (opcjonalny wynik)
            line = line.strip()
            if not line:
                continue
            group_name, pair_str = line.split(':', 1)
            group_name = group_name.strip()
            pair_str = pair_str.strip()
            # Odetnij opcjonalny wynik na końcu (np. "2-1")
            parts = pair_str.rsplit(None, 1)
            if len(parts) == 2 and _is_score(parts[1]):
                pair_str = parts[0].strip()
            home, away = [n.strip() for n in pair_str.split(' - ', 1)]
            pair = (countries_by_name[home], countries_by_name[away])
            schedules.setdefault(group_name, []).append(pair)
    return schedules


def _is_score(s):
    """Sprawdza czy string ma format X-Y (liczby całkowite)."""
    parts = s.split('-')
    if len(parts) != 2:
        return False
    try:
        int(parts[0])
        int(parts[1])
        return True
    except ValueError:
        return False


def load_schedule_presets(schedule_file):
    """Zwraca dict {(home, away): (score1, score2)} dla meczów z wpisanym wynikiem w pliku harmonogramu."""
    presets = {}
    with open(schedule_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, pair_str = line.split(':', 1)
            pair_str = pair_str.strip()
            parts = pair_str.rsplit(None, 1)
            if len(parts) == 2 and _is_score(parts[1]):
                s1, s2 = parts[1].split('-')
                home, away = [n.strip() for n in parts[0].split(' - ', 1)]
                presets[(home, away)] = (int(s1), int(s2))
    return presets
    
def parse_slot(slot_str):
    slot_str = slot_str.strip()
    if slot_str.startswith('W_') or slot_str.startswith('L_'):
        rest = slot_str[2:]
        try:
            int(rest)
            # sama liczba po prefiksie → W/L to nazwa grupy, np. L_2
        except ValueError:
            # nie jest liczbą → referencja do wyniku meczu, np. W_R_32_1
            kind = 'winner' if slot_str.startswith('W_') else 'loser'
            return (kind, rest)
    last_underscore = slot_str.rfind('_')
    pos = int(slot_str[last_underscore + 1:])
    groups = slot_str[:last_underscore].split('/')
    return ('group_pos', groups, pos)

def load_knockout_schedule(schedule_file):
    raw = []
    with open(schedule_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match_id, pair_str = line.split(':', 1)
            match_id = match_id.strip()
            slot1_str, slot2_str = pair_str.split(' - ', 1)
            raw.append((match_id, parse_slot(slot1_str), parse_slot(slot2_str)))
    return raw

def match_id_to_stage(match_id):
    if match_id.startswith('R_32'):
        return 'R_32'
    if match_id.startswith('R_16'):
        return 'R_16'
    if match_id.startswith('QF'):
        return 'QF'
    if match_id.startswith('SF'):
        return 'SF'
    return match_id  # '3RD', 'F'

def simulate_tournament_once(original_elos, countries_by_name, groups, knockout_raw, lambda_base, k, fixed_group_results=None):
    """Rozgrywa jeden turniej. Zwraca słownik z wynikami."""
    for name, elo in original_elos.items():
        countries_by_name[name].elo = elo
    for group in groups:
        group.standings = []
    group_standings_by_name = {}
    group_match_pairs = []
    group_match_details: dict = {}
    for group in groups:
        matches = group.simulate_group_stage(lambda_base, k, fixed_results=fixed_group_results)
        standings = group.calculate_standings(matches)
        group_standings_by_name[group.name] = standings
        for m in matches:
            group_match_pairs.append((m.country1.name, m.country2.name))
        group_match_details[group.name] = [
            {
                'team1': m.country1.name,
                'team2': m.country2.name,
                'score1': m.score1,
                'score2': m.score2,
            }
            for m in matches
        ]
    exit_stage = {name: 'Grupa' for name in countries_by_name}
    qualified_thirds = select_qualified_thirds(groups)
    knockout_matches = [KnockoutMatch(mid, s1, s2) for mid, s1, s2 in knockout_raw]
    knockout_match_pairs = []
    knockout_match_details = []
    match_results = {}
    for km in knockout_matches:
        km.country1 = resolve_slot(km.slot1, group_standings_by_name, qualified_thirds, match_results)
        km.country2 = resolve_slot(km.slot2, group_standings_by_name, qualified_thirds, match_results)
        stage = match_id_to_stage(km.match_id)
        km.simulate(lambda_base, k)
        match_results[km.match_id] = km
        knockout_match_pairs.append((km.country1.name, km.country2.name, stage))
        knockout_match_details.append({
            'match_id': km.match_id,
            'team1': km.country1.name,
            'team2': km.country2.name,
            'score1': km.score1,
            'score2': km.score2,
            'winner': km.winner.name,
            'stage': stage,
            'penalties': km.score1 == km.score2,
        })
        exit_stage[km.loser.name] = stage
    if 'F' in match_results:
        exit_stage[match_results['F'].winner.name] = 'Zwycięzca'
    if '3RD' in match_results:
        exit_stage[match_results['3RD'].winner.name] = '3RD'
    group_standings_snapshot = {
        gname: [(c.name, dict(stats)) for c, stats in standings]
        for gname, standings in group_standings_by_name.items()
    }
    return {
        'exit_stages': exit_stage,
        'group_match_pairs': group_match_pairs,
        'knockout_match_pairs': knockout_match_pairs,
        'knockout_match_details': knockout_match_details,
        'group_standings': group_standings_snapshot,
        'group_match_details': group_match_details,
    }

def select_qualified_thirds(groups, n=8):
    thirds = [(group.name, group.standings[2][0], group.standings[2][1]) for group in groups]
    thirds_sorted = sorted(thirds, key=lambda x: (x[2]["points"], x[2]["goal_diff"], x[2]["goals_scored"]), reverse=True)
    return {group_name: country for group_name, country, _ in thirds_sorted[:n]}

def resolve_slot(slot, group_standings_by_name, qualified_thirds, match_results):
    kind = slot[0]
    if kind == 'winner':
        return match_results[slot[1]].winner
    if kind == 'loser':
        return match_results[slot[1]].loser
    _, groups, pos = slot
    if len(groups) == 1:
        return group_standings_by_name[groups[0]][pos - 1][0]
    # Slot trzeciego miejsca — szukamy awansującej drużyny spośród podanych grup
    for g in groups:
        if g in qualified_thirds:
            return qualified_thirds[g]
    raise ValueError(f"Nie znaleziono awansującej trzeciej drużyny dla grup {groups}")

def simulate_knockout_stage(knockout_raw, group_standings_by_name, qualified_thirds, lambda_base, k):
    match_results = {}
    for mid, s1, s2 in knockout_raw:
        km = KnockoutMatch(mid, s1, s2)
        km.country1 = resolve_slot(km.slot1, group_standings_by_name, qualified_thirds, match_results)
        km.country2 = resolve_slot(km.slot2, group_standings_by_name, qualified_thirds, match_results)
        km.simulate(lambda_base, k)
        match_results[km.match_id] = km
        print(f"[{km.match_id}] {km}")
    return match_results

def main():
    #Countries with their ELO, https://eloratings.net/
    #TODO: Enable parametrization of those constants
    lambda_base = 1.3 #average number of goals per team during world cup
    k = 0.25 #scaling factor
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
    group_standings_by_name = {}
    for group in groups:
        matches = group.simulate_group_stage(lambda_base, k)
        standings = group.calculate_standings(matches)
        group_standings_by_name[group.name] = standings
        print(f"Group {group.name} Standings:")
        for country, stats in standings:
            print(f"{country.name}: {stats['points']} pts, GD: {stats['goal_diff']}, GF: {stats['goals_scored']}")
        print()
    qualified_thirds = select_qualified_thirds(groups)
    knockout_raw = load_knockout_schedule(knockout_file)
    print("=== Faza Pucharowa ===")
    simulate_knockout_stage(knockout_raw, group_standings_by_name, qualified_thirds, lambda_base, k)



if __name__ == '__main__':
    main()