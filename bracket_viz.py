from __future__ import annotations

import html
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from wc_logic import assign_thirds_to_slots, load_knockout_schedule

BASE_DIR = Path(__file__).resolve().parent
KNOCKOUT_SCHEDULE_FILE = BASE_DIR / "schedule_knockout.txt"

GRID_COLS = 8

# Harmonogram meczów MŚ 2026 (data + miasto) — tylko do podtytułu karty
KNOCKOUT_VENUES: dict[str, str] = {
    "R_32_1": "29 czerwca · Foxborough",
    "R_32_2": "30 czerwca · East Rutherford",
    "R_32_3": "28 czerwca · Inglewood",
    "R_32_4": "29 czerwca · Guadalupe",
    "R_32_5": "2 lipca · Toronto",
    "R_32_6": "2 lipca · Inglewood",
    "R_32_7": "1 lipca · Santa Clara",
    "R_32_8": "1 lipca · Seattle",
    "R_32_9": "29 czerwca · Houston",
    "R_32_10": "30 czerwca · Arlington",
    "R_32_11": "30 czerwca · Meksyk",
    "R_32_12": "1 lipca · Atlanta",
    "R_32_13": "3 lipca · Miami Gardens",
    "R_32_14": "3 lipca · Arlington",
    "R_32_15": "2 lipca · Vancouver",
    "R_32_16": "3 lipca · Kansas City",
    "R_16_1": "4 lipca · Filadelfia",
    "R_16_2": "4 lipca · Houston",
    "R_16_3": "6 lipca · Arlington",
    "R_16_4": "6 lipca · Seattle",
    "R_16_5": "5 lipca · East Rutherford",
    "R_16_6": "5 lipca · Meksyk",
    "R_16_7": "7 lipca · Atlanta",
    "R_16_8": "7 lipca · Vancouver",
    "QF_1": "9 lipca · Foxborough",
    "QF_2": "10 lipca · Inglewood",
    "QF_3": "11 lipca · Miami Gardens",
    "QF_4": "11 lipca · Kansas City",
    "SF_1": "14 lipca · Arlington",
    "SF_2": "15 lipca · Atlanta",
    "3RD": "18 lipca · Miami Gardens",
    "F": "19 lipca · East Rutherford",
}

TOP_R32_IDS = [f"R_32_{i}" for i in range(1, 9)]
BOTTOM_R32_IDS = [f"R_32_{i}" for i in range(9, 17)]
TOP_R16_IDS = [f"R_16_{i}" for i in range(1, 5)]
BOTTOM_R16_IDS = [f"R_16_{i}" for i in range(5, 9)]
TOP_QF_IDS = ["QF_1", "QF_2"]
BOTTOM_QF_IDS = ["QF_3", "QF_4"]

ROUND_ROW_SPECS: list[tuple[str, list[str], int]] = [
    ("1/16 finału", TOP_R32_IDS, 1),
    ("1/8 finału", TOP_R16_IDS, 2),
    ("Ćwierćfinały", TOP_QF_IDS, 4),
    ("Półfinał", ["SF_1"], 8),
]
BOTTOM_ROUND_ROW_SPECS: list[tuple[str, list[str], int]] = [
    ("Półfinał", ["SF_2"], 8),
    ("Ćwierćfinały", BOTTOM_QF_IDS, 4),
    ("1/8 finału", BOTTOM_R16_IDS, 2),
    ("1/16 finału", BOTTOM_R32_IDS, 1),
]

BUTTERFLY_CSS = """
<style>
.bfly-wrap {
    overflow-x: auto;
    padding: 8px 4px 16px;
    background: #f1f3f5;
    border-radius: 12px;
}
.bfly {
    min-width: 960px;
    max-width: 1280px;
    margin: 0 auto;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.bfly-grid {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    column-gap: 6px;
    row-gap: 4px;
    align-items: center;
    margin-bottom: 0;
}
.bfly-round-label {
    grid-column: 1 / -1;
    text-align: center;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #868e96;
    padding: 4px 0 2px;
}
.bfly-gap {
    grid-column: 1 / -1;
    height: 4px;
}
.bfly-slot {
    display: flex;
    justify-content: center;
    min-width: 0;
}
.bfly-card {
    border: 1px solid #ced4da;
    border-radius: 8px;
    background: #fff;
    padding: 6px 8px;
    width: 100%;
    max-width: 148px;
    font-size: 0.72rem;
    color: #212529;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.bfly-card.probable {
    border-color: #74c0fc;
    background: #e7f5ff;
}
.bfly-card .venue {
    font-size: 0.58rem;
    color: #868e96;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.bfly-card .venue.pct {
    color: #1971c2;
    font-weight: 600;
}
.bfly-card .team {
    padding: 3px 0;
    border-bottom: 1px solid #e9ecef;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.25;
}
.bfly-card.probable .team {
    border-bottom-color: #a5d8ff;
}
.bfly-card .team:last-child { border-bottom: none; }
.bfly-card .team.winner { font-weight: 700; color: #198754; }
.bfly-card.probable .team.winner { color: #1864ab; }
.bfly-card .score {
    float: right;
    font-weight: 600;
    color: #495057;
    margin-left: 4px;
}
/* Łączniki między rundami */
.bfly-conn-row {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    column-gap: 6px;
    height: 22px;
    margin: 0;
}
.bfly-conn-row.flip {
    transform: scaleY(-1);
}
.bfly-conn {
    position: relative;
    height: 22px;
}
.bfly-conn::before {
    content: "";
    position: absolute;
    top: 0;
    left: 25%;
    right: 25%;
    height: 11px;
    border-left: 2px solid #adb5bd;
    border-right: 2px solid #adb5bd;
    border-bottom: 2px solid #adb5bd;
    box-sizing: border-box;
}
.bfly-conn::after {
    content: "";
    position: absolute;
    top: 11px;
    left: 50%;
    transform: translateX(-50%);
    width: 2px;
    height: 11px;
    background: #adb5bd;
}
.bfly-bridge-row {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    column-gap: 6px;
    height: 18px;
}
.bfly-bridge-row.flip { transform: scaleY(-1); }
.bfly-bridge {
    grid-column: 4 / span 2;
    position: relative;
}
.bfly-bridge::after {
    content: "";
    position: absolute;
    left: 50%;
    top: 0;
    transform: translateX(-50%);
    width: 2px;
    height: 100%;
    background: #adb5bd;
}
/* Środek — finał i mecz o 3. miejsce */
.bfly-center {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    column-gap: 12px;
    align-items: start;
    margin: 4px 0;
    padding: 12px 0;
    border-top: 2px solid #dee2e6;
    border-bottom: 2px solid #dee2e6;
    background: #fff;
}
.bfly-center-block {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
}
.bfly-center-block.third {
    grid-column: 1 / span 3;
}
.bfly-center-block.final {
    grid-column: 6 / span 3;
}
.bfly-center-title {
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 4px 12px;
    border-radius: 20px;
    white-space: nowrap;
}
.bfly-center-title.third {
    color: #862e9c;
    background: #f3d9fa;
    border: 1px solid #cc5de8;
}
.bfly-center-title.final {
    color: #e67700;
    background: #fff3bf;
    border: 1px solid #ffd43b;
}
.bfly-center-block .bfly-card {
    max-width: 168px;
    width: 100%;
}
.bfly-center-block.third .bfly-card {
    border: 2px solid #cc5de8;
    background: #f8f0fc;
}
.bfly-center-block.final .bfly-card {
    border: 2px solid #ffd43b;
    background: #fff9db;
    box-shadow: 0 2px 8px rgba(255, 212, 59, 0.35);
}
.bfly-center-divider {
    grid-column: 4 / span 2;
    display: flex;
    align-items: center;
    justify-content: center;
    align-self: stretch;
    color: #ced4da;
    font-size: 1.4rem;
    font-weight: 300;
}
.bfly-half-bottom { margin-top: 2px; }
</style>
"""


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _grid_span(index: int, span: int) -> str:
    """Return grid-column style for match index in a row."""
    start = index * span + 1
    return f"grid-column: {start} / span {span};"


def _match_card(
    match_id: str,
    team1: str,
    team2: str,
    result: dict[str, Any] | None = None,
    extra_class: str = "",
) -> str:
    """Build a single match card HTML fragment."""
    venue = KNOCKOUT_VENUES.get(match_id, "")
    winner = result.get("winner") if result else None
    score1 = result.get("score1") if result else None
    score2 = result.get("score2") if result else None
    pair_pct = result.get("pair_pct") if result else None
    winner_pct = result.get("winner_pct") if result else None
    is_probable = pair_pct is not None
    cls1 = "winner" if winner and winner == team1 else ""
    cls2 = "winner" if winner and winner == team2 else ""
    if is_probable:
        venue_html = (
            f'<div class="venue pct">Najczęstsza para: {pair_pct:.1f}%</div>'
        )
        s1 = (
            f'<span class="score">{winner_pct:.1f}%</span>'
            if winner == team1 and winner_pct is not None
            else ""
        )
        s2 = (
            f'<span class="score">{winner_pct:.1f}%</span>'
            if winner == team2 and winner_pct is not None
            else ""
        )
    else:
        venue_html = (
            f'<div class="venue">{_escape(venue)}</div>' if venue else ""
        )
        s1 = f'<span class="score">{score1}</span>' if score1 is not None else ""
        s2 = f'<span class="score">{score2}</span>' if score2 is not None else ""
    probable_cls = " probable" if is_probable else ""
    card_cls = f"bfly-card {extra_class}{probable_cls}".strip()
    return (
        f'<div class="{card_cls}">'
        f"{venue_html}"
        f'<div class="team {cls1}">{s1}{_escape(team1)}</div>'
        f'<div class="team {cls2}">{s2}{_escape(team2)}</div>'
        f"</div>"
    )


def _render_connector_row(
    child_span: int,
    num_children: int,
    flip: bool = False,
) -> str:
    """Render bracket merge lines between two rounds."""
    parent_span = child_span * 2
    num_groups = num_children // 2
    flip_cls = " flip" if flip else ""
    parts = [f'<div class="bfly-conn-row{flip_cls}">']
    for idx in range(num_groups):
        start = idx * parent_span + 1
        parts.append(
            f'<div class="bfly-conn" style="grid-column: {start} / span {parent_span};">'
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _render_sf_bridge(flip: bool = False) -> str:
    """Vertical line from semi-final toward the final."""
    flip_cls = " flip" if flip else ""
    return (
        f'<div class="bfly-bridge-row{flip_cls}">'
        '<div class="bfly-bridge"></div>'
        "</div>"
    )


def _render_round_row(
    round_label: str,
    match_ids: list[str],
    col_span: int,
    labels: dict[str, tuple[str, str]],
    results: dict[str, dict[str, Any]],
    gap_before: bool = True,
) -> str:
    """Render one bracket row with evenly spaced matches on an 8-column grid."""
    parts = []
    if gap_before:
        parts.append('<div class="bfly-gap"></div>')
    parts.append(f'<div class="bfly-round-label">{round_label}</div>')
    parts.append('<div class="bfly-grid">')
    for idx, match_id in enumerate(match_ids):
        pair = labels.get(match_id)
        if pair is None:
            continue
        t1, t2 = pair
        span_style = _grid_span(idx, col_span)
        card = _match_card(match_id, t1, t2, results.get(match_id))
        parts.append(
            f'<div class="bfly-slot" style="{span_style}">{card}</div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _render_half(
    round_specs: list[tuple[str, list[str], int]],
    labels: dict[str, tuple[str, str]],
    results: dict[str, dict[str, Any]],
    half_class: str,
) -> str:
    """Render top or bottom bracket half with connectors between rounds."""
    is_top = half_class == "bfly-half-top"
    # Góra: R32→SF w dół. Dół: SF→R32 (półfinał przy środku, 1/16 na dole).
    specs = round_specs
    rows = []
    for idx, (round_label, match_ids, col_span) in enumerate(specs):
        rows.append(
            _render_round_row(
                round_label,
                match_ids,
                col_span,
                labels,
                results,
                gap_before=idx > 0,
            )
        )
        if idx < len(specs) - 1:
            rows.append(
                _render_connector_row(
                    col_span, len(match_ids), flip=not is_top
                )
            )
    return f'<div class="bfly-half {half_class}">' + "".join(rows) + "</div>"


def _render_center(
    labels: dict[str, tuple[str, str]],
    results: dict[str, dict[str, Any]],
) -> str:
    """Render final and third-place matches with distinct styling."""
    final_pair = labels.get("F", ("—", "—"))
    third_pair = labels.get("3RD", ("—", "—"))
    t1f, t2f = final_pair
    t1t, t2t = third_pair
    third_card = _match_card("3RD", t1t, t2t, results.get("3RD"))
    final_card = _match_card("F", t1f, t2f, results.get("F"))
    return (
        '<div class="bfly-center">'
        '<div class="bfly-center-block third">'
        '<div class="bfly-center-title third">Mecz o 3. miejsce</div>'
        f"{third_card}"
        "</div>"
        '<div class="bfly-center-divider">|</div>'
        '<div class="bfly-center-block final">'
        '<div class="bfly-center-title final">Finał</div>'
        f"{final_card}"
        "</div>"
        "</div>"
    )


def build_butterfly_bracket_html(
    labels: dict[str, tuple[str, str]],
    results: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Return full HTML for a center-converging butterfly bracket."""
    resolved = results or {}
    top = _render_half(ROUND_ROW_SPECS, labels, resolved, "bfly-half-top")
    bridge_top = _render_sf_bridge(flip=False)
    center = _render_center(labels, resolved)
    bridge_bottom = _render_sf_bridge(flip=True)
    bottom = _render_half(
        BOTTOM_ROUND_ROW_SPECS, labels, resolved, "bfly-half-bottom"
    )
    return (
        BUTTERFLY_CSS
        + '<div class="bfly-wrap"><div class="bfly">'
        + top
        + bridge_top
        + center
        + bridge_bottom
        + bottom
        + "</div></div>"
    )


def labels_from_knockout_raw(
    knockout_raw: list[tuple[str, Any, Any]],
    fmt_slot: Any,
    resolved: dict[str, tuple[str, str]] | None = None,
) -> dict[str, tuple[str, str]]:
    """Build label map from schedule slots and optional resolved team names."""
    labels: dict[str, tuple[str, str]] = {}
    for match_id, s1, s2 in knockout_raw:
        if resolved and match_id in resolved:
            labels[match_id] = resolved[match_id]
        else:
            labels[match_id] = (fmt_slot(s1), fmt_slot(s2))
    return labels


def results_from_last_bracket(
    knockout_details: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Convert last_bracket knockout details to results dict."""
    out: dict[str, dict[str, Any]] = {}
    for match in knockout_details:
        out[match["match_id"]] = {
            "team1": match["team1"],
            "team2": match["team2"],
            "score1": match["score1"],
            "score2": match["score2"],
            "winner": match["winner"],
        }
    return out


GroupPhaseKey = tuple[tuple[str, tuple[tuple[str, int, int, int], ...]], ...]
TeamStandingRow = tuple[str, int, int, int]


def marginal_group_standings_from_phase(
    phase_counts: dict[GroupPhaseKey, int],
) -> dict[str, dict[str, list[str]]]:
    """Pick the modal table order independently for each group."""
    order_counters: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    for phase, weight in phase_counts.items():
        for gname, rows in phase:
            order = tuple(row[0] for row in rows)
            order_counters[gname][order] += weight
    return {
        gname: {"order": list(counter.most_common(1)[0][0])}
        for gname, counter in order_counters.items()
    }


def marginal_third_place_rows_from_phase(
    phase_counts: dict[GroupPhaseKey, int],
) -> list[tuple[str, str, int, int, int]]:
    """Return the modal third-place team and stats for every group."""
    third_counters: dict[str, Counter[TeamStandingRow]] = defaultdict(Counter)
    for phase, weight in phase_counts.items():
        for gname, rows in phase:
            if len(rows) >= 3:
                third_counters[gname][rows[2]] += weight
    rows: list[tuple[str, str, int, int, int]] = []
    for gname, counter in third_counters.items():
        team, pts, gd, gs = counter.most_common(1)[0][0]
        rows.append((gname, team, pts, gd, gs))
    return rows


def most_probable_group_standings(
    stats: dict[str, Any],
) -> dict[str, dict[str, list[str]]]:
    """Return per-group modal tables aggregated across all simulations."""
    phase_counts = stats.get("group_phase_counts", {})
    if phase_counts:
        standings = marginal_group_standings_from_phase(phase_counts)
    else:
        standings = {}
    for gname, counts in stats.get("group_standings_counts", {}).items():
        if gname in standings:
            continue
        order, _ = max(counts.items(), key=lambda item: item[1])
        standings[gname] = {"order": list(order)}
    return standings


def qualified_thirds_from_standings(
    group_standings: dict[str, dict[str, list[str]]],
    stats: dict[str, Any],
) -> dict[str, str]:
    """Return the eight best third-place teams from per-group modal rows."""
    phase_counts = stats.get("group_phase_counts", {})
    if phase_counts:
        thirds = marginal_third_place_rows_from_phase(phase_counts)
        thirds_sorted = sorted(
            thirds,
            key=lambda row: (row[2], row[3], row[4]),
            reverse=True,
        )
        return {gname: team for gname, team, _, _, _ in thirds_sorted[:8]}
    return most_probable_qualified_thirds_legacy(stats, group_standings)


def most_probable_qualified_thirds_legacy(
    stats: dict[str, Any],
    group_standings: dict[str, dict[str, list[str]]],
) -> dict[str, str]:
    """Fallback when joint group-phase snapshots are unavailable."""
    thirds_counts = stats.get("qualified_thirds_counts", {})
    if thirds_counts:
        best_ranked, _ = max(thirds_counts.items(), key=lambda item: item[1])
        return {gname: team for gname, team in best_ranked}
    team_counts = stats.get("qualified_thirds_team_counts", {})
    if team_counts and group_standings:
        qualified: dict[str, str] = {}
        for team, _ in sorted(team_counts.items(), key=lambda x: -x[1])[:8]:
            for gname, data in group_standings.items():
                order = data["order"]
                if len(order) >= 3 and order[2] == team and gname not in qualified:
                    qualified[gname] = team
                    break
        return qualified
    return {}


def most_common_third_in_match_slot(
    match_id: str,
    groups: tuple[str, ...],
    qualified_thirds: dict[str, str],
    stats: dict[str, Any],
    thirds_assignment: dict[frozenset[str], str],
    *,
    exclude: set[str] | None = None,
) -> str | None:
    """Pick the third-place team most often seen in this bracket slot."""
    blocked = exclude or set()
    candidates = {
        qualified_thirds[group]
        for group in groups
        if group in qualified_thirds
    } - blocked
    if not candidates:
        return None
    pair_counts = stats.get("match_slot_pairs", {}).get(match_id, {})
    team_counts: Counter[str] = Counter()
    for (team1, team2), count in pair_counts.items():
        if team1 in candidates:
            team_counts[team1] += count
        if team2 in candidates:
            team_counts[team2] += count
    if team_counts:
        return team_counts.most_common(1)[0][0]
    key = frozenset(groups)
    group = thirds_assignment.get(key)
    if group and group in qualified_thirds:
        team = qualified_thirds[group]
        if team not in blocked:
            return team
    return next(iter(candidates))


def modal_winner_from_stats(
    feeder_match_id: str,
    stats: dict[str, Any],
) -> str | None:
    """Return the most common winner for a feeder match from raw stats."""
    winners = stats.get("match_slot_winners", {}).get(feeder_match_id, {})
    if not winners:
        return None
    return max(winners.items(), key=lambda item: item[1])[0]


def modal_loser_from_stats(
    feeder_match_id: str,
    stats: dict[str, Any],
) -> str | None:
    """Return the most common loser for a feeder match from raw stats."""
    winner = modal_winner_from_stats(feeder_match_id, stats)
    if winner is None:
        return None
    pair_counts = stats.get("match_slot_pairs", {}).get(feeder_match_id, {})
    for team1, team2 in pair_counts:
        if winner == team1:
            return team2
        if winner == team2:
            return team1
    return None


def modal_pair_from_stats(
    match_id: str,
    stats: dict[str, Any],
    *,
    exclude: set[str] | None = None,
) -> tuple[str, str] | None:
    """Return the most common team pair for one match slot."""
    pair_counts = stats.get("match_slot_pairs", {}).get(match_id, {})
    if not pair_counts:
        return None
    blocked = exclude or set()
    ranked = sorted(pair_counts.items(), key=lambda item: -item[1])
    for (team1, team2), _ in ranked:
        if team1 not in blocked and team2 not in blocked:
            return team1, team2
    return ranked[0][0]


def _record_probable_match(
    match_id: str,
    team1: str,
    team2: str,
    stats: dict[str, Any],
    n: int,
    labels: dict[str, tuple[str, str]],
    results: dict[str, dict[str, Any]],
    match_winners: dict[str, tuple[str, str]],
) -> None:
    """Store one resolved probable-bracket fixture."""
    labels[match_id] = (team1, team2)
    pair_pct, winner, winner_pct = slot_match_stats(
        match_id, team1, team2, stats, n,
    )
    loser = team2 if winner == team1 else team1
    match_winners[match_id] = (winner, loser)
    results[match_id] = {
        "team1": team1,
        "team2": team2,
        "winner": winner,
        "pair_pct": pair_pct,
        "winner_pct": winner_pct,
    }


def resolve_bracket_slot_name(
    match_id: str,
    slot: tuple,
    group_standings: dict[str, dict[str, list[str]]],
    qualified_thirds: dict[str, str],
    match_winners: dict[str, tuple[str, str]],
    stats: dict[str, Any],
    thirds_assignment: dict[frozenset[str], str] | None = None,
    *,
    used_teams: set[str] | None = None,
) -> str | None:
    """Resolve one bracket slot to a team name."""
    blocked = used_teams or set()
    kind = slot[0]
    if kind == "winner":
        result = match_winners.get(slot[1])
        if result:
            return result[0]
        return modal_winner_from_stats(slot[1], stats)
    if kind == "loser":
        result = match_winners.get(slot[1])
        if result:
            return result[1]
        return modal_loser_from_stats(slot[1], stats)
    _, groups, pos = slot
    if len(groups) == 1:
        data = group_standings.get(groups[0])
        if data and pos <= len(data["order"]):
            team = data["order"][pos - 1]
            if team in blocked:
                for candidate in data["order"]:
                    if candidate not in blocked:
                        return candidate
            return team
        return None
    if pos == 3:
        return most_common_third_in_match_slot(
            match_id,
            groups,
            qualified_thirds,
            stats,
            thirds_assignment or {},
            exclude=blocked,
        )
    for group in groups:
        if group in qualified_thirds:
            team = qualified_thirds[group]
            if team not in blocked:
                return team
    return None


def slot_match_stats(
    match_id: str,
    team1: str,
    team2: str,
    stats: dict[str, Any],
    n: int,
) -> tuple[float, str, float]:
    """Return pair frequency and modal winner for a resolved fixture."""
    pair_counts = stats.get("match_slot_pairs", {}).get(match_id, {})
    pair_winners = stats.get("match_slot_pair_winners", {}).get(match_id, {})
    count = pair_counts.get((team1, team2), 0) + pair_counts.get((team2, team1), 0)
    if count > 0:
        wins1 = (
            pair_winners.get((team1, team2), {}).get(team1, 0)
            + pair_winners.get((team2, team1), {}).get(team1, 0)
        )
        wins2 = (
            pair_winners.get((team1, team2), {}).get(team2, 0)
            + pair_winners.get((team2, team1), {}).get(team2, 0)
        )
        winner = team1 if wins1 >= wins2 else team2
        return 100 * count / n, winner, 100 * max(wins1, wins2) / count
    slot_winners = stats.get("match_slot_winners", {}).get(match_id, {})
    wins1 = slot_winners.get(team1, 0)
    wins2 = slot_winners.get(team2, 0)
    total = wins1 + wins2
    if total > 0:
        winner = team1 if wins1 >= wins2 else team2
        return 0.0, winner, 100 * max(wins1, wins2) / total
    return 0.0, team1, 50.0


def probable_bracket_from_stats(
    stats: dict[str, Any],
    knockout_raw: list[tuple[str, tuple, tuple]] | None = None,
) -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, Any]]]:
    """Build a globally consistent probable bracket from Monte Carlo stats."""
    if knockout_raw is None:
        knockout_raw = load_knockout_schedule(str(KNOCKOUT_SCHEDULE_FILE))
    n = stats["n_simulations"]
    group_standings = most_probable_group_standings(stats)
    qualified_thirds = qualified_thirds_from_standings(group_standings, stats)
    thirds_assignment = (
        assign_thirds_to_slots(qualified_thirds, knockout_raw)
        if qualified_thirds
        else {}
    )
    match_winners: dict[str, tuple[str, str]] = {}
    used_r32_teams: set[str] = set()
    labels: dict[str, tuple[str, str]] = {}
    results: dict[str, dict[str, Any]] = {}
    for match_id, slot1, slot2 in knockout_raw:
        blocked = used_r32_teams if match_id.startswith("R_32") else None
        team1 = resolve_bracket_slot_name(
            match_id,
            slot1,
            group_standings,
            qualified_thirds,
            match_winners,
            stats,
            thirds_assignment,
            used_teams=blocked,
        )
        team2 = resolve_bracket_slot_name(
            match_id,
            slot2,
            group_standings,
            qualified_thirds,
            match_winners,
            stats,
            thirds_assignment,
            used_teams=blocked,
        )
        if not team1 or not team2:
            blocked = used_r32_teams if match_id.startswith("R_32") else None
            fallback = modal_pair_from_stats(match_id, stats, exclude=blocked)
            if fallback:
                team1, team2 = fallback
        if not team1 or not team2:
            fallback = modal_pair_from_stats(match_id, stats)
            if fallback:
                team1, team2 = fallback
        if not team1 or not team2:
            continue
        if match_id.startswith("R_32"):
            used_r32_teams.add(team1)
            used_r32_teams.add(team2)
        _record_probable_match(
            match_id, team1, team2, stats, n, labels, results, match_winners,
        )
    for match_id, _, _ in knockout_raw:
        if match_id in labels:
            continue
        fallback = modal_pair_from_stats(match_id, stats)
        if not fallback:
            continue
        team1, team2 = fallback
        _record_probable_match(
            match_id, team1, team2, stats, n, labels, results, match_winners,
        )
    return labels, results
