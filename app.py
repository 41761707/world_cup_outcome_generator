"""Streamlit UI for World Cup Monte Carlo simulations."""

from __future__ import annotations

import contextlib
import io
import os
import sys
from collections import defaultdict
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from bracket_viz import (
    build_butterfly_bracket_html,
    labels_from_knockout_raw,
    probable_bracket_from_stats,
    results_from_last_bracket,
)
from main import run_monte_carlo
from wc_logic import (
    assign_thirds_to_slots,
    get_countries,
    load_knockout_schedule,
    load_schedule_presets,
)

# TODO: stałe do konfiga
COUNTRIES_FILE = os.path.join(BASE_DIR, "countries.txt")
GROUPS_FILE = os.path.join(BASE_DIR, "groups.txt")
SCHEDULE_GROUPS_FILE = os.path.join(BASE_DIR, "schedule_groups.txt")
SCHEDULE_KNOCKOUT_FILE = os.path.join(BASE_DIR, "schedule_knockout.txt")
STAGES_ORDER = ["Grupa", "R_32", "R_16", "QF", "3RD", "F", "Zwycięzca"]
STAGES_LABELS = {
    "Grupa": "Faza grupowa",
    "R_32": "1/16 finału",
    "R_16": "1/8 finału",
    "QF": "Ćwierćfinał",
    "SF": "Półfinał",
    "3RD": "3. miejsce",
    "F": "Finał",
    "Zwycięzca": "Zwycięzca",
}
ROUND_ORDER = ["R_32", "R_16", "QF", "SF", "3RD", "F"]
ROUND_LABELS = {
    "R_32": "1/16 finału",
    "R_16": "1/8 finału",
    "QF": "Ćwierćfinały",
    "SF": "Półfinały",
    "3RD": "Mecz o 3. miejsce",
    "F": "Finał",
}


@st.cache_data
def load_initial_data() -> tuple[
    dict[str, list[tuple[str, int]]],
    dict[str, list[tuple[str, str]]],
    list[tuple[str, Any, Any]],
    dict[tuple[str, str], tuple[int, int]],
]:
    """Load countries, groups, schedules and knockout fixture data."""
    countries = get_countries(COUNTRIES_FILE)
    elo_map = {c.name: c.elo for c in countries}
    groups_data: dict[str, list[tuple[str, int]]] = {}
    with open(GROUPS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            group_name, rest = line.split(":", 1)
            names = [n.strip() for n in rest.split(",")]
            groups_data[group_name] = [(n, elo_map.get(n, 0)) for n in names]
    group_schedule: dict[str, list[tuple[str, str]]] = {}
    with open(SCHEDULE_GROUPS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            group_name, pair_str = line.split(":", 1)
            home, away = [n.strip() for n in pair_str.split(" - ", 1)]
            group_schedule.setdefault(group_name.strip(), []).append((home, away))
    knockout_raw = load_knockout_schedule(SCHEDULE_KNOCKOUT_FILE)
    schedule_presets = load_schedule_presets(SCHEDULE_GROUPS_FILE)
    return groups_data, group_schedule, knockout_raw, schedule_presets


def _fmt_slot(slot: tuple) -> str:
    kind = slot[0]
    if kind == "winner":
        return f"Wyg. {slot[1]}"
    if kind == "loser":
        return f"Prz. {slot[1]}"
    _, groups, pos = slot
    suffix = {1: "1.", 2: "2.", 3: "3."}.get(pos, f"{pos}.")
    label = f"Gr. {groups[0]}" if len(groups) == 1 else f"Gr. {'/'.join(groups)}"
    return f"{suffix} {label}"


def _group_by_round(
    knockout_raw: list[tuple[str, Any, Any]],
) -> dict[str, list[tuple[str, Any, Any]]]:
    rounds: dict[str, list] = defaultdict(list)
    for match_id, s1, s2 in knockout_raw:
        if match_id.startswith("R_32"):
            rounds["R_32"].append((match_id, s1, s2))
        elif match_id.startswith("R_16"):
            rounds["R_16"].append((match_id, s1, s2))
        elif match_id.startswith("QF"):
            rounds["QF"].append((match_id, s1, s2))
        elif match_id.startswith("SF"):
            rounds["SF"].append((match_id, s1, s2))
        elif match_id == "3RD":
            rounds["3RD"].append((match_id, s1, s2))
        elif match_id == "F":
            rounds["F"].append((match_id, s1, s2))
    return rounds


def _calc_group_standings(
    fixed_group_results: dict[tuple[str, str], tuple[int, int]] | None,
    group_schedule: dict[str, list[tuple[str, str]]],
    groups_data: dict[str, list[tuple[str, int]]],
) -> dict[str, dict[str, Any]]:
    """Compute group tables for groups with complete fixed results."""
    fr = fixed_group_results or {}
    group_standings: dict = {}
    for gname, matches in group_schedule.items():
        if not all((h, a) in fr or (a, h) in fr for h, a in matches):
            continue
        teams = [name for name, _ in groups_data[gname]]
        stats = {t: {"pts": 0, "gd": 0, "gs": 0} for t in teams}
        for home, away in matches:
            if (home, away) in fr:
                s1, s2 = fr[(home, away)]
            else:
                s2, s1 = fr[(away, home)]
            stats[home]["gd"] += s1 - s2
            stats[away]["gd"] += s2 - s1
            stats[home]["gs"] += s1
            stats[away]["gs"] += s2
            if s1 > s2:
                stats[home]["pts"] += 3
            elif s2 > s1:
                stats[away]["pts"] += 3
            else:
                stats[home]["pts"] += 1
                stats[away]["pts"] += 1
        order = sorted(teams, key=lambda t: (stats[t]["pts"], stats[t]["gd"], stats[t]["gs"]), reverse=True)
        group_standings[gname] = {"order": order, "stats": stats}
    return group_standings


def _calc_qualified_thirds(
    group_standings: dict[str, dict[str, Any]],
    groups_data: dict[str, list[tuple[str, int]]],
) -> dict[str, str]:
    """Return top eight third-place teams when all groups are known."""
    if len(group_standings) != len(groups_data):
        return {}
    thirds = [
        (gname, data["order"][2], data["stats"][data["order"][2]])
        for gname, data in group_standings.items()
        if len(data["order"]) >= 3
    ]
    thirds_sorted = sorted(thirds, key=lambda x: (x[2]["pts"], x[2]["gd"], x[2]["gs"]), reverse=True)
    return {gname: team for gname, team, _ in thirds_sorted[:8]}


def _compute_resolved_slots(
    fixed_group_results: dict[tuple[str, str], tuple[int, int]] | None,
    group_schedule: dict[str, list[tuple[str, str]]],
    groups_data: dict[str, list[tuple[str, int]]],
    knockout_raw: list[tuple[str, Any, Any]],
    ko_raw_values: dict[str, tuple[int, int, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, tuple[str, str]]]:
    """Resolve bracket slots from group and knockout inputs.

    ko_raw_values maps match_id to (score1, score2, pen_label).
    Returns (group_standings, qualified_thirds, ko_results).
    """
    group_standings = _calc_group_standings(fixed_group_results, group_schedule, groups_data)
    qualified_thirds = _calc_qualified_thirds(group_standings, groups_data)
    thirds_assignment = assign_thirds_to_slots(qualified_thirds, knockout_raw) if qualified_thirds else {}

    ko_results: dict = {}
    for match_id, s1_slot, s2_slot in knockout_raw:
        t1 = _resolve_single_slot_inner(s1_slot, group_standings, qualified_thirds, ko_results, thirds_assignment)
        t2 = _resolve_single_slot_inner(s2_slot, group_standings, qualified_thirds, ko_results, thirds_assignment)
        if t1 and t2 and match_id in ko_raw_values:
            s1_score, s2_score, pen_label = ko_raw_values[match_id]
            if s1_score > s2_score:
                ko_results[match_id] = (t1, t2)
            elif s2_score > s1_score:
                ko_results[match_id] = (t2, t1)
            else:
                # Remis — pen_label to nazwa drużyny
                if pen_label == t1:
                    ko_results[match_id] = (t1, t2)
                elif pen_label == t2:
                    ko_results[match_id] = (t2, t1)
    return group_standings, qualified_thirds, ko_results


def _resolve_single_slot_inner(
    slot: tuple,
    group_standings: dict[str, dict[str, Any]],
    qualified_thirds: dict[str, str],
    ko_results: dict[str, tuple[str, str]],
    thirds_assignment: dict[frozenset[str], str] | None = None,
) -> str | None:
    """Return team name for a slot when it is deterministically known."""
    kind = slot[0]
    if kind == "winner":
        r = ko_results.get(slot[1])
        return r[0] if r else None
    if kind == "loser":
        r = ko_results.get(slot[1])
        return r[1] if r else None
    _, groups, pos = slot
    if len(groups) == 1:
        data = group_standings.get(groups[0])
        if data and pos <= len(data["order"]):
            return data["order"][pos - 1]
        return None
    # Slot 3. miejsca — używamy bipartytowego przypisania
    if thirds_assignment:
        key = frozenset(groups)
        group = thirds_assignment.get(key)
        if group and group in qualified_thirds:
            return qualified_thirds[group]
    # Fallback
    for g in groups:
        if g in qualified_thirds:
            return qualified_thirds[g]
    return None


# TODO: CSS do osobnego pliku
CARD_CSS = """
<style>
.match-card {
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 10px 12px;
    background: #f8f9fa;
    margin-bottom: 6px;
    font-size: 0.82rem;
    color: #212529 !important;
}
.match-card .mid {
    font-size: 0.68rem;
    color: #888 !important;
    margin-bottom: 6px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.match-card .team {
    padding: 4px 0;
    border-bottom: 1px solid #dee2e6;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: #212529 !important;
}
.match-card .team:last-child { border-bottom: none; }
</style>
"""


def _match_card(match_id: str, label1: str, label2: str) -> str:
    return f"""
<div class="match-card">
  <div class="mid">{match_id}</div>
  <div class="team">▫ {label1}</div>
  <div class="team">▫ {label2}</div>
</div>"""


def display_groups(groups_data: dict, last_standings: dict | None = None) -> None:
    group_names = sorted(groups_data.keys())
    cols_per_row = 3
    for i in range(0, len(group_names), cols_per_row):
        batch = group_names[i : i + cols_per_row]
        cols = st.columns(len(batch))
        for col, gname in zip(cols, batch):
            with col:
                st.markdown(f"**Grupa {gname}**")
                if last_standings and gname in last_standings:
                    rows = [
                        {
                            "#": idx + 1,
                            "Drużyna": name,
                            "Pkt": s["points"],
                            "RB": s["goal_diff"],
                            "G": s["goals_scored"],
                        }
                        for idx, (name, s) in enumerate(last_standings[gname])
                    ]
                    df = pd.DataFrame(rows).set_index("#")
                else:
                    df = (
                        pd.DataFrame(groups_data[gname], columns=["Drużyna", "ELO"])
                        .sort_values("ELO", ascending=False)
                        .reset_index(drop=True)
                    )
                    df.index += 1
                st.dataframe(df, use_container_width=True)
        st.write("")

# TODO: CSS do osobnego pliku
CARD_CSS_RESULT = """
<style>
.match-card-result {
    border: 1px solid #adb5bd;
    border-radius: 10px;
    padding: 10px 12px;
    background: #ffffff;
    margin-bottom: 6px;
    font-size: 0.82rem;
    color: #212529 !important;
}
.match-card-result .mid {
    font-size: 0.68rem;
    color: #888 !important;
    margin-bottom: 6px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.match-card-result .team {
    padding: 4px 0;
    border-bottom: 1px solid #e9ecef;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: #212529 !important;
}
.match-card-result .team:last-child { border-bottom: none; }
.match-card-result .team.winner { font-weight: 700; color: #198754 !important; }
.match-card-result .score { float: right; font-weight: 600; color: #495057 !important; margin-left: 6px; }
</style>
"""

def display_probable_groups(stats: dict) -> None:
    n = stats["n_simulations"]
    group_standings_counts = stats["group_standings_counts"]
    group_names = sorted(group_standings_counts.keys())
    cols_per_row = 3
    for i in range(0, len(group_names), cols_per_row):
        batch = group_names[i : i + cols_per_row]
        cols = st.columns(len(batch))
        for col, gname in zip(cols, batch):
            with col:
                counts = group_standings_counts[gname]
                best_order, best_count = max(counts.items(), key=lambda x: x[1])
                pct = 100 * best_count / n
                st.markdown(f"**Grupa {gname}** <span style='color:#868e96;font-size:0.8rem'>({pct:.1f}% symulacji)</span>", unsafe_allow_html=True)
                rows = [{"#": idx + 1, "Drużyna": name} for idx, name in enumerate(best_order)]
                st.dataframe(pd.DataFrame(rows).set_index("#"), use_container_width=True)
        st.write("")
    # Najczęściej awansujące drużyny z 3. miejsc
    qualified_thirds_team_counts = stats.get("qualified_thirds_team_counts", {})
    if qualified_thirds_team_counts:
        st.markdown("**Najczęściej awansujące drużyny z 3. miejsca (TOP 8)**")
        top8 = sorted(qualified_thirds_team_counts.items(), key=lambda x: -x[1])[:8]
        rows = [
            {"#": idx + 1, "Drużyna": name, "Procent": f"{100 * count / n:.1f}%"}
            for idx, (name, count) in enumerate(top8)
        ]
        st.dataframe(pd.DataFrame(rows).set_index("#"), use_container_width=True)


def display_probable_bracket(stats: dict) -> None:
    """Render most probable knockout bracket using butterfly layout."""
    labels, results = probable_bracket_from_stats(stats)
    html_content = build_butterfly_bracket_html(labels, results)
    components.html(html_content, height=1180, scrolling=True)


def _match_card_result(mid: str, m: dict) -> str:
    pen = " <small style='color:#888'>(k.)</small>" if m["penalties"] else ""
    score_str = f"{m['score1']}–{m['score2']}"
    cls1 = "winner" if m["winner"] == m["team1"] else ""
    cls2 = "winner" if m["winner"] == m["team2"] else ""
    return f"""
<div class="match-card-result">
  <div class="mid">{mid}{pen}</div>
  <div class="team {cls1}"><span class="score">{m['score1']}</span>{m['team1']}</div>
  <div class="team {cls2}"><span class="score">{m['score2']}</span>{m['team2']}</div>
</div>"""


def display_empty_bracket(knockout_raw: list[tuple[str, Any, Any]]) -> None:
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    rounds = _group_by_round(knockout_raw)
    for rnd in ROUND_ORDER:
        if rnd not in rounds:
            continue
        st.markdown(f"#### {ROUND_LABELS[rnd]}")
        matches = rounds[rnd]
        cols_per_row = min(4, len(matches))
        for i in range(0, len(matches), cols_per_row):
            batch = matches[i : i + cols_per_row]
            cols = st.columns(len(batch))
            for col, (mid, s1, s2) in zip(cols, batch):
                col.markdown(
                    _match_card(mid, _fmt_slot(s1), _fmt_slot(s2)),
                    unsafe_allow_html=True,
                )


def display_last_bracket(last_bracket: dict) -> None:
    st.markdown(CARD_CSS_RESULT, unsafe_allow_html=True)
    by_round: dict[str, list] = defaultdict(list)
    for m in last_bracket["knockout"]:
        by_round[m["stage"]].append(m)
    for rnd in ROUND_ORDER:
        if rnd not in by_round:
            continue
        st.markdown(f"#### {ROUND_LABELS[rnd]}")
        matches = by_round[rnd]
        cols_per_row = min(4, len(matches))
        for i in range(0, len(matches), cols_per_row):
            batch = matches[i : i + cols_per_row]
            cols = st.columns(len(batch))
            for col, m in zip(cols, batch):
                col.markdown(
                    _match_card_result(m["match_id"], m),
                    unsafe_allow_html=True,
                )


def display_butterfly_bracket(
    knockout_raw: list[tuple[str, Any, Any]],
    last_bracket: dict | None = None,
) -> None:
    """Render center-converging butterfly bracket visualization."""
    resolved_labels: dict[str, tuple[str, str]] = {}
    if last_bracket:
        for match in last_bracket["knockout"]:
            resolved_labels[match["match_id"]] = (
                match["team1"],
                match["team2"],
            )
    labels = labels_from_knockout_raw(
        knockout_raw, _fmt_slot, resolved_labels or None
    )
    results = (
        results_from_last_bracket(last_bracket["knockout"])
        if last_bracket
        else None
    )
    html_content = build_butterfly_bracket_html(labels, results)
    components.html(html_content, height=1180, scrolling=True)


SCORE_HEATMAP_SCALE = [
    [0.0, "#1a1d29"],
    [0.08, "#2d3a4f"],
    [0.2, "#3d6b4f"],
    [0.4, "#74c69d"],
    [0.65, "#ffd166"],
    [1.0, "#ef476f"],
]
SCORE_OUTCOME_COLORS = {
    "Wygrana gospodarza": "#2f9e44",
    "Remis": "#f59f00",
    "Wygrana gościa": "#e03131",
}


def _score_outcome(s1: int, s2: int) -> str:
    if s1 > s2:
        return "Wygrana gospodarza"
    if s1 < s2:
        return "Wygrana gościa"
    return "Remis"


def plot_top_results_bar(
    counts: dict[tuple[int, int], int],
    home: str,
    away: str,
    n: int,
    top_n: int = 12,
) -> go.Figure:
    """Bar chart of the most frequent scores for a single fixture."""
    n_total = n if n > 0 else 1
    
    sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # Weź top N wyników
    top_results = sorted_results[:top_n]

    # Przygotuj dane
    labels = [f"{h}:{a}" for (h, a), _ in top_results]
    percentages = [100 * count / n_total for _, count in top_results]

    # Kolory dla wygranej gospodarza, remisu, gościa
    colors = []
    for (h, a), _ in top_results:
        if h > a:
            colors.append("#2ca02c")  # zielony - wygrana gospodarza
        elif h < a:
            colors.append("#de2d26")  # czerwony - wygrana gościa
        else:
            colors.append("#ffb800")  # żółty/piaskowy - remis

    # Tworzenie słupków
    fig = go.Figure(data=go.Bar(
        x=labels,
        y=percentages,
        marker_color=colors,
        text=[f"{p:.1f}%" for p in percentages],
        textposition="outside",
        textfont=dict(size=11, color="#333333"),
        hovertemplate='<b>%{x}</b><br>Prawdopodobieństwo: <b>%{y:.1f}%</b><extra></extra>'
    ))

    # Dodanie legendy (ręcznie, bo plotly nie ogarnia automatycznie)
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color="#2ca02c"),
        name='Wygrana gospodarza',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color="#de2d26"),
        name='Wygrana gościa',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color="#ffb800"),
        name='Remis',
        showlegend=True
    ))

    # Ustawienie układu wykresu
    fig.update_layout(
        title=dict(
            text=f"<b>{home} – {away}</b><br><sub>{n} unikalnych wyników • top {top_n}</sub>",
            font=dict(size=14),
            x=0.5
        ),
        xaxis=dict(
            title="Wynik (gospodarz : gość)",
            tickangle=-45,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title="Prawdopodobieństwo (%)",
            tickfont=dict(size=11),
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='#cccccc'
        ),
        height=500,
        margin=dict(l=50, r=50, t=80, b=100),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            title="Typ wyniku",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    return fig

def _group_match_score_top_bars(counts: dict, n: int, top_n: int = 12):
    """Poziomy wykres najczęstszych wyników."""
    rows = [
        {
            "Wynik": f"{s1}:{s2}",
            "Procent": round(100 * count / n, 2),
            "Liczba": count,
            "Typ": _score_outcome(s1, s2),
        }
        for (s1, s2), count in counts.items()
    ]
    df = pd.DataFrame(rows).sort_values("Procent", ascending=True).tail(top_n)
    fig = px.bar(
        df,
        x="Procent",
        y="Wynik",
        orientation="h",
        color="Typ",
        color_discrete_map=SCORE_OUTCOME_COLORS,
        text="Procent",
        custom_data=["Liczba"],
        labels={"Procent": "Prawdopodobieństwo (%)", "Wynik": "Wynik", "Typ": "Typ wyniku"},
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{customdata[0]} razy (%{x:.1f}%)<extra></extra>",
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(title="Typ wyniku", orientation="h", yanchor="bottom", y=1.02, x=0),
        height=max(250, len(df) * 28),
        margin=dict(l=0, r=40, t=30, b=20),
        xaxis_title="Prawdopodobieństwo (%)",
        yaxis_title=None,
    )
    return fig


def display_team_info(stats: dict):
    n = stats["n_simulations"]
    team_exit = stats["team_exit_stages"]
    st.subheader("👕 Szczegóły wybranej drużyny")
    selected = st.selectbox("Wybierz drużynę:", sorted(team_exit.keys()), key="team_detail_select")
    if not selected:
        return
    stages = team_exit[selected]
    detail_df = pd.DataFrame(
        [
            {
                "Etap": STAGES_LABELS[s],
                "Liczba": stages.count(s),
                "Procent": f"{100 * stages.count(s) / n:.1f}%",
            }
            for s in STAGES_ORDER
            if stages.count(s) > 0
        ]
    )
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
    with col_b:
        STAGE_COLORS = {
            "Faza grupowa":  "#6c757d",
            "1/16 finału":   "#4dabf7",
            "1/8 finału":    "#339af0",
            "Ćwierćfinał":   "#f59f00",
            "Półfinał":      "#f76707",
            "3. miejsce":    "#ae3ec9",
            "Finał":         "#e03131",
            "Zwycięzca":     "#2f9e44",
        }
        detail_df["_color"] = detail_df["Etap"].map(STAGE_COLORS)
        fig_detail = px.pie(
            detail_df,
            names="Etap",
            values="Liczba",
            title=f"Rozkład etapów — {selected}",
            color="Etap",
            color_discrete_map=STAGE_COLORS,
        )
        fig_detail.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>%{value} symulacji (%{percent})<extra></extra>",
        )
        fig_detail.update_layout(showlegend=True)
        st.plotly_chart(fig_detail, use_container_width=True)
    st.markdown("**Szansa na dotarcie do etapu (lub dalej)**")
    cumulative_data = [
        ("1/16 finału",  ("R_32", "R_16", "QF", "SF", "3RD", "F", "Zwycięzca")),
        ("1/8 finału",   ("R_16", "QF", "SF", "3RD", "F", "Zwycięzca")),
        ("Ćwierćfinał",  ("QF", "SF", "3RD", "F", "Zwycięzca")),
        ("Półfinał",     ("SF", "3RD", "F", "Zwycięzca")),
        ("Finał",        ("F", "Zwycięzca")),
        ("Zwycięzca",    ("Zwycięzca",)),
    ]
    cumul_df = pd.DataFrame([
        {"Etap": label, "Procent": round(100 * sum(1 for s in stages if s in stage_set) / n, 2)}
        for label, stage_set in cumulative_data
    ])
    fig_cumul = px.bar(
        cumul_df,
        x="Etap",
        y="Procent",
        color="Procent",
        color_continuous_scale="YlGn",
        text="Procent",
        labels={"Procent": "Prawdopodobieństwo (%)"},
    )
    fig_cumul.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_cumul.update_layout(
        coloraxis_showscale=False,
        yaxis_title="Prawdopodobieństwo (%)",
        yaxis_range=[0, 110],
        xaxis_title=None,
        margin=dict(b=40, t=20),
    )
    st.plotly_chart(fig_cumul, use_container_width=True)
    st.markdown("**Najczęstsi rywale w fazie pucharowej**")
    ko_counts = stats["knockout_meeting_counts"]
    ko_wins = stats.get("knockout_meeting_wins", {})
    opponents_by_stage: dict[str, list] = defaultdict(list)
    for (pair, stage), count in ko_counts.items():
        if selected in pair:
            remainder = list(pair - {selected})
            if not remainder:
                continue
            opponent = remainder[0]
            wins = ko_wins.get((pair, stage), {}).get(selected, 0)
            losses = count - wins
            opponents_by_stage[stage].append({
                "Rywal": opponent,
                "Spotkań": count,
                "Wygrane": wins,
                "Przegrane": losses,
            })
    if any(opponents_by_stage.values()):
        st.markdown("""
<style>
.opp-table { width:100%; border-collapse:collapse; font-size:0.82rem; margin-bottom:8px; }
.opp-table th { text-align:left; padding:5px 8px; color:#868e96; font-weight:600;
                border-bottom:2px solid #343a40; font-size:0.72rem; text-transform:uppercase; letter-spacing:.04em; }
.opp-table td { padding:5px 8px; border-bottom:1px solid #2c2f33; vertical-align:middle; }
.opp-table tr:last-child td { border-bottom:none; }
.opp-name { font-weight:500; }
.opp-pct  { color:#868e96; font-size:0.75rem; }
.bar-wrap { background:#2c2f33; border-radius:4px; height:8px; width:120px; overflow:hidden; display:flex; }
.bar-win  { background:#2f9e44; height:8px; }
.bar-loss { background:#e03131; height:8px; }
.wl-nums  { font-size:0.75rem; white-space:nowrap; }
.wl-w { color:#2f9e44; font-weight:600; }
.wl-l { color:#e03131; font-weight:600; }
</style>
""", unsafe_allow_html=True)
        for stage in ["R_32", "R_16", "QF", "SF", "3RD", "F"]:
            if stage not in opponents_by_stage:
                continue
            rows = sorted(opponents_by_stage[stage], key=lambda x: -x["Spotkań"])[:10]
            st.markdown(f"<div style='color:#868e96;font-size:0.75rem;font-weight:600;text-transform:uppercase;"
                        f"letter-spacing:.04em;margin:10px 0 4px'>{ROUND_LABELS[stage]}</div>",
                        unsafe_allow_html=True)
            html = '<table class="opp-table"><tr><th>Rywal</th><th>Spotk.</th><th style="min-width:140px">W / P</th><th>Wynik</th><th>% wygranych</th></tr>'
            for r in rows:
                total = r["Spotkań"]
                w_pct = r["Wygrane"] / total if total else 0
                l_pct = r["Przegrane"] / total if total else 0
                meet_pct = 100 * total / n
                win_pct = 100 * w_pct
                bar = (f'<div class="bar-wrap">'
                       f'<div class="bar-win" style="width:{w_pct*100:.1f}%"></div>'
                       f'<div class="bar-loss" style="width:{l_pct*100:.1f}%"></div>'
                       f'</div>')
                wl = (f'<span class="wl-w">{r["Wygrane"]}W</span>'
                      f' <span style="color:#495057">·</span> '
                      f'<span class="wl-l">{r["Przegrane"]}P</span>')
                if win_pct > 50:
                    win_pct_color = "#2f9e44"
                elif win_pct < 50:
                    win_pct_color = "#e03131"
                else:
                    win_pct_color = "#f59f00"
                html += (f'<tr>'
                         f'<td><span class="opp-name">{r["Rywal"]}</span></td>'
                         f'<td><span class="opp-pct">{meet_pct:.1f}%</span></td>'
                         f'<td>{bar}</td>'
                         f'<td class="wl-nums">{wl}</td>'
                         f'<td style="font-weight:700;color:{win_pct_color}">{win_pct:.1f}%</td>'
                         f'</tr>')
            html += '</table>'
            st.markdown(html, unsafe_allow_html=True)
    else:
        st.caption("Brak danych — drużyna nie dotarła do fazy pucharowej w żadnej symulacji.")


def display_group_score_distributions(
    stats: dict, group_schedule: dict
) -> None:
    """Show score distribution charts for group-stage matches."""
    score_counts = stats.get("group_match_score_counts")
    if not score_counts:
        return
    n = stats["n_simulations"]
    st.subheader("📋 Rozkład wyników meczów grupowych")
    for gname in sorted(group_schedule.keys()):
        matches = group_schedule[gname]
        options = [f"{home} – {away}" for home, away in matches]
        with st.expander(f"Grupa {gname}", expanded=False):
            selected = st.selectbox(
                "Wybierz mecz",
                options,
                key=f"group_score_match_{gname}",
            )
            idx = options.index(selected)
            home, away = matches[idx]
            counts = score_counts.get((home, away), {})
            if not counts:
                st.info("Brak danych dla tego meczu.")
                continue
            unique_n = len(counts)
            unique_label = (
                "unikalny wynik" if unique_n == 1
                else "unikalne wyniki" if unique_n in (2, 3, 4)
                else "unikalnych wyników"
            )
            st.markdown(f"**{home} – {away}** · {unique_n} {unique_label}")
            bars_fig = _group_match_score_top_bars(counts, n)
            if st.session_state.get("horizontal_charts", False):
                st.markdown("**Najczęstsze wyniki**")
                st.plotly_chart(bars_fig, use_container_width=True)
            else:
                st.markdown(
                    "<div style='font-size:0.8rem;color:#868e96;margin-bottom:4px'>"
                    "Najczęstsze wyniki</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(bars_fig, use_container_width=True)


def display_results(stats: dict):
    n = stats["n_simulations"]
    team_exit = stats["team_exit_stages"]
    # 1. Szansa na wygranie turnieju
    def _bar(df, colorscale):
        """Wykres słupkowy — poziomy lub pionowy zależnie od ustawienia użytkownika."""
        horizontal = st.session_state.get("horizontal_charts", False)
        if horizontal:
            df = df.sort_values("Procent", ascending=True)
            fig = px.bar(
                df,
                x="Procent",
                y="Drużyna",
                orientation="h",
                color="Procent",
                color_continuous_scale=colorscale,
                text="Procent",
                labels={"Procent": "%"},
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="auto", textfont=dict(size=15))
            fig.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Prawdopodobieństwo (%)",
                yaxis_title=None,
                height=max(320, len(df) * 26),
                margin=dict(l=0, r=10, t=10, b=30),
                yaxis=dict(automargin=True, tickfont=dict(size=13)),
                xaxis=dict(automargin=True, tickfont=dict(size=12)),
                font=dict(size=13),
            )
        else:
            df = df.sort_values("Procent", ascending=False)
            fig = px.bar(
                df,
                x="Drużyna",
                y="Procent",
                color="Procent",
                color_continuous_scale=colorscale,
                text="Procent",
                labels={"Procent": "Prawdopodobieństwo (%)"},
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                coloraxis_showscale=False,
                yaxis_title="Prawdopodobieństwo (%)",
                xaxis_tickangle=-40,
                margin=dict(b=100),
            )
        return fig

    st.subheader("🏆 Szansa na wygranie turnieju")
    winner_df = (
        pd.DataFrame(
            [
                (t, round(100 * sum(1 for s in stages if s == "Zwycięzca") / n, 2))
                for t, stages in team_exit.items()
            ],
            columns=["Drużyna", "Procent"],
        )
        .query("Procent > 0")
        .reset_index(drop=True)
    )
    st.plotly_chart(_bar(winner_df, "YlOrRd"), use_container_width=True)
    # 2. Wykresy "szansa na dotarcie do etapu X lub dalej"
    REACH_STAGES = [
        ("🥈 Szansa na dotarcie do finału",     ("F", "Zwycięzca"),              "Reds"),
        ("🥊 Szansa na dotarcie do półfinału",  ("SF", "3RD", "F", "Zwycięzca"), "Oranges"),
        ("⚽ Szansa na dotarcie do ćwierćfinału", ("QF", "SF", "3RD", "F", "Zwycięzca"), "YlOrBr"),
        ("🔵 Szansa na dotarcie do 1/8 finału", ("R_16", "QF", "SF", "3RD", "F", "Zwycięzca"), "Blues"),
        ("⚪ Szansa na dotarcie do 1/16 finału", ("R_32", "R_16", "QF", "SF", "3RD", "F", "Zwycięzca"), "Greys"),
    ]
    for title, reach_stages, colorscale in REACH_STAGES:
        st.subheader(title)
        df = (
            pd.DataFrame(
                [
                    (t, round(100 * sum(1 for s in stages if s in reach_stages) / n, 2))
                    for t, stages in team_exit.items()
                ],
                columns=["Drużyna", "Procent"],
            )
            .query("Procent > 0")
            .reset_index(drop=True)
        )
        st.plotly_chart(_bar(df, colorscale), use_container_width=True)
    # 3. Heatmapa rozkładu etapów
    st.subheader("📊 Rozkład etapów dla każdej drużyny (%)")
    all_teams = sorted(team_exit.keys())
    stage_weights = {s: i for i, s in enumerate(STAGES_ORDER)}
    rows = []
    for team in all_teams:
        stages = team_exit[team]
        row = {s: round(100 * stages.count(s) / n, 1) for s in STAGES_ORDER}
        row["_sort"] = sum(row[s] * stage_weights[s] for s in STAGES_ORDER)
        row["Drużyna"] = team
        rows.append(row)
    heat_df = (
        pd.DataFrame(rows)
        .sort_values("_sort", ascending=False)
        .set_index("Drużyna")
        .drop(columns=["_sort"])[STAGES_ORDER]
    )
    heat_df.columns = [STAGES_LABELS[c] for c in heat_df.columns]
    if st.session_state.get("horizontal_charts", False):
        # Mobile: tabela z kolorowym tłem — przewijalna, pinch-zoomowalna
        st.caption("Przewiń w poziomie, żeby zobaczyć wszystkie etapy.")
        styled = (
            heat_df.style
            .background_gradient(cmap="Blues", axis=None, vmin=0)
            .format("{:.1f}%")
            .set_properties(**{"font-size": "13px", "text-align": "center"})
        )
        st.dataframe(styled, use_container_width=True, height=min(600, max(300, len(all_teams) * 28 + 40)))
    else:
        fig_heat = px.imshow(
            heat_df,
            color_continuous_scale="Blues",
            labels=dict(color="%"),
            aspect="auto",
            text_auto=".1f",
        )
        fig_heat.update_traces(textfont=dict(size=11))
        fig_heat.update_layout(
            height=max(500, len(all_teams) * 22),
            xaxis_title="Etap turnieju",
            yaxis_title=None,
            coloraxis_colorbar=dict(title="%"),
            margin=dict(l=0, r=20, t=10, b=30),
            yaxis=dict(automargin=True),
            font=dict(size=12),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    # 4. Rozkład wyników meczów grupowych
    display_group_score_distributions(stats, group_schedule)

    # 5. Najczęstsze spotkania w fazie pucharowej
    st.subheader("⚔️ Najczęstsze spotkania w fazie pucharowej (top 20)")
    ko_rows = []
    for (pair, stage), count in sorted(
        stats["knockout_meeting_counts"].items(), key=lambda x: -x[1]
    )[:20]:
        t1, t2 = list(pair)
        ko_rows.append(
            {
                "Etap": STAGES_LABELS.get(stage, stage),
                "Drużyna 1": t1,
                "Drużyna 2": t2,
                "Spotkań": count,
                "Procent": f"{100 * count / n:.1f}%",
            }
        )
    st.dataframe(pd.DataFrame(ko_rows), use_container_width=True, hide_index=True)

st.set_page_config(
    page_title="Symulator MŚ",
    page_icon="⚽",
    layout="wide",
)
st.title("⚽ Symulator Mistrzostw Świata")
st.caption("Analiza Monte Carlo — rozkłady prawdopodobieństwa wyników wszystkich drużyn. Wybierz parametry symulacji po lewej stronie i kliknij 'Uruchom symulację'.")
groups_data, group_schedule, knockout_raw, schedule_presets = load_initial_data()

# Wykrywanie szerokości ekranu przy pierwszym ładowaniu — ustawia domyślną orientację wykresów
if "horizontal_charts" not in st.session_state:
    components.html(
        """
        <script>
        const width = window.innerWidth ||
                      document.documentElement.clientWidth ||
                      document.body.clientWidth;
        const isMobile = width < 768;
        const msg = JSON.stringify({type: "streamlit:setComponentValue", value: isMobile});
        window.parent.postMessage(msg, "*");
        </script>
        """,
        height=0,
    )
    st.session_state["horizontal_charts"] = False

# Pasek boczny
with st.sidebar:
    st.header("⚙️ Parametry symulacji")
    with st.form("sim_params_form"):
        lambda_base = st.number_input(
            "lambda_base",
            min_value=0.5,
            max_value=5.0,
            value=1.3,
            step=0.05,
            format="%.2f",
            help="Bazowe oczekiwane gole na drużynę przy równych ELO (typowo 1.0–1.5 dla MŚ)",
        )
        k = st.number_input(
            "k (skala wpływu ELO)",
            min_value=0.01,
            max_value=2.0,
            value=0.25,
            step=0.01,
            format="%.2f",
            help="Jak mocno różnica ELO skaluje oczekiwane gole (Dobierac eksperymentalnie, typowo 0.1–0.3)",
        )
        n_simulations = st.number_input(
            "Liczba symulacji",
            min_value=1,
            max_value=100_000,
            value=1000,
            step=1,
            help="Więcej symulacji = dokładniejsze wyniki, ale dłuższy czas obliczeń",
        )
        st.divider()
        run_btn = st.form_submit_button("▶ Uruchom symulację", type="primary", use_container_width=True)
    st.divider()
    st.checkbox(
        "📱 Poziome wykresy",
        key="horizontal_charts",
        help="Włącz dla telefonów i wąskich ekranów. Na komputerze domyślnie wyłączone.",
    )

# Symulacja
if run_btn:
    with st.spinner(f"Trwa symulacja {int(n_simulations):,} turniejów… ⏳"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _stats = run_monte_carlo(
                COUNTRIES_FILE,
                GROUPS_FILE,
                SCHEDULE_GROUPS_FILE,
                SCHEDULE_KNOCKOUT_FILE,
                n=int(n_simulations),
                lambda_base=float(lambda_base),
                k=float(k),
                fixed_group_results=st.session_state.get("fixed_group_results"),
                fixed_knockout_results=st.session_state.get(
                    "fixed_knockout_results"
                ),
            )
        st.session_state["stats"] = _stats
        st.session_state["sim_params"] = {
            "n": int(n_simulations),
            "lambda_base": float(lambda_base),
            "k": float(k),
        }
    st.success(f"✅ Ukończono {int(n_simulations):,} symulacji!")

# Zakładki
tab_groups, tab_butterfly, tab_probable, tab_results, tab_team, tab_info = st.tabs(
    [
        "📋 Grupy",
        "🏟️ Wizualizacja drabinki",
        "🎯 Najbardziej prawdopodobna drabinka",
        "📊 Wyniki symulacji",
        "👕 Szczegóły drużyn",
        "ℹ️ Opis działania",
    ]
)

with tab_groups:
    n_presets = len(schedule_presets)
    if n_presets:
        use_presets = st.checkbox(
            f"📥 Użyj prawdziwych wyników z harmonogramu ({n_presets} {'mecz' if n_presets == 1 else 'mecze' if n_presets in (2, 3, 4) else 'meczów'})",
            value=True,
            key="use_presets",
            help="Odznacz, żeby zignorować wpisane wyniki i wygenerować cały turniej losowo.",
        )
    else:
        use_presets = False

    with st.expander("✏️ Wprowadź własne wyniki meczów grupowych (opcjonalnie)", expanded=False):
        st.caption(
            "Wypełnij wyniki dla wybranych meczów — zostaną one użyte we wszystkich symulacjach jako stały wynik. "
            "Pozostawienie pól pustych oznacza, że mecz zostanie rozegrany losowo."
        )
        # Klucz widgetów zależny od use_presets + licznik czyszczeń — wymusza reset wartości
        _grp_clear_count = st.session_state.get("grp_clear_count", 0)
        preset_key_suffix = f"{'on' if use_presets else 'off'}_{_grp_clear_count}"

        for gname in sorted(group_schedule.keys()):
            st.markdown(f"**Grupa {gname}**")
            for home, away in group_schedule[gname]:
                preset = schedule_presets.get((home, away)) if use_presets else None
                col_home, col_s1, col_dash, col_s2, col_away = st.columns([3, 1, 0.3, 1, 3])
                col_home.markdown(f"<div style='padding-top:6px;text-align:right'>{home}</div>", unsafe_allow_html=True)
                col_s1.number_input(
                    "g1", min_value=0, max_value=30, value=preset[0] if preset is not None else None,
                    key=f"fg_{gname}_{home}_{away}_1_{preset_key_suffix}", label_visibility="collapsed"
                )
                col_dash.markdown("<div style='padding-top:6px;text-align:center'>–</div>", unsafe_allow_html=True)
                col_s2.number_input(
                    "g2", min_value=0, max_value=30, value=preset[1] if preset is not None else None,
                    key=f"fg_{gname}_{home}_{away}_2_{preset_key_suffix}", label_visibility="collapsed"
                )
                col_away.markdown(f"<div style='padding-top:6px'>{away}</div>", unsafe_allow_html=True)

        col_apply_g, col_clear_g = st.columns(2)
        if col_apply_g.button("✅ Zastosuj wyniki", use_container_width=True, key="apply_group_btn"):
            fixed_inputs: dict[tuple[str, str], tuple[int, int]] = {}
            for gname in sorted(group_schedule.keys()):
                for home, away in group_schedule[gname]:
                    s1 = st.session_state.get(f"fg_{gname}_{home}_{away}_1_{preset_key_suffix}")
                    s2 = st.session_state.get(f"fg_{gname}_{home}_{away}_2_{preset_key_suffix}")
                    if s1 is not None and s2 is not None:
                        fixed_inputs[(home, away)] = (int(s1), int(s2))
            st.session_state["fixed_group_results"] = fixed_inputs if fixed_inputs else None
            n_fixed = len(fixed_inputs)
            st.success(f"Zablokowano {n_fixed} {'mecz' if n_fixed == 1 else 'mecze' if n_fixed in (2, 3, 4) else 'meczów'}." if n_fixed else "Wyniki wyczyszczone — wszystkie mecze będą losowane.")
        elif col_clear_g.button("🗑️ Wyczyść wszystkie", use_container_width=True, key="clear_group_btn"):
            st.session_state["fixed_group_results"] = None
            st.session_state["grp_clear_count"] = _grp_clear_count + 1
            st.success("Wszystkie wyniki grupowe zostały wyczyszczone.")
        elif "fixed_group_results" not in st.session_state:
            st.session_state["fixed_group_results"] = None
        _saved = st.session_state.get("fixed_group_results")
        if _saved:
            n_saved = len(_saved)
            st.info(f"Aktualnie zapisano: {n_saved} {'mecz' if n_saved == 1 else 'mecze' if n_saved in (2, 3, 4) else 'meczów'}.")

    _lb = st.session_state.get("stats", {}).get("last_bracket")
    if _lb:
        st.subheader("Wyniki fazy grupowej — ostatnie losowanie")
        st.caption("Tabela końcowa z ostatniej przeprowadzonej symulacji (Pkt = punkty, RB = różnica bramek, G = bramki strzelone).")
        display_groups(groups_data, last_standings=_lb["groups"])
        # Tabela 3. miejsc — ostatnie losowanie
        lb_groups = _lb["groups"]
        thirds_last = [
            (gname, standings[2][0], standings[2][1])
            for gname, standings in lb_groups.items()
            if len(standings) > 2
        ]
        thirds_last_sorted = sorted(thirds_last, key=lambda x: (x[2]["points"], x[2]["goal_diff"], x[2]["goals_scored"]), reverse=True)
        st.markdown("---")
        st.markdown("**Tabela 3. miejsc — ostatnie losowanie**")
        st.caption("Zielone wiersze = drużyny awansowane do fazy pucharowej (8 najlepszych 3. miejsc).")
        thirds_rows = [
            {
                "#": idx + 1,
                "Grupa": gname,
                "Drużyna": name,
                "Pkt": stats["points"],
                "RB": stats["goal_diff"],
                "G": stats["goals_scored"],
                "_qualified": idx < 8,
            }
            for idx, (gname, name, stats) in enumerate(thirds_last_sorted)
        ]
        thirds_df = pd.DataFrame(thirds_rows).set_index("#")
        qualified_mask = thirds_df.pop("_qualified")
        styled_thirds = thirds_df.style.apply(
            lambda _: ["background-color: #2d6a4f" if qualified_mask.iloc[i] else "" for i in range(len(thirds_df))],
            axis=0,
        )
        st.dataframe(styled_thirds, use_container_width=True)
        # Wyniki poszczególnych meczów grupowych
        st.markdown("---")
        st.markdown("**Wyniki meczów — ostatnie losowanie**")
        for gname in sorted(_lb["group_matches"].keys()):
            with st.expander(f"Grupa {gname}"):
                for m in _lb["group_matches"][gname]:
                    winner = m["team1"] if m["score1"] > m["score2"] else (m["team2"] if m["score2"] > m["score1"] else None)
                    t1_bold = f"**{m['team1']}**" if winner == m["team1"] else m["team1"]
                    t2_bold = f"**{m['team2']}**" if winner == m["team2"] else m["team2"]
                    st.markdown(f"{t1_bold} {m['score1']} – {m['score2']} {t2_bold}")
    else:
        st.subheader("Podział na grupy")


with tab_butterfly:
    st.subheader("Wizualizacja drabinki")
    st.caption(
        "Drabinka w układzie „motylkowym”: finał w środku, górna połowa (mecze 1–8 "
        "w 1/16) zbiega w dół, dolna połowa (mecze 9–16) zbiega do środka."
    )
    _lb_bfly = st.session_state.get("stats", {}).get("last_bracket")
    if _lb_bfly:
        st.info(
            "Pokazano wyniki **ostatniego losowania** z symulacji."
        )
    else:
        st.info(
            "Uruchom symulację, aby zobaczyć nazwy drużyn i wyniki. "
            "Na razie widoczne są etykiety slotów (np. „1. Gr. A”)."
        )
    display_butterfly_bracket(knockout_raw, last_bracket=_lb_bfly)

with tab_probable:
    st.subheader("Najbardziej prawdopodobna drabinka")
    if "stats" in st.session_state and st.session_state["stats"].get("match_slot_pairs"):
        st.info(
            "🎯 Drabinka pokazuje **najczęściej występującą parę drużyn** w każdym meczu "
            "oraz **najczęstszego zwycięzcę** (procent przy nazwie drużyny). "
            "„Najczęstsza para” = jak często te dwie drużyny spotkały się w tym miejscu drabinki."
        )
        st.subheader("Najbardziej prawdopodobne tabele grup")
        display_probable_groups(st.session_state["stats"])
        st.subheader("Najbardziej prawdopodobna drabinka pucharowa")
        display_probable_bracket(st.session_state["stats"])
    else:
        st.info("Uruchom symulację, aby zobaczyć najbardziej prawdopodobną drabinkę.")

with tab_info:
    st.subheader("Jak działa symulator?")
    st.markdown(
        """
        Symulator przeprowadza tysiące wirtualnych turniejów i na podstawie ich wyników
        oblicza **prawdopodobieństwo** różnych rezultatów — np. kto ma największe szanse na
        wygranie mistrzostw. Poniżej dowiesz się, jak dokładnie przebiega każdy krok.
        """
    )

    st.markdown("---")

    # --- KROK 1: ELO ---
    st.markdown("### 1️⃣ Punkty ELO — siła każdej drużyny")
    st.markdown(
        """
        Każda drużyna ma przypisany **ranking ELO** — liczbę, która mówi, jak silna jest ta drużyna.
        Im wyższy ELO, tym silniejsza drużyna. Ranking ten jest używany przez wiele organizacji
        piłkarskich i wyliczany jest na podstawie historycznych wyników meczów. Dane ELO dla drużyny zostały pobrane ze strony [World Football Elo Ratings](https://www.eloratings.net/), która aktualizuje rankingi po każdym meczu międzynarodowym.

        **Przykład:**
        | Drużyna | ELO |
        |---|---|
        | Brazylia | 2090 |
        | Francja | 2055 |
        | Polska | 1720 |
        | San Marino | 780 |

        Różnica ELO między drużynami bezpośrednio wpływa na to, ile goli spodziewamy się w meczu.
        """
    )

    st.markdown("---")

    # --- KROK 2: SYMULACJA MECZU ---
    st.markdown("### 2️⃣ Symulacja wyniku meczu")
    st.markdown(
        """
        Wynik meczu nie jest z góry ustalony — jest **losowany**. Symulator oblicza, ile goli
        *statystycznie spodziewamy się* po każdej drużynie, a następnie losuje rzeczywisty wynik.

        #### Jak to działa krok po kroku?

        **Krok A — Obliczenie oczekiwanej liczby goli (λ)**

        Parametr `lambda_base` to bazowa liczba goli przy meczu równych drużyn (np. 1,3).
        Jeśli jedna drużyna jest silniejsza, jej λ rośnie, a słabszej — maleje:

        ```
        λ₁ = lambda_base × 10^(+diff × k)
        λ₂ = lambda_base × 10^(−diff × k)

        gdzie: diff = (ELO₁ − ELO₂) / 400
        ```

        **Przykład:** Brazylia (ELO 2090) vs Polska (ELO 1720), lambda_base = 1,3, k = 0,25:
        - diff = (2090 − 1720) / 400 = **0,925**
        - λ_Brazylia = 1,3 × 10^(0,925 × 0,25) ≈ **2,14 gola**
        - λ_Polska   = 1,3 × 10^(−0,925 × 0,25) ≈ **0,79 gola**

        Brazylia statystycznie strzela dwa razy więcej goli niż Polska.

        **Krok B — Losowanie bramek (rozkład Poissona)**

        Dla każdej drużyny liczba strzelonych bramek jest **losowana** z rozkładu Poissona
        o parametrze λ. Oznacza to, że np. przy λ = 2,14 najczęściej padną 2 gole, ale
        zdarzają się też 0, 1, 3, 4...

        > 🎲 Dlatego nawet słabsza drużyna może wygrać — podobnie jak w prawdziwej piłce nożnej!

        **Przykładowe losowe wyniki** przy tych samych oczekiwaniach (λ_Brazylia=2,14, λ_Polska=0,79):
        | Symulacja | Brazylia | Polska |
        |---|---|---|
        | 1 | 3 | 0 |
        | 2 | 1 | 1 |
        | 3 | 2 | 2 |
        | 4 | 0 | 1 |
        """
    )

    st.markdown("---")

    # --- KROK 3: AKTUALIZACJA ELO ---
    st.markdown("### 3️⃣ Aktualizacja rankingu ELO po meczu")
    st.markdown(
        """
        Po każdym meczu rankingi ELO obu drużyn są **aktualizowane** na podstawie wyniku.
        Wygrana z mocniejszym rywalem daje więcej punktów niż wygrana z słabszym.
        Wysoka różnica bramek dodatkowo zwiększa zmianę rankingu.

        | Różnica bramek | Mnożnik K |
        |---|---|
        | 1 gol | ×1,0 (standardowy) |
        | 2 gole | ×1,5 |
        | 3 gole | ×1,75 |
        | 4+ gole | rosnący |

        Dzięki temu rankingi ELO ewoluują przez cały turniej — drużyna, która wygrywa,
        staje się faworytem kolejnych meczów.
        """
    )

    st.markdown("---")

    # --- KROK 4: FAZA GRUPOWA ---
    st.markdown("### 4️⃣ Faza grupowa")
    st.markdown(
        """
        Każda drużyna rozgrywa 3 mecze w grupie (po jednym z każdym rywalem).
        Po wszystkich meczach drużyny są **klasyfikowane** według kolejności:

        1. **Punkty** (3 za wygraną, 1 za remis, 0 za porażkę)
        2. **Różnica bramek** (przy równej liczbie punktów)
        3. **Liczba strzelonych bramek** (przy równej różnicy bramek)

        **Przykład tabeli grupy:**
        | # | Drużyna | Pkt | RB | G |
        |---|---|---|---|---|
        | 1 | Brazylia | 7 | +4 | 6 |
        | 2 | Francja | 6 | +2 | 5 |
        | 3 | Polska | 3 | −2 | 3 |
        | 4 | Arabia Saudyjska | 0 | −4 | 1 |

        Z każdej grupy awansują **dwie pierwsze drużyny** do fazy pucharowej.
        oraz 8 najlepszych z trzecich miejsc.
        """
    )

    st.markdown("---")

    # --- KROK 5: FAZA PUCHAROWA ---
    st.markdown("### 5️⃣ Faza pucharowa")
    st.markdown(
        """
        Od etapu 1/16 finału obowiązuje zasada **"przegrany odpada"**.
        Mecze są symulowane tak samo jak w fazie grupowej — poprzez losowanie goli z rozkładu Poissona.

        **Co jeśli jest remis?**
        - W fazie grupowej remis jest normalnym wynikiem.
        - W fazie pucharowej, jeśli po 90 minutach jest remis, gra się **rzuty karne**.
        - Rzuty karne są symulowane jako **moneta** — każda drużyna ma 50% szans na wygraną.
        """
    )

    st.markdown("---")

    # --- KROK 6: MONTE CARLO ---
    st.markdown("### 6️⃣ Metoda Monte Carlo — tysiące turniejów")
    st.markdown(
        """
        Jeden turniej to za mało, żeby wyciągnąć wnioski — wynik zależy od losowania.
        Dlatego symulator powtarza cały turniej **N razy** (domyślnie 1000).

        Po N symulacjach sprawdzamy, ile razy każda drużyna osiągnęła dany etap:

        **Przykład dla 1000 symulacji:**
        | Drużyna | Wygrała turniej | Dotarła do finału |
        |---|---|---|
        | Brazylia | 183 razy (18,3%) | 312 razy (31,2%) |
        | Francja | 151 razy (15,1%) | 278 razy (27,8%) |
        | Kolumbia | 12 razy (1,2%) | 34 razy (3,4%) |

        Im więcej symulacji, tym **dokładniejsze** są wyniki. Przy 10 000 symulacjach
        wyniki są bardzo stabilne; przy 100 już bardziej przypadkowe.

        > 💡 **Wskazówka:** Zmień parametr `lambda_base` i `k` w panelu bocznym, aby zobaczyć,
        > jak wpływają na rozkład szans. Wyższe `k` = większy wpływ rankingu ELO na wynik.
        """
    )

with tab_results:
    if "stats" in st.session_state:
        p = st.session_state["sim_params"]
        st.caption(
            f"Parametry: lambda_base={p['lambda_base']}, k={p['k']}, "
            f"n={p['n']:,} symulacji"
        )
        display_results(st.session_state["stats"])
    elif not run_btn:
        st.info(
            "Ustaw parametry w panelu bocznym i kliknij **▶ Uruchom symulację**, "
            "aby zobaczyć wyniki."
        )

with tab_team:
    if "stats" in st.session_state:
        p = st.session_state["sim_params"]
        st.caption(
            f"Parametry: lambda_base={p['lambda_base']}, k={p['k']}, "
            f"n={p['n']:,} symulacji"
        )
        display_team_info(st.session_state["stats"])
    elif not run_btn:
        st.info(
            "Ustaw parametry w panelu bocznym i kliknij **▶ Uruchom symulację**, "
            "aby zobaczyć szczegółowe statystyki drużyn."
        )
