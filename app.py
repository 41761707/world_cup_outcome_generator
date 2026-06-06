import io
import os
import sys
import contextlib
from collections import defaultdict
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from main import run_monte_carlo
from wc_logic import get_countries, load_knockout_schedule

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
def load_initial_data():
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
    return groups_data, group_schedule, knockout_raw

def _fmt_slot(slot) -> str:
    kind = slot[0]
    if kind == "winner":
        return f"Wyg. {slot[1]}"
    if kind == "loser":
        return f"Prz. {slot[1]}"
    _, groups, pos = slot
    suffix = {1: "1.", 2: "2.", 3: "3."}.get(pos, f"{pos}.")
    label = f"Gr. {groups[0]}" if len(groups) == 1 else f"Gr. {'/'.join(groups)}"
    return f"{suffix} {label}"

def _group_by_round(knockout_raw):
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

#TODO: CSS do osobnego pliku
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

def display_groups(groups_data: dict, last_standings: dict | None = None):
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

#TODO: CSS do osobnego pliku
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

def display_empty_bracket(knockout_raw):
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
                col.markdown(_match_card(mid, _fmt_slot(s1), _fmt_slot(s2)), unsafe_allow_html=True)

def display_last_bracket(last_bracket: dict):
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
                col.markdown(_match_card_result(m["match_id"], m), unsafe_allow_html=True)

def display_results(stats: dict):
    n = stats["n_simulations"]
    team_exit = stats["team_exit_stages"]
    # 1. Szansa na wygranie turnieju
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
        .sort_values("Procent", ascending=False)
        .reset_index(drop=True)
    )
    fig_win = px.bar(
        winner_df,
        x="Drużyna",
        y="Procent",
        color="Procent",
        color_continuous_scale="YlOrRd",
        text="Procent",
        labels={"Procent": "Prawdopodobieństwo (%)"},
    )
    fig_win.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_win.update_layout(
        coloraxis_showscale=False,
        yaxis_title="Prawdopodobieństwo (%)",
        xaxis_tickangle=-40,
        margin=dict(b=100),
    )
    st.plotly_chart(fig_win, use_container_width=True)
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
            .sort_values("Procent", ascending=False)
            .reset_index(drop=True)
        )
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
        st.plotly_chart(fig, use_container_width=True)
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
    fig_heat = px.imshow(
        heat_df,
        color_continuous_scale="Blues",
        labels=dict(color="%"),
        aspect="auto",
        text_auto=".1f",
    )
    fig_heat.update_layout(
        height=max(500, len(all_teams) * 22),
        xaxis_title="Etap turnieju",
        yaxis_title=None,
        coloraxis_colorbar=dict(title="%"),
        margin=dict(l=160),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    # 4. Najczęstsze spotkania w fazie pucharowej
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
    # 5. Szczegóły wybranej drużyny
    st.subheader("🔍 Szczegóły wybranej drużyny")
    selected = st.selectbox("Wybierz drużynę:", sorted(team_exit.keys()), key="team_detail_select")
    if selected:
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

st.set_page_config(
    page_title="Symulator MŚ",
    page_icon="⚽",
    layout="wide",
)
st.title("⚽ Symulator Mistrzostw Świata")
st.caption("Analiza Monte Carlo — rozkłady prawdopodobieństwa wyników wszystkich drużyn")
groups_data, group_schedule, knockout_raw = load_initial_data()
# Pasek boczny
with st.sidebar:
    st.header("⚙️ Parametry symulacji")
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
        help="Jak mocno różnica ELO skaluje oczekiwane gole",
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
    run_btn = st.button("▶ Uruchom symulację", type="primary", use_container_width=True)

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
            )
        st.session_state["stats"] = _stats
        st.session_state["sim_params"] = {
            "n": int(n_simulations),
            "lambda_base": float(lambda_base),
            "k": float(k),
        }
    st.success(f"✅ Ukończono {int(n_simulations):,} symulacji!")

# Zakładki
tab_groups, tab_bracket, tab_results, tab_info = st.tabs(
    ["📋 Grupy", "🗂️ Drabinka", "📊 Wyniki symulacji", "ℹ️ Opis działania"]
)

with tab_groups:
    # --- Expander: własne wyniki meczów grupowych ---
    with st.expander("✏️ Wprowadź własne wyniki meczów grupowych (opcjonalnie)", expanded=False):
        st.caption(
            "Wypełnij wyniki dla wybranych meczów — zostaną one użyte we wszystkich symulacjach jako stały wynik. "
            "Pozostawienie pól pustych oznacza, że mecz zostanie rozegrany losowo."
        )
        fixed_inputs: dict[tuple[str, str], tuple[int, int]] = {}
        for gname in sorted(group_schedule.keys()):
            st.markdown(f"**Grupa {gname}**")
            for home, away in group_schedule[gname]:
                col_home, col_s1, col_dash, col_s2, col_away = st.columns([3, 1, 0.3, 1, 3])
                col_home.markdown(f"<div style='padding-top:6px;text-align:right'>{home}</div>", unsafe_allow_html=True)
                s1 = col_s1.number_input(
                    "g1", min_value=0, max_value=30, value=None,
                    key=f"fg_{gname}_{home}_{away}_1", label_visibility="collapsed"
                )
                col_dash.markdown("<div style='padding-top:6px;text-align:center'>–</div>", unsafe_allow_html=True)
                s2 = col_s2.number_input(
                    "g2", min_value=0, max_value=30, value=None,
                    key=f"fg_{gname}_{home}_{away}_2", label_visibility="collapsed"
                )
                col_away.markdown(f"<div style='padding-top:6px'>{away}</div>", unsafe_allow_html=True)
                if s1 is not None and s2 is not None:
                    fixed_inputs[(home, away)] = (int(s1), int(s2))
        st.session_state["fixed_group_results"] = fixed_inputs if fixed_inputs else None
        n_fixed = len(fixed_inputs)
        if n_fixed:
            st.success(f"Zablokowano {n_fixed} {'mecz' if n_fixed == 1 else 'mecze' if n_fixed in (2, 3, 4) else 'meczów'}.")

    _lb = st.session_state.get("stats", {}).get("last_bracket")
    if _lb:
        st.subheader("Wyniki fazy grupowej — ostatnie losowanie")
        st.caption("Tabela końcowa z ostatniej przeprowadzonej symulacji (Pkt = punkty, RB = różnica bramek, G = bramki strzelone).")
        display_groups(groups_data, last_standings=_lb["groups"])
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
        display_groups(groups_data)

with tab_bracket:
    st.subheader("Drabinka fazy pucharowej")
    if "stats" in st.session_state and st.session_state["stats"].get("last_bracket"):
        st.info(
            "📌 Poniżej pokazano wyniki **ostatniego losowania** (ostatniej z przeprowadzonych symulacji). "
            "Każde uruchomienie symulacji może dać inny wynik — to tylko jedna z możliwych wersji turnieju."
        )
        display_last_bracket(st.session_state["stats"]["last_bracket"])
    else:
        st.caption("Drabinka pokazuje zaplanowane mecze — drużyny zostaną wylosowane po fazie grupowej. "
                   "Po uruchomieniu symulacji zobaczysz tutaj wyniki ostatniego losowania.")
        display_empty_bracket(knockout_raw)

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
