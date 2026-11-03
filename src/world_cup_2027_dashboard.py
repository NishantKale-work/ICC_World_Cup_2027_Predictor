from pathlib import Path
from io import BytesIO
import sys
import itertools
import math

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# ICC Men's Cricket World Cup 2027 — Interactive Predictor
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_RESULTS = PROJECT_ROOT / "reports" / "world_cup_2027_monte_carlo_results.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "venue_logistic_regression_2027.joblib"
CUTOFF = "2026-10-25"

GROUPS = {
    "A": ["India", "Australia", "Pakistan", "Afghanistan", "Zimbabwe", "West Indies"],
    "B": ["New Zealand", "South Africa", "Sri Lanka", "England", "Bangladesh", "Ireland"],
}
VENUES = {
    "South Africa": ["Bloemfontein", "Cape Town", "Centurion", "Durban", "KuGompo City",
                     "Gqeberha", "Johannesburg", "Paarl"],
    "Zimbabwe": ["Bulawayo", "Harare", "Victoria Falls"],
    "Namibia": ["Windhoek"],
}
VENUE_COUNTRY = {city: country for country, cities in VENUES.items() for city in cities}
ALL_CITIES = list(VENUE_COUNTRY)

FLAGS = {
    "India":"🇮🇳","Australia":"🇦🇺","Pakistan":"🇵🇰","Afghanistan":"🇦🇫",
    "Zimbabwe":"🇿🇼","New Zealand":"🇳🇿","South Africa":"🇿🇦","Sri Lanka":"🇱🇰",
    "England":"🏴","Bangladesh":"🇧🇩","West Indies":"WI","Scotland":"🏴",
    "Ireland":"🇮🇪","Netherlands":"🇳🇱",
}
FLAG_CODES = {
    "India":"in", "Australia":"au", "Pakistan":"pk", "Afghanistan":"af",
    "Zimbabwe":"zw", "New Zealand":"nz", "South Africa":"za", "Sri Lanka":"lk",
    "England":"gb-eng", "Bangladesh":"bd", "Scotland":"gb-sct",
    "Ireland":"ie", "Netherlands":"nl"
}
def flag(team):
    return FLAGS.get(team, "🏏")

def flag_html(team):
    # Ireland uses its national flag; West Indies is a multi-country team,
    # so show a branded maroon WI badge instead of a misleading country flag.
    if team == "West Indies":
        return '<span class="team-logo team-logo-wi" aria-label="West Indies team logo">WI</span>'
    code = FLAG_CODES.get(team, "")
    if code:
        return f'<img class="country-flag" src="https://flagcdn.com/w40/{code}.png" alt="{team} flag">'
    return '<span class="team-flag">🏏</span>'

st.set_page_config(page_title="World Cup 2027 Predictor", page_icon="🏆", layout="wide")

st.markdown("""
<style>
:root{--gold:#f5c45e;--gold2:#ffd982;--ink:#090f1b;--panel:#152238;--panel2:#1d2d46;--line:#344761;--muted:#9eacc1;--text:#edf2f8}
.stApp{background:radial-gradient(ellipse at 75% -10%,#2a3d59 0%,#111c2e 42%,#080e19 100%);color:var(--text)}
.block-container{max-width:1500px;padding-top:1.2rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:#080f1c;border-right:1px solid #2b3c54;min-width:290px!important}
[data-testid="stSidebar"] *{font-size:1.04rem}
h1,h2,h3{color:var(--gold)!important;letter-spacing:-.025em}
p,label,[data-testid="stCaptionContainer"]{color:#c6d1e0}
[data-testid="stMetric"]{background:linear-gradient(145deg,#1a2a42,#22344e);border:1px solid #3b506d;padding:17px 19px;border-radius:16px}
[data-testid="stMetricLabel"]{color:#b9c8dc}
[data-testid="stMetricValue"]{color:#f5c45e}
.stButton>button{background:#080b10;color:#fff;border:1px solid #46536a;border-radius:11px;font-weight:750;padding:.72rem 1.1rem;transition:.15s}
.stButton>button:hover{background:#1c2635;color:#fff;border-color:var(--gold);transform:translateY(-1px)}
button[kind="primary"]{background:var(--gold)!important;color:#101722!important;border:0!important}
div[data-baseweb="select"]>div{background:#edf0f5!important;color:#172033!important;border-radius:9px}
[data-testid="stTabs"]{margin-top:18px;margin-bottom:26px;width:100%}
[data-testid="stTabs"] [data-baseweb="tab-list"]{display:flex;gap:14px;border-bottom:2px solid #344761;width:100%;padding:0 0 8px;flex-wrap:wrap}
[data-testid="stTabs"] button{color:#e2eaf5;font-size:1.25rem!important;font-weight:850!important;padding:18px 28px!important;min-height:68px;border-radius:12px 12px 0 0;background:#1b2b43;flex:1 1 260px;justify-content:center}
[data-testid="stTabs"] button[aria-selected="true"]{color:#101722!important;background:var(--gold)!important;border-bottom:4px solid var(--gold);box-shadow:0 6px 18px #0003}
[data-testid="stTabs"] button p{font-size:1.25rem!important;font-weight:850!important}
.predict-title{font-size:clamp(2.5rem,4vw,3.5rem);font-weight:950;letter-spacing:-.035em;color:#f5c45e;margin:.45rem 0 .75rem;line-height:1.12}
.predict-subtitle{font-size:clamp(1.65rem,2.6vw,2.15rem);font-weight:900;color:#f2d08a;margin:1rem 0 .65rem;line-height:1.2}
.country-flag{width:30px;height:21px;object-fit:cover;border-radius:3px;vertical-align:middle;box-shadow:0 1px 4px #0007;margin-right:8px}
.team-flag{font-size:1.2rem;margin-right:8px}
.team-logo{display:inline-flex;align-items:center;justify-content:center;width:30px;height:21px;border-radius:3px;margin-right:8px;font-size:10px;font-weight:900;letter-spacing:.4px;vertical-align:middle;box-shadow:0 1px 4px #0007}
.team-logo-wi{background:linear-gradient(135deg,#7b1026,#3c0712);color:#f5c45e;border:1px solid #d4a94f}
.team-logo-ie{background:linear-gradient(90deg,#169b62 0 33%,#fff 33% 66%,#ff883e 66% 100%);color:#172033;border:1px solid #d8e0eb}
/* Stable select styling: leave BaseWeb's chevron in its native layout.
   Avoid absolute positioning / flex overrides that caused the arrow to drift. */
div[data-baseweb="select"]>div{
    min-height:44px!important;
    background:#edf0f5!important;
    border-radius:9px!important;
    box-sizing:border-box!important;
}
div[data-baseweb="select"] [role="combobox"]{
    min-height:44px!important;
    display:flex!important;
    align-items:center!important;
    box-sizing:border-box!important;
    padding-top:0!important;
    padding-bottom:0!important;
}
div[data-baseweb="select"] [role="combobox"] svg{
    position:static!important;
    transform:none!important;
    display:block!important;
    margin:0!important;
}
.match-card{padding:22px 24px}
.section-kicker{font-size:.86rem;letter-spacing:.17em}
[data-testid="stDataFrame"]{border:1px solid #344761;border-radius:12px;overflow:hidden}
hr{border-color:#2c3e57}
.hero{position:relative;border:1px solid #4a3c2b;border-radius:22px;overflow:hidden;padding:52px 48px 44px;margin-bottom:22px;background:linear-gradient(90deg,rgba(8,13,24,.96) 0%,rgba(8,13,24,.78) 48%,rgba(8,13,24,.25) 100%),linear-gradient(135deg,#4a1019,#141b2b 55%,#7a4b18);min-height:210px}
.hero-kicker{font-size:.82rem;letter-spacing:.18em;text-transform:uppercase;color:#f7d78c;font-weight:800}
.hero-title{font-size:clamp(2rem,4vw,3.35rem);line-height:1.06;font-weight:850;color:#fff;margin:.55rem 0}
.hero-copy{font-size:1rem;color:#d5deeb;max-width:650px}
.section-kicker{color:#f2d08a;text-transform:uppercase;letter-spacing:.15em;font-size:.75rem;font-weight:800}
.match-card{background:linear-gradient(145deg,#1b2b43,#111c2e);border:1px solid #344a66;border-radius:16px;padding:18px;margin:8px 0}
.team-chip{display:inline-flex;align-items:center;gap:9px;background:#233650;border:1px solid #405572;border-radius:999px;padding:7px 12px;font-weight:700}
.stage-pill{display:inline-block;border:1px solid #465b76;border-radius:999px;padding:4px 10px;color:#d5dfec;font-size:.8rem}
.muted{color:#9eacc1;font-size:.88rem}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_results(path, mtime):
    del mtime
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_uploaded(data):
    return pd.read_csv(BytesIO(data))

def validate_results(df):
    required = ["team","qualifier_a_probability","qualifier_b_probability","group_stage_probability",
                "super7_probability","semi_final_probability","final_probability","champion_probability"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Results CSV missing columns: {missing}")
    df = df.copy()
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df["team"].isna().any() or df["team"].duplicated().any():
        raise ValueError("Team names must be present and unique.")
    if df[required[1:]].isna().any().any():
        raise ValueError("Probability columns contain missing/non-numeric values.")
    if ((df[required[1:]] < 0) | (df[required[1:]] > 1)).any().any():
        raise ValueError("Probabilities must be between 0 and 1.")
    return df

@st.cache_resource(show_spinner="Loading ODI model...")
def load_prediction_assets():
    import build_2027_match_features as fb
    history = fb.load_historical_data()
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")
    artifact = joblib.load(MODEL_FILE)
    return fb, history, artifact

def predict_match(team1, team2, city, country, assets):
    fb, history, artifact = assets
    cutoff = pd.Timestamp(CUTOFF)
    s1 = fb.get_latest_team_state(history, team1, cutoff)
    s2 = fb.get_latest_team_state(history, team2, cutoff)
    hist = history[history["date"] <= cutoff]
    city_key, country_key = fb.normalize_text(city), fb.normalize_country(country)
    def venue_count(team):
        mask = (hist["team_1"].str.casefold().eq(team.casefold()) |
                hist["team_2"].str.casefold().eq(team.casefold()))
        return int((mask & hist["city_resolved"].map(fb.normalize_text).eq(city_key) &
                    hist["country_resolved"].map(fb.normalize_country).eq(country_key)).sum())
    v1, v2 = venue_count(team1), venue_count(team2)
    h1, h2 = fb.is_home_team(team1, country), fb.is_home_team(team2, country)
    row = {
        "team_1_elo":s1["elo"],"team_2_elo":s2["elo"],"elo_difference":s1["elo"]-s2["elo"],
        "team_1_recent_form":s1["recent_form"],"team_2_recent_form":s2["recent_form"],
        "recent_form_difference":s1["recent_form"]-s2["recent_form"],
        "team_1_matches_played":s1["matches_played"],"team_2_matches_played":s2["matches_played"],
        "team_1_venue_matches":v1,"team_2_venue_matches":v2,"venue_familiarity_difference":v1-v2,
        "team_1_home":h1,"team_2_home":h2,"home_advantage_difference":h1-h2,
    }
    features = pd.DataFrame([row])[artifact["feature_columns"]]
    p1 = float(artifact["model"].predict_proba(features)[0,1])
    cold = [t for t,s in [(team1,s1),(team2,s2)] if s.get("state_source")=="COLD_START_ASSUMPTION"]
    return p1, row, cold

def fixture_list(group, teams):
    return [(f"{group}-{i:02d}", a, b) for i,(a,b) in enumerate(itertools.combinations(teams,2),1)]

def init_state():
    defaults = {
        "group_picks": {}, "super7_teams": [], "semi_picks": {}, "final_pick": None,
        "predictor_reset_key": 0, "match_probability_table": None
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def render_team(team, large=False):
    size = "1.25rem" if large else "1rem"
    st.markdown(f'<span style="font-size:{size};font-weight:750">{flag_html(team)} {team}</span>', unsafe_allow_html=True)

def calculate_standings(group, teams):
    rows = {t:{"Team":t,"P":0,"W":0,"L":0,"Pts":0} for t in teams}
    for match_id,a,b in fixture_list(group,teams):
        winner = st.session_state.group_picks.get(match_id)
        if winner not in (a,b): continue
        loser = b if winner==a else a
        rows[a]["P"]+=1; rows[b]["P"]+=1
        rows[winner]["W"]+=1; rows[loser]["L"]+=1
        rows[winner]["Pts"]+=2
    return sorted(rows.values(),key=lambda r:(r["Pts"],r["W"],r["Team"]),reverse=True)

def reset_bracket():
    st.session_state.group_picks={}
    st.session_state.super7_teams=[]
    st.session_state.semi_picks={}
    st.session_state.final_pick=None

# Data options tucked away; never dominates the interface.
with st.expander("⚙️ Data settings", expanded=False):
    uploaded = st.file_uploader("Upload results CSV", type=["csv"])
    st.caption("The dashboard reads existing simulation results. It does not rerun the tournament simulation.")

try:
    if uploaded:
        results=load_uploaded(uploaded.getvalue())
    elif DEFAULT_RESULTS.exists():
        results=load_results(str(DEFAULT_RESULTS),DEFAULT_RESULTS.stat().st_mtime)
    else:
        raise FileNotFoundError(f"Results CSV not found: {DEFAULT_RESULTS}")
    results=validate_results(results)
except Exception as e:
    st.error(str(e)); st.stop()

ordered=results.sort_values("champion_probability",ascending=False).reset_index(drop=True)
champ=ordered.iloc[0]
teams=sorted(results.team.tolist())
init_state()

# Hero banner
st.markdown("""
<div class="hero">
  <div class="hero-kicker">South Africa · Zimbabwe · Namibia</div>
  <div class="hero-title">ICC Men's Cricket<br>World Cup 2027</div>
  <div class="hero-copy">Explore the model's match intelligence, then build your own route to the trophy.</div>
</div>
""", unsafe_allow_html=True)

# Tournament overview
st.markdown('<div class="section-kicker">Tournament dashboard</div>',unsafe_allow_html=True)
st.subheader("The race for the trophy")
m1,m2,m3,m4=st.columns(4)
m1.metric("Model favourite",f"{flag(champ['team'])} {champ['team']}",f"{champ['champion_probability']:.1%} title chance")
m2.metric("Top two contenders",f"{ordered.head(2)['champion_probability'].sum():.1%}","combined title chance")
m3.metric("Teams in the model",str(len(results)))
m4.metric("Title probability total",f"{results.champion_probability.sum():.1%}")

left,right=st.columns([1.1,1])
with left:
    st.markdown("#### Contenders")
    chart=ordered.head(8).copy()
    chart["Team label"]=chart.team.map(lambda t:f"{flag(t)}  {t}")
    chart["Title chance"]=chart.champion_probability*100
    fig=px.bar(chart,x="Title chance",y="Team label",orientation="h",
               text=chart["Title chance"].map(lambda x:f"{x:.1f}%"),
               color="Title chance",color_continuous_scale=["#405d82","#f5c45e"])
    fig.update_yaxes(categoryorder="total ascending",title=None)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                     font_color="#dbe6dc",height=390,margin=dict(l=8,r=12,t=12,b=10))
    st.plotly_chart(fig,use_container_width=True)
with right:
    st.markdown("#### A team's road to the final")
    selected=st.selectbox("Choose a team",ordered.team.tolist(),key="overview_team")
    r=results.loc[results.team.eq(selected)].iloc[0]
    stages=["Group","Super 7","Semi-final","Final","Champion"]
    values=[r.group_stage_probability,r.super7_probability,r.semi_final_probability,r.final_probability,r.champion_probability]
    fig=go.Figure(go.Bar(x=stages,y=[v*100 for v in values],text=[f"{v:.1%}" for v in values],
                         textposition="outside",marker_color=["#536f94","#6f8eb1","#9bb1c8","#d4b46c","#f5c45e"]))
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                     font_color="#dbe6dc",height=390,margin=dict(l=8,r=12,t=12,b=10),yaxis=dict(range=[0,108],title=None))
    st.plotly_chart(fig,use_container_width=True)

st.divider()

# Both experiences live below the dashboard.
tournament_tab, bracket_tab, match_tab, probabilities_tab = st.tabs([
    "🏆 Tournament", "🧭 Predict Your Tournament", "🏏 Match Intelligence", "📊 Match-wise Probabilities"
])

with tournament_tab:
    st.markdown('<div class="predict-title">Tournament Outlook</div>', unsafe_allow_html=True)
    st.markdown('<div class="predict-subtitle">The model’s current view of the race</div>', unsafe_allow_html=True)
    st.caption("These are tournament-stage probabilities from the saved Monte Carlo simulation. Your personal bracket is built separately in Predict Your Tournament.")
    top = ordered.head(8).copy()
    top["Champion chance"] = top["champion_probability"].map(lambda x: f"{x:.1%}")
    top["Final chance"] = top["final_probability"].map(lambda x: f"{x:.1%}")
    top["Semi-final chance"] = top["semi_final_probability"].map(lambda x: f"{x:.1%}")
    st.dataframe(top[["team", "Champion chance", "Final chance", "Semi-final chance"]], hide_index=True, use_container_width=True)

with match_tab:
    st.markdown('<div class="section-kicker">Model estimates</div>',unsafe_allow_html=True)
    st.markdown('<div class="predict-subtitle">Match-wise probability</div>',unsafe_allow_html=True)
    st.caption("Compare two teams at a selected venue. These are model estimates—not your tournament picks.")
    c1,c2=st.columns(2)
    with c1:
        team1=st.selectbox("Team A",teams,index=teams.index("India") if "India" in teams else 0,key="mi_team1")
    with c2:
        default2=teams.index("Australia") if "Australia" in teams else min(1,len(teams)-1)
        team2=st.selectbox("Team B",teams,index=default2,key="mi_team2")
    venue=st.selectbox("Venue",ALL_CITIES,index=ALL_CITIES.index("Cape Town"),key="mi_venue")
    country=VENUE_COUNTRY[venue]
    if team1==team2:
        st.warning("Select two different teams.")
    elif st.button("⚡ Predict match",type="primary",use_container_width=True,key="mi_predict"):
        try:
            assets=load_prediction_assets()
            p1,features,cold=predict_match(team1,team2,venue,country,assets)
            p2=1-p1
            st.session_state["last_match_prediction"]={"team1":team1,"team2":team2,"venue":venue,"p1":p1,"p2":p2,"cold":cold}
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Check that the trained model and historical feature files are present.")
    pred=st.session_state.get("last_match_prediction")
    if pred:
        a,b=st.columns(2)
        with a:
            st.markdown(f'<div class="match-card"><div class="muted">TEAM A</div><div style="font-size:1.45rem;font-weight:800">{flag(pred["team1"])} {pred["team1"]}</div><div style="font-size:2rem;color:#f5c45e;font-weight:850">{pred["p1"]:.1%}</div><div class="muted">model win probability</div></div>',unsafe_allow_html=True)
        with b:
            st.markdown(f'<div class="match-card"><div class="muted">TEAM B</div><div style="font-size:1.45rem;font-weight:800">{flag(pred["team2"])} {pred["team2"]}</div><div style="font-size:2rem;color:#f5c45e;font-weight:850">{pred["p2"]:.1%}</div><div class="muted">model win probability</div></div>',unsafe_allow_html=True)
        fig=go.Figure(go.Bar(x=[pred["p1"]*100,pred["p2"]*100],y=[pred["team1"],pred["team2"]],orientation="h",
                             marker_color=["#f5c45e","#7397bd"],text=[f'{pred["p1"]:.1%}',f'{pred["p2"]:.1%}'],textposition="inside"))
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                         height=220,margin=dict(l=8,r=8,t=12,b=8),xaxis=dict(range=[0,100],title=None))
        st.plotly_chart(fig,use_container_width=True)
        st.caption(f'Venue: {pred["venue"]}, {VENUE_COUNTRY[pred["venue"]]} · Historical data through {CUTOFF}')
        if pred["cold"]: st.warning("Cold-start priors used for "+", ".join(pred["cold"])+". Higher uncertainty.")
        st.caption("Model estimates do not include toss, confirmed XI, injuries, or match-day conditions.")


with probabilities_tab:
    st.markdown('<div class="section-kicker">MODEL ESTIMATES</div>', unsafe_allow_html=True)
    st.markdown('<div class="predict-title">Match-by-match win probabilities</div>', unsafe_allow_html=True)
    st.markdown('<div class="predict-subtitle">Explore pairings by group</div>', unsafe_allow_html=True)
    st.caption("Each pairing is estimated by the trained match model and averaged across the tournament venue pool. These are model estimates, not outcomes from individual tournament simulations.")

    selected_group = st.radio("Select group", ["A", "B"], horizontal=True, key="probability_group")
    group_teams = GROUPS[selected_group]
    cache_key = f"match_probability_table_{selected_group}"

    # Keep each group's results separately so switching groups never blanks or
    # overwrites the other group's calculated table.
    if st.button("Calculate group probabilities", type="primary", use_container_width=True,
                 key=f"calculate_probabilities_{selected_group}"):
        try:
            assets = load_prediction_assets()
            fixtures = fixture_list(selected_group, group_teams)
            rows = []
            progress = st.progress(0, text="Preparing match estimates…")
            total = len(fixtures) * len(ALL_CITIES)
            completed = 0
            for _, team_a, team_b in fixtures:
                venue_probs = []
                cold_teams = set()
                for venue_name in ALL_CITIES:
                    p_a, _, cold = predict_match(
                        team_a, team_b, venue_name, VENUE_COUNTRY[venue_name], assets
                    )
                    venue_probs.append(p_a)
                    cold_teams.update(cold)
                    completed += 1
                    progress.progress(completed / total,
                                      text=f"Estimating {team_a} vs {team_b} · {completed}/{total}")
                p_a = float(sum(venue_probs) / len(venue_probs))
                rows.append({
                    "Group": selected_group,
                    "Team A": team_a,
                    "Team A win probability": p_a,
                    "Team B": team_b,
                    "Team B win probability": 1 - p_a,
                    "Cold start": "Yes" if cold_teams else "No",
                })
            st.session_state[cache_key] = pd.DataFrame(rows)
            progress.empty()
            st.success(f"Group {selected_group} estimates are ready.")
        except Exception as e:
            st.error(f"Could not calculate Group {selected_group} probabilities: {e}")
            st.info("Check that the trained model and historical feature files exist in their expected project folders.")

    probability_df = st.session_state.get(cache_key)
    if probability_df is not None and not probability_df.empty:
        st.markdown(f"#### Group {selected_group} · 15 pairings")
        st.dataframe(
            probability_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Team A win probability": st.column_config.ProgressColumn(
                    "Team A win probability", format="%.1f%%", min_value=0, max_value=1
                ),
                "Team B win probability": st.column_config.ProgressColumn(
                    "Team B win probability", format="%.1f%%", min_value=0, max_value=1
                ),
            },
        )
        st.download_button(
            "Download group probabilities CSV",
            probability_df.to_csv(index=False).encode("utf-8"),
            f"world_cup_2027_group_{selected_group.lower()}_match_probabilities.csv",
            "text/csv",
            key=f"download_probabilities_{selected_group}",
        )
    else:
        st.info(f"Select Group {selected_group} and calculate to see all 15 pairings. West Indies and Ireland are included as your provisional qualifier selections.")

with bracket_tab:
    st.markdown('<div class="section-kicker">Your picks · your bracket</div>',unsafe_allow_html=True)
    st.markdown('<div class="predict-title">🏆 Predict Your Tournament</div>',unsafe_allow_html=True)
    st.markdown('<div class="predict-subtitle">Build your route to the trophy</div>',unsafe_allow_html=True)
    st.caption("Choose match winners. Your selections are retained in this session and build your personal tournament bracket.")
    progress=sum(1 for v in st.session_state.group_picks.values() if v)
    st.progress(progress/30, text=f"Group-stage picks · {progress} of 30 completed")
    group_tabs=st.tabs(["Group A","Group B"])
    for tab,group in zip(group_tabs,["A","B"]):
        with tab:
            gteams=GROUPS[group]
            for match_id,a,b in fixture_list(group,gteams):
                with st.container(border=True):
                    col1,col2,col3=st.columns([2,2,2])
                    with col1: render_team(a)
                    with col2: render_team(b)
                    with col3:
                        options=["— choose winner —",a,b]
                        current=st.session_state.group_picks.get(match_id)
                        idx=options.index(current) if current in options else 0
                        pick=st.selectbox("Winner",options,index=idx,key=f"pick_{match_id}",label_visibility="collapsed")
                        if pick=="— choose winner —":
                            st.session_state.group_picks.pop(match_id,None)
                        else:
                            st.session_state.group_picks[match_id]=pick
            standings=calculate_standings(group,gteams)
            st.markdown("#### Your predicted standings")
            if any(row["P"] for row in standings):
                for rank,row in enumerate(standings,1):
                    st.markdown(f'<div class="match-card" style="padding:10px 14px;display:flex;justify-content:space-between"><span>{rank}. {flag(row["Team"])} <b>{row["Team"]}</b></span><span>{row["Pts"]} pts · {row["W"]}W</span></div>',unsafe_allow_html=True)
            else:
                st.caption("Standings will appear as you make picks.")

    all_complete=len(st.session_state.group_picks)==30
    if all_complete:
        standings_a=calculate_standings("A",GROUPS["A"])
        standings_b=calculate_standings("B",GROUPS["B"])
        top3=[r["Team"] for r in standings_a[:3]]+[r["Team"] for r in standings_b[:3]]
        fourth=[standings_a[3],standings_b[3]]
        best_fourth=sorted(fourth,key=lambda r:(r["Pts"],r["W"],r["Team"]),reverse=True)[0]["Team"]
        generated=top3+[best_fourth]
        # Keep user edits when the same group outcomes remain; otherwise reset downstream bracket.
        if st.session_state.super7_teams and set(st.session_state.super7_teams)!=set(generated):
            st.session_state.super7_teams=[]
            st.session_state.semi_picks={}
            st.session_state.final_pick=None
        if not st.session_state.super7_teams:
            st.session_state.super7_teams=generated
        st.divider()
        st.markdown('<div class="section-kicker">Stage 2</div>',unsafe_allow_html=True)
        st.subheader("Your Super 7")
        st.caption("Automatically formed from your group predictions: top three from each group plus the better fourth-place team.")
        cols=st.columns(4)
        for i,t in enumerate(st.session_state.super7_teams):
            with cols[i%4]:
                st.markdown(f'<div class="match-card"><div class="muted">SUPER 7 TEAM {i+1}</div><div style="font-size:1.1rem;font-weight:800">{flag(t)} {t}</div></div>',unsafe_allow_html=True)
        st.markdown("#### Choose your four semi-finalists")
        chosen=st.multiselect("Select exactly four teams",st.session_state.super7_teams,
                              default=st.session_state.super7_teams[:4],max_selections=4,key="semi_teams")
        if len(chosen)==4:
            st.session_state.super7_teams=generated
            st.session_state["chosen_semis"]=chosen
            st.markdown('<div class="section-kicker">Stage 3</div>',unsafe_allow_html=True)
            st.subheader("Semi-finals")
            semi1,semi2=st.columns(2)
            with semi1:
                st.markdown("**Semi-final 1**")
                s1a, s1b = chosen[0], chosen[1]
                st.markdown(f'<div class="match-card">{flag(s1a)} <b>{s1a}</b> &nbsp; <span style="color:#f5c45e">VS</span> &nbsp; {flag(s1b)} <b>{s1b}</b></div>',unsafe_allow_html=True)
                sf1=st.radio("Winner of semi-final 1",[s1a,s1b],horizontal=True,key="sf1winner")
            with semi2:
                st.markdown("**Semi-final 2**")
                s2a, s2b = chosen[2], chosen[3]
                st.markdown(f'<div class="match-card">{flag(s2a)} <b>{s2a}</b> &nbsp; <span style="color:#f5c45e">VS</span> &nbsp; {flag(s2b)} <b>{s2b}</b></div>',unsafe_allow_html=True)
                sf2=st.radio("Winner of semi-final 2",[s2a,s2b],horizontal=True,key="sf2winner")
            st.markdown('<div class="section-kicker">Stage 4</div>',unsafe_allow_html=True)
            st.subheader("The Final")
            finalists=[sf1,sf2]
            st.markdown(f'<div class="match-card" style="text-align:center"><div style="font-size:1.4rem;font-weight:850">{flag(sf1)} {sf1} &nbsp; <span style="color:#f5c45e">VS</span> &nbsp; {flag(sf2)} {sf2}</div><div class="muted">Choose your World Cup champion</div></div>',unsafe_allow_html=True)
            champion_pick=st.radio("Champion",finalists,horizontal=True,key="champion_choice")
            st.success(f"Your predicted champion: {flag(champion_pick)} **{champion_pick}**")
    else:
        st.info("Complete all 30 group-stage picks to unlock your Super 7 and knockout bracket.")

    if st.button("↺ Reset my tournament picks",key="reset_bracket"):
        reset_bracket()
        st.rerun()

st.divider()
st.markdown('<p class="muted">A fan prediction experience powered by historical ODI model estimates. Provisional tournament format and qualifier assumptions; not betting advice.</p>',unsafe_allow_html=True)
