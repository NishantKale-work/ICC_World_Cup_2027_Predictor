"""
ICC Men's Cricket World Cup 2027 — Monte Carlo Simulator

Uses:
- config/world_cup_2027_format.py
- src/build_2027_match_features.py
- models/venue_logistic_regression_2027.joblib

Important assumptions:
1. The four working qualifiers play a provisional round-robin; top team qualifies
   directly to Group A, remaining three play a three-team round-robin Super Series.
2. Group-stage and Super 7 standings use points (win=2) plus a randomized
   tie-break proxy. This is NOT official NRR.
3. Match outcomes are sampled from the trained model's predicted win probability.
4. Results are scenario estimates, not reliable tournament forecasts.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
REPORTS_DIR = PROJECT_ROOT / "reports"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

import build_2027_match_features as feature_builder
import world_cup_2027_format as tournament


# ------------------------------------------------------------
# Fixture venue lookup
# ------------------------------------------------------------

# Cities/countries are based on the venue list supplied for this project.
# Match scheduling is represented as a venue pool by stage, not an official
# match-by-match fixture reproduction.
SOUTH_AFRICA_VENUES = [
    ("Bloemfontein", "South Africa"),
    ("Cape Town", "South Africa"),
    ("Centurion", "South Africa"),
    ("Durban", "South Africa"),
    ("KuGompo City", "South Africa"),
    ("Gqeberha", "South Africa"),
    ("Johannesburg", "South Africa"),
    ("Paarl", "South Africa"),
]
ZIMBABWE_VENUES = [
    ("Bulawayo", "Zimbabwe"),
    ("Harare", "Zimbabwe"),
    ("Victoria Falls", "Zimbabwe"),
]
NAMIBIA_VENUE = ("Windhoek", "Namibia")


def venue_for_stage(stage: str, rng: np.random.Generator) -> tuple[str, str]:
    if stage in {"qualifier", "super_series"}:
        return NAMIBIA_VENUE
    if stage == "group":
        # Group-stage matches are assigned to the South African/Zimbabwean
        # venue pool; Namibia is excluded after the Super Series.
        return SOUTH_AFRICA_VENUES + ZIMBABWE_VENUES[
            int(rng.integers(0, len(ZIMBABWE_VENUES)))
        : int(rng.integers(0, len(ZIMBABWE_VENUES))) + 1
        ][0] if False else tuple(
            list(SOUTH_AFRICA_VENUES)
            + list(ZIMBABWE_VENUES)
        )[int(rng.integers(0, len(SOUTH_AFRICA_VENUES) + len(ZIMBABWE_VENUES)))]
    if stage == "super7":
        return SOUTH_AFRICA_VENUES[int(rng.integers(0, len(SOUTH_AFRICA_VENUES)))]
    if stage == "semi_final_1":
        return ("Cape Town", "South Africa")
    if stage == "semi_final_2":
        return ("Centurion", "South Africa")
    if stage == "final":
        return ("Johannesburg", "South Africa")
    raise ValueError(f"Unknown tournament stage: {stage}")


class WorldCupSimulator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.history = feature_builder.load_historical_data()

        if not feature_builder.MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Trained venue model not found: {feature_builder.MODEL_FILE}"
            )

        artifact = feature_builder.joblib.load(feature_builder.MODEL_FILE)
        self.model = artifact["model"]
        self.feature_columns = artifact["feature_columns"]
        self.cutoff_date = feature_builder.DEFAULT_CUTOFF_DATE

        # Precompute team snapshots and venue familiarity once. The original
        # implementation rescanned the full historical DataFrame for every
        # simulated fixture, making large Monte Carlo runs unnecessarily slow.
        cutoff = pd.Timestamp(self.cutoff_date)
        self.team_states = {
            team: feature_builder.get_latest_team_state(self.history, team, cutoff)
            for team in tournament.ALL_WORKING_TEAMS
        }
        self.venue_counts = defaultdict(int)
        history = self.history[self.history["date"] <= cutoff]
        for row in history.itertuples(index=False):
            city = feature_builder.normalize_text(row.city_resolved)
            country = feature_builder.normalize_country(row.country_resolved)
            for team in (row.team_1, row.team_2):
                self.venue_counts[
                    (str(team).casefold(), city, country)
                ] += 1

        # Report cold-start assumptions once at startup, not once per match.
        cold_start = [
            team for team, state in self.team_states.items()
            if state.get("state_source") == "COLD_START_ASSUMPTION"
        ]
        if cold_start:
            print(
                "COLD-START WARNING: no local ODI history for: "
                + ", ".join(sorted(cold_start))
            )
            print(
                "Their neutral priors are assumptions; predictions involving "
                "them have higher uncertainty."
            )

        self.progress = defaultdict(lambda: defaultdict(int))
        self.champion_counts = defaultdict(int)
        self.qualifier_a_counts = defaultdict(int)
        self.qualifier_b_counts = defaultdict(int)

    def probability(self, team1: str, team2: str, stage: str) -> float:
        city, country = venue_for_stage(stage, self.rng)
        state1 = self.team_states[team1]
        state2 = self.team_states[team2]

        city_key = feature_builder.normalize_text(city)
        country_key = feature_builder.normalize_country(country)
        venue1 = self.venue_counts.get(
            (team1.casefold(), city_key, country_key), 0
        )
        venue2 = self.venue_counts.get(
            (team2.casefold(), city_key, country_key), 0
        )
        home1 = feature_builder.is_home_team(team1, country)
        home2 = feature_builder.is_home_team(team2, country)

        row = {
            "team_1_elo": state1["elo"],
            "team_2_elo": state2["elo"],
            "elo_difference": state1["elo"] - state2["elo"],
            "team_1_recent_form": state1["recent_form"],
            "team_2_recent_form": state2["recent_form"],
            "recent_form_difference": state1["recent_form"] - state2["recent_form"],
            "team_1_matches_played": state1["matches_played"],
            "team_2_matches_played": state2["matches_played"],
            "team_1_venue_matches": venue1,
            "team_2_venue_matches": venue2,
            "venue_familiarity_difference": venue1 - venue2,
            "team_1_home": home1,
            "team_2_home": home2,
            "home_advantage_difference": home1 - home2,
        }
        features = pd.DataFrame([row])[self.feature_columns]
        p = float(self.model.predict_proba(features)[0, 1])
        return float(np.clip(p, 0.001, 0.999))

    def play(self, team1: str, team2: str, stage: str) -> str:
        p1 = self.probability(team1, team2, stage)
        return team1 if self.rng.random() < p1 else team2

    def round_robin(self, teams: list[str], stage: str) -> list[dict]:
        table = {
            team: {"team": team, "played": 0, "wins": 0, "losses": 0, "points": 0,
                   "tie_break_proxy": 0.0}
            for team in teams
        }

        for team1, team2 in itertools.combinations(teams, 2):
            winner = self.play(team1, team2, stage)
            loser = team2 if winner == team1 else team1
            table[winner]["played"] += 1
            table[winner]["wins"] += 1
            table[winner]["points"] += 2
            table[loser]["played"] += 1
            table[loser]["losses"] += 1

            # Proxy only: creates a small random separation among tied teams.
            # It is not calculated from runs, wickets, or official NRR.
            table[winner]["tie_break_proxy"] += float(self.rng.uniform(0.2, 1.0))
            table[loser]["tie_break_proxy"] -= float(self.rng.uniform(0.2, 1.0))

        rows = list(table.values())
        self.rng.shuffle(rows)
        rows.sort(
            key=lambda row: (row["points"], row["tie_break_proxy"]),
            reverse=True,
        )
        return rows

    def simulate_qualifiers(self) -> tuple[str, str]:
        qualifier_teams = list(tournament.QUALIFIER_TEAMS)

        qualifier_table = self.round_robin(qualifier_teams, "qualifier")
        direct_qualifier = qualifier_table[0]["team"]
        super_series_teams = [row["team"] for row in qualifier_table[1:]]

        super_series_table = self.round_robin(super_series_teams, "super_series")
        super_series_winner = super_series_table[0]["team"]

        return direct_qualifier, super_series_winner

    def simulate_group(self, group_name: str, teams: list[str]) -> tuple[list[str], list[str]]:
        table = self.round_robin(teams, "group")
        qualifiers = [row["team"] for row in table[:3]]
        fourth = [row["team"] for row in table[3:4]]
        return qualifiers, fourth

    def simulate_super7(self, seeds: dict[str, str]) -> list[str]:
        teams = list(seeds.values())
        table = self.round_robin(teams, "super7")
        return [row["team"] for row in table]

    def simulate_one_tournament(self) -> None:
        qualifier_a, qualifier_b = self.simulate_qualifiers()
        self.qualifier_a_counts[qualifier_a] += 1
        self.qualifier_b_counts[qualifier_b] += 1

        group_a = list(tournament.GROUP_A_SEEDED_TEAMS) + [qualifier_a]
        group_b = list(tournament.GROUP_B_SEEDED_TEAMS) + [qualifier_b]

        a_top3, a_fourth = self.simulate_group("A", group_a)
        b_top3, b_fourth = self.simulate_group("B", group_b)

        # Approximate best fourth-place comparison: model head-to-head at a
        # randomly selected South African venue. Not official NRR.
        fourth_a, fourth_b = a_fourth[0], b_fourth[0]
        fourth_winner = self.play(fourth_a, fourth_b, "group")

        super7_seeds = {
            "A1": a_top3[0],
            "A2": a_top3[1],
            "A3": a_top3[2],
            "A4": fourth_winner,
            "B1": b_top3[0],
            "B2": b_top3[1],
            "B3": b_top3[2],
        }

        super7_order = self.simulate_super7(super7_seeds)
        semifinalists = super7_order[:4]

        semi1_winner = self.play(semifinalists[0], semifinalists[3], "semi_final_1")
        semi2_winner = self.play(semifinalists[1], semifinalists[2], "semi_final_2")
        champion = self.play(semi1_winner, semi2_winner, "final")

        for team in group_a + group_b:
            self.progress[team]["group_stage"] += 1
        for team in a_top3 + b_top3 + [fourth_winner]:
            self.progress[team]["super7"] += 1
        for team in semifinalists:
            self.progress[team]["semi_final"] += 1
        for team in [semi1_winner, semi2_winner]:
            self.progress[team]["final"] += 1
        self.champion_counts[champion] += 1

    def run(self, simulations: int) -> pd.DataFrame:
        for i in range(simulations):
            self.simulate_one_tournament()
            if (i + 1) % max(1, simulations // 10) == 0:
                print(f"Completed {i + 1:,}/{simulations:,} simulations")

        all_teams = list(tournament.ALL_WORKING_TEAMS)
        rows = []
        for team in all_teams:
            rows.append({
                "team": team,
                "qualifier_a_probability": self.qualifier_a_counts[team] / simulations,
                "qualifier_b_probability": self.qualifier_b_counts[team] / simulations,
                "group_stage_probability": self.progress[team]["group_stage"] / simulations,
                "super7_probability": self.progress[team]["super7"] / simulations,
                "semi_final_probability": self.progress[team]["semi_final"] / simulations,
                "final_probability": self.progress[team]["final"] / simulations,
                "champion_probability": self.champion_counts[team] / simulations,
            })

        results = pd.DataFrame(rows)
        results = results.sort_values(
            ["champion_probability", "final_probability"],
            ascending=False,
        ).reset_index(drop=True)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulator for the ICC Men's Cricket World Cup 2027."
    )
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.simulations < 1:
        parser.error("--simulations must be at least 1")

    tournament.validate_tournament_config()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    simulator = WorldCupSimulator(seed=args.seed)
    results = simulator.run(args.simulations)

    output_file = REPORTS_DIR / "world_cup_2027_monte_carlo_results.csv"
    results.to_csv(output_file, index=False)

    print("\n" + "=" * 78)
    print("ICC MEN'S CRICKET WORLD CUP 2027 — MONTE CARLO RESULTS")
    print("=" * 78)
    print(f"Simulations: {args.simulations:,}")
    print(f"Random seed: {args.seed}")
    print(f"Historical cutoff: {simulator.cutoff_date}")
    print("\nTop champion probabilities:")
    print(
        results[["team", "champion_probability", "final_probability",
                 "semi_final_probability", "super7_probability"]]
        .head(10)
        .to_string(index=False, formatters={
            "champion_probability": "{:.1%}".format,
            "final_probability": "{:.1%}".format,
            "semi_final_probability": "{:.1%}".format,
            "super7_probability": "{:.1%}".format,
        })
    )
    print(f"\nFull results saved to:\n{output_file}")
    print("\nCaveats:")
    print("- Qualifier round-robin is a provisional project assumption.")
    print("- Group/Super 7 standings use a random tie-break proxy, not official NRR.")
    print("- Venue assignment is stage-level random sampling, not exact official fixtures.")
    print("- These are model-based scenarios, not guarantees or betting advice.")


if __name__ == "__main__":
    main()