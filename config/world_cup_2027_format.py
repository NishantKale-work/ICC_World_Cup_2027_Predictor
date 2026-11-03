"""
ICC Men's Cricket World Cup 2027
Tournament configuration for the Monte Carlo simulator.

The four qualifier teams are the user's working assumptions:
West Indies, Ireland, Scotland, Netherlands.

The direct qualifier winner and Super Series winner are not assigned
to specific Group A / Group B slots yet. Keep those mappings configurable.
"""

TOURNAMENT_NAME = "ICC Men's Cricket World Cup 2027"

# ---------------------------------------------------------
# 1. Working team assumptions
# ---------------------------------------------------------

SEEDED_TEAMS = [
    "India",
    "Australia",
    "Pakistan",
    "Afghanistan",
    "Zimbabwe",
    "New Zealand",
    "South Africa",
    "Sri Lanka",
    "England",
    "Bangladesh",
]

QUALIFIER_TEAMS = [
    "West Indies",
    "Ireland",
    "Scotland",
    "Netherlands",
]

ALL_WORKING_TEAMS = SEEDED_TEAMS + QUALIFIER_TEAMS


# ---------------------------------------------------------
# 2. Group-stage configuration
# ---------------------------------------------------------

GROUP_A_SEEDED_TEAMS = [
    "India",
    "Australia",
    "Pakistan",
    "Afghanistan",
    "Zimbabwe",
]

GROUP_B_SEEDED_TEAMS = [
    "New Zealand",
    "South Africa",
    "Sri Lanka",
    "England",
    "Bangladesh",
]

# These are placeholders from the supplied fixture information.
# We will assign actual teams after modeling the qualifier pathway.
GROUP_A_QUALIFIER_SLOT = "Qualifier A"
GROUP_B_QUALIFIER_SLOT = "Qualifier B"

GROUPS = {
    "A": GROUP_A_SEEDED_TEAMS + [GROUP_A_QUALIFIER_SLOT],
    "B": GROUP_B_SEEDED_TEAMS + [GROUP_B_QUALIFIER_SLOT],
}

GROUP_STAGE_MATCHES_PER_TEAM = 5
GROUP_STAGE_MATCHES_TOTAL = 30

GROUP_QUALIFICATION_RULE = {
    "top_n_per_group": 3,
    "best_fourth_place_across_groups": 1,
    "total_teams_qualifying": 7,
}


# ---------------------------------------------------------
# 3. Qualifier and Super Series
# ---------------------------------------------------------

DIRECT_QUALIFIER_ADVANCES_TO_GROUP_STAGE = True

SUPER_SERIES_TEAM_COUNT = 3
SUPER_SERIES_MATCHES_PER_TEAM = 2
SUPER_SERIES_MATCHES_TOTAL = 3

SUPER_SERIES_FORMAT = "round_robin"

# The winner of the Super Series advances to the group stage.
SUPER_SERIES_QUALIFICATION_SPOT = "Qualifier B"

# The direct qualifier winner advances to the group stage.
DIRECT_QUALIFIER_SPOT = "Qualifier A"


# ---------------------------------------------------------
# 4. Super 7 configuration
# ---------------------------------------------------------

SUPER_7_TEAM_COUNT = 7
SUPER_7_MATCHES_PER_TEAM = 6
SUPER_7_MATCHES_TOTAL = 21

SUPER_7_FORMAT = "single_round_robin"

SUPER_7_QUALIFICATION_RULE = {
    "top_n": 4,
    "advance_to": "Semi-finals",
}

# Super 7 seed mapping supplied in the tournament information.
SUPER_7_SEED_MAPPING = {
    "A1": "Group A winner",
    "A2": "Group A runner-up",
    "A3": "Group A third place",
    "A4": "Best fourth-placed team across Groups A and B",
    "B1": "Group B winner",
    "B2": "Group B runner-up",
    "B3": "Group B third place",
}


# ---------------------------------------------------------
# 5. Knockout configuration
# ---------------------------------------------------------

SEMI_FINALS = {
    "semi_final_1": {
        "team_1": "Super 7 first place",
        "team_2": "Super 7 fourth place",
    },
    "semi_final_2": {
        "team_1": "Super 7 second place",
        "team_2": "Super 7 third place",
    },
}

FINAL = {
    "team_1": "Semi-final 1 winner",
    "team_2": "Semi-final 2 winner",
}

KNOCKOUT_MATCHES_TOTAL = 3


# ---------------------------------------------------------
# 6. Tournament totals
# ---------------------------------------------------------

TOURNAMENT_MATCH_TOTALS = {
    "super_series": 3,
    "group_stage": 30,
    "super_7": 21,
    "semi_finals": 2,
    "final": 1,
}

TOTAL_TOURNAMENT_MATCHES = sum(
    TOURNAMENT_MATCH_TOTALS.values()
)


# ---------------------------------------------------------
# 7. Basic validation
# ---------------------------------------------------------

def validate_tournament_config():
    """Validate basic team counts and tournament totals."""

    assert len(SEEDED_TEAMS) == 10, (
        "Expected 10 seeded teams."
    )

    assert len(QUALIFIER_TEAMS) == 4, (
        "Expected 4 working qualifier teams."
    )

    assert len(ALL_WORKING_TEAMS) == 14, (
        "Expected 14 teams in the working tournament lineup."
    )

    assert len(GROUP_A_SEEDED_TEAMS) == 5, (
        "Group A should contain 5 seeded teams."
    )

    assert len(GROUP_B_SEEDED_TEAMS) == 5, (
        "Group B should contain 5 seeded teams."
    )

    assert len(GROUPS["A"]) == 6, (
        "Group A should contain 6 teams including its qualifier slot."
    )

    assert len(GROUPS["B"]) == 6, (
        "Group B should contain 6 teams including its qualifier slot."
    )

    assert GROUP_STAGE_MATCHES_TOTAL == 30, (
        "Group stage should contain 30 matches."
    )

    assert SUPER_SERIES_MATCHES_TOTAL == 3, (
        "Super Series should contain 3 matches."
    )

    assert SUPER_7_MATCHES_TOTAL == 21, (
        "Super 7 should contain 21 matches."
    )

    assert TOTAL_TOURNAMENT_MATCHES == 57, (
        "Expected 57 total tournament matches."
    )

    return True


if __name__ == "__main__":
    if validate_tournament_config():
        print("Tournament configuration validated successfully.")
        print(f"Tournament: {TOURNAMENT_NAME}")
        print(f"Working teams: {len(ALL_WORKING_TEAMS)}")
        print(f"Group A teams/slots: {GROUPS['A']}")
        print(f"Group B teams/slots: {GROUPS['B']}")
        print(f"Total scheduled matches: {TOTAL_TOURNAMENT_MATCHES}")