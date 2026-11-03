"""
Working team set for the ICC Men's Cricket World Cup 2027 Predictor.

Scotland, Netherlands, Ireland, and West Indies are treated as
assumed qualifiers for project development—not officially confirmed.
"""

WORLD_CUP_2027_TEAMS = {
    "Afghanistan",
    "Australia",
    "Bangladesh",
    "England",
    "India",
    "Ireland",
    "Netherlands",
    "New Zealand",
    "Pakistan",
    "Scotland",
    "South Africa",
    "Sri Lanka",
    "West Indies",
    "Zimbabwe",
}

ASSUMED_QUALIFIERS = {
    "Ireland",
    "Netherlands",
    "Scotland",
    "West Indies",
}

CONFIRMED_TEAM_SET_FOR_PROJECT = (
    WORLD_CUP_2027_TEAMS - ASSUMED_QUALIFIERS
)