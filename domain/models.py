from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MatchTeams:
    home: str
    away: str

@dataclass
class ExpectedGoals:
    home: float
    away: float

@dataclass
class Probabilities1X2:
    home_win: float
    draw: float
    away_win: float

@dataclass
class FairOdds:
    home_win: float
    draw: float
    away_win: float

@dataclass
class ExactScore:
    score: str
    formatted_score: str
    probability_percent: float

@dataclass
class PredictionResult:
    teams: MatchTeams
    expected_goals: ExpectedGoals
    probabilities_1X2: Probabilities1X2
    fair_odds: FairOdds
    top_exact_scores: List[ExactScore]