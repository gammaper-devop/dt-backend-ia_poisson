import pandas as pd
import numpy as np
from typing import Dict
from sklearn.linear_model import LinearRegression
from domain.interfaces import PredictiveModel
from domain.models import MatchTeams, ExpectedGoals

class SklearnLinearPredictiveModel(PredictiveModel):
    """Implementación de Machine Learning usando Regresión Lineal de Scikit-Learn."""
    def __init__(self):
        self._model_home = LinearRegression()
        self._model_away = LinearRegression()

    def fit(self, data: pd.DataFrame, team_stats: Dict[str, Dict[str, float]]) -> None:
        X, y_h, y_a = [], [], []
        for _, row in data.iterrows():
            h_team, a_team = row["home_team"], row["away_team"]
            features = [
                team_stats[h_team]["att_home"],
                team_stats[a_team]["def_away"],
                team_stats[a_team]["att_away"],
                team_stats[h_team]["def_home"]
            ]
            X.append(features)
            y_h.append(row["home_score"])
            y_a.append(row["away_score"])
            
        self._model_home.fit(X, y_h)
        self._model_away.fit(X, y_a)

    def predict_lambdas(self, teams: MatchTeams, team_stats: Dict[str, Dict[str, float]]) -> ExpectedGoals:
        features = [
            team_stats[teams.home]["att_home"],
            team_stats[teams.away]["def_away"],
            team_stats[teams.away]["att_away"],
            team_stats[teams.home]["def_home"]
        ]
        
        # Evitamos Lambdas menores o iguales a cero
        lambda_h = max(0.05, self._model_home.predict([features])[0])
        lambda_a = max(0.05, self._model_away.predict([features])[0])
        return ExpectedGoals(home=lambda_h, away=lambda_a)