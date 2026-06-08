import numpy as np
from scipy.stats import poisson
from typing import Dict, Tuple, List
from domain.interfaces import ProbabilityCalculator
from domain.models import ExpectedGoals

class ScipyPoissonCalculator(ProbabilityCalculator):
    """Motor estadístico basado en la distribución de Poisson utilizando SciPy."""
    def __init__(self, max_goals: int = 6):
        self.max_goals = max_goals

    def calculate_distribution(self, expected_goals: ExpectedGoals, top_n: int = 5) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        matrix = np.zeros((self.max_goals, self.max_goals))
        
        for i in range(self.max_goals):
            for j in range(self.max_goals):
                matrix[i, j] = poisson.pmf(i, expected_goals.home) * poisson.pmf(j, expected_goals.away)
                
        outcomes = {
            "home_win": float(np.sum(np.tril(matrix, -1))),
            "draw": float(np.sum(np.diag(matrix))),
            "away_win": float(np.sum(np.triu(matrix, 1)))
        }
        
        exact_scores = []
        for i in range(self.max_goals):
            for j in range(self.max_goals):
                exact_scores.append({"score": f"{i}-{j}", "prob": float(matrix[i, j])})
                
        # Hacemos el slicing dinámico usando el parámetro 🎯
        top_scores = sorted(exact_scores, key=lambda x: x["prob"], reverse=True)[:top_n]
        return outcomes, top_scores