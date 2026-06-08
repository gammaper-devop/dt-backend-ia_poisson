from abc import ABC, abstractmethod
from typing import Dict, Tuple, List # <-- Asegúrate de importar List si no estaba
from domain.models import MatchTeams, ExpectedGoals

class MatchDataLoader(ABC):
    """Interfaz para cargar datos históricos (S del SOLID: Única responsabilidad)."""
    @abstractmethod
    def load_recent_data(self, file_path: str) -> Tuple[object, Dict[str, Dict[str, float]]]:
        pass

class PredictiveModel(ABC):
    """Interfaz para el modelo de Machine Learning (O de SOLID: Abierto a extensión)."""
    @abstractmethod
    def fit(self, data: object, team_stats: Dict[str, Dict[str, float]]) -> None:
        pass

    @abstractmethod
    def predict_lambdas(self, teams: MatchTeams, team_stats: Dict[str, Dict[str, float]]) -> ExpectedGoals:
        pass

class ProbabilityCalculator(ABC):
    """Interfaz para cálculos estadísticos (Poisson o cualquier otro método futuro)."""
    @abstractmethod
    def calculate_distribution(self, expected_goals: ExpectedGoals, top_n: int = 5) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        pass