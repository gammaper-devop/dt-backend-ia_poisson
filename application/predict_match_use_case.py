from domain.interfaces import MatchDataLoader, PredictiveModel, ProbabilityCalculator
from domain.models import MatchTeams, ExpectedGoals, Probabilities1X2, FairOdds, ExactScore, PredictionResult

class PredictMatchUseCase:
    """Caso de uso para predecir un partido usando componentes desacoplados."""
    def __init__(
        self, 
        data_loader: MatchDataLoader, 
        predictive_model: PredictiveModel, 
        prob_calculator: ProbabilityCalculator,
        data_path: str,
        top_n: int = 5 # <-- Agregado al constructor por defecto en 5
    ):
        self._data_loader = data_loader
        self._predictive_model = predictive_model
        self._prob_calculator = prob_calculator
        self._data_path = data_path
        self.top_n = top_n # <-- Guardamos la variable de configuración
        self._team_stats = {}
        
    def initialize(self):
        """Inicializa cargando datos y entrenando el modelo de Machine Learning."""
        raw_data, self._team_stats = self._data_loader.load_recent_data(self._data_path)
        self._predictive_model.fit(raw_data, self._team_stats)
        
    def execute(self, home_team: str, away_team: str) -> PredictionResult:
        teams = MatchTeams(home=home_team, away=away_team)
        
        if home_team not in self._team_stats or away_team not in self._team_stats:
            raise ValueError("Uno o ambos equipos no existen en los registros recientes.")
            
        # 1. Obtener Lambdas mediante ML
        lambdas = self._predictive_model.predict_lambdas(teams, self._team_stats)
        
        # 2. Calcular probabilidades enviando de manera inyectada el valor de top_n 🎯
        outcomes, top_scores = self._prob_calculator.calculate_distribution(lambdas, top_n=self.top_n)
        
        # 3. Mapear al modelo de Dominio
        prob_1x2 = Probabilities1X2(
            home_win=round(outcomes["home_win"] * 100, 2),
            draw=round(outcomes["draw"] * 100, 2),
            away_win=round(outcomes["away_win"] * 100, 2)
        )
        
        fair_odds = FairOdds(
            home_win=round(1 / outcomes["home_win"], 2) if outcomes["home_win"] > 0 else 0.0,
            draw=round(1 / outcomes["draw"], 2) if outcomes["draw"] > 0 else 0.0,
            away_win=round(1 / outcomes["away_win"], 2) if outcomes["away_win"] > 0 else 0.0
        )
        
        exact_scores_models = []
        for s in top_scores:
            # s["score"] viene en formato "golesLocal-golesVisitante" (Ej: "2-1")
            goles_home, goles_away = s["score"].split("-")
            
            formatted = f"{home_team} {goles_home} - {goles_away} {away_team}"
            
            exact_scores_models.append(
                ExactScore(
                    score=s["score"],
                    formatted_score=formatted, # <-- Se añade el nombre explícito de las selecciones 🚀
                    probability_percent=round(s["prob"] * 100, 2)
                )
            )
        
        return PredictionResult(
            teams=teams,
            expected_goals=ExpectedGoals(home=round(lambdas.home, 4), away=round(lambdas.away, 4)),
            probabilities_1X2=prob_1x2,
            fair_odds=fair_odds,
            top_exact_scores=exact_scores_models
        )