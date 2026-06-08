import datetime
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LinearRegression

app = FastAPI(
    title="API de Predicciones Deportivas - Poisson & ML",
    description="Backend para calcular probabilidades de partidos de fútbol usando datos históricos.",
)

# Variables globales para almacenar el modelo y los mapeos
model_home = None
model_away = None
global_averages = {}
team_stats_cache = {}


def train_poisson_ml_model():
    """Carga los datos de results.csv, calcula las fuerzas históricas recientes

    (2024-2026) y entrena el modelo de Machine Learning.
    """
    global model_home, model_away, global_averages, team_stats_cache
    print("Iniciando el entrenamiento del modelo de ML...")

    #  Cargar los datos (results.csv)
    ruta_csv = os.path.join("data", "results.csv")
    #  Limpiar datos
    df = pd.read_csv(ruta_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])

    # Filtrar ventana de tiempo óptima (2024 - 2026)
    df_recent = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2026-12-31")].copy()

    if df_recent.empty:
        raise RuntimeError(
            "No hay suficientes datos en el rango de fechas especificado."
        )

    # 2. Calcular promedios globales
    global_averages["home_avg"] = df_recent["home_score"].mean()
    global_averages["away_avg"] = df_recent["away_score"].mean()

    # 3. Calcular fuerzas de ataque y defensa por equipo
    teams = set(df_recent["home_team"]).union(set(df_recent["away_team"]))

    for team in teams:
        home_matches = df_recent[df_recent["home_team"] == team]
        away_matches = df_recent[df_recent["away_team"] == team]

        # Por defecto, si el equipo no tiene partidos en alguna condición, toma el promedio (fuerza = 1.0)
        att_home = (
            (home_matches["home_score"].mean() / global_averages["home_avg"])
            if len(home_matches) > 0
            else 1.0
        )
        def_home = (
            (home_matches["away_score"].mean() / global_averages["away_avg"])
            if len(home_matches) > 0
            else 1.0
        )

        att_away = (
            (away_matches["away_score"].mean() / global_averages["away_avg"])
            if len(away_matches) > 0
            else 1.0
        )
        def_away = (
            (away_matches["home_score"].mean() / global_averages["home_avg"])
            if len(away_matches) > 0
            else 1.0
        )

        team_stats_cache[team] = {
            "att_home": att_home,
            "def_home": def_home,
            "att_away": att_away,
            "def_away": def_away,
        }

    # 4. Preparar el dataset para Machine Learning
    # Queremos predecir 'home_score' y 'away_score' usando las fuerzas cruzadas como características (Features)
    X_features = []
    y_home = []
    y_away = []

    for _, row in df_recent.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]

        # Características: [Ataque Local, Defensa Visitante, Ataque Visitante, Defensa Local]
        features = [
            team_stats_cache[h_team]["att_home"],
            team_stats_cache[a_team]["def_away"],
            team_stats_cache[a_team]["att_away"],
            team_stats_cache[h_team]["def_home"],
        ]
        X_features.append(features)
        y_home.append(row["home_score"])
        y_away.append(row["away_score"])

    # 5. Entrenar modelos de regresión lineal para obtener Lambdas dinámicos
    model_home = LinearRegression().fit(X_features, y_home)
    model_away = LinearRegression().fit(X_features, y_away)
    print("¡Modelo entrenado exitosamente!")


# Ejecutar el entrenamiento al levantar el servidor de FastAPI
@app.on_event("startup")
def startup_event():
    train_poisson_ml_model()


# Esquema de datos para la solicitud (Request body)
class MatchRequest(BaseModel):
    home_team: str
    away_team: str


@app.post("/predict")
def predict_match(match: MatchRequest):
    # Validar que los equipos existan en la base de datos
    if (
        match.home_team not in team_stats_cache
        or match.away_team not in team_stats_cache
    ):
        raise HTTPException(
            status_code=404,
            detail="Uno o ambos equipos no se encuentran en la base de datos reciente.",
        )

    # 1. Obtener características de los equipos involucrados
    features = [
        team_stats_cache[match.home_team]["att_home"],
        team_stats_cache[match.away_team]["def_away"],
        team_stats_cache[match.away_team]["att_away"],
        team_stats_cache[match.home_team]["def_home"],
    ]

    # 2. El modelo de ML predice los goles esperados (Lambda) corrigiendo sesgos
    lambda_home = max(0.05, model_home.predict([features])[0])
    lambda_away = max(0.05, model_away.predict([features])[0])

    # 3. Calcular la distribución de Poisson (Matriz 6x6 goles)
    max_goals = 6
    matrix = np.zeros((max_goals, max_goals))

    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)

    # 4. Calcular probabilidades de resultados generales (1X2)
    prob_home_win = float(np.sum(np.tril(matrix, -1)))
    prob_draw = float(np.sum(np.diag(matrix)))
    prob_away_win = float(np.sum(np.triu(matrix, 1)))

    # 5. Encontrar los 3 marcadores exactos más probables
    exact_scores = []
    for i in range(max_goals):
        for j in range(max_goals):
            exact_scores.append(
                {"score": f"{i}-{j}", "probability": float(matrix[i, j])}
            )

    # Ordenar de mayor a menor probabilidad
    exact_scores = sorted(exact_scores, key=lambda x: x["probability"], reverse=True)[
        :3
    ]

    # Formatear respuesta en JSON estructurado
    return {
        "teams": {"home": match.home_team, "away": match.away_team},
        "expected_goals": {
            "home": round(lambda_home, 4),
            "away": round(lambda_away, 4),
        },
        "probabilities_1X2": {
            "home_win": round(prob_home_win * 100, 2),
            "draw": round(prob_draw * 100, 2),
            "away_win": round(prob_away_win * 100, 2),
        },
        "fair_odds": {
            "home_win": round(1 / prob_home_win, 2) if prob_home_win > 0 else None,
            "draw": round(1 / prob_draw, 2) if prob_draw > 0 else None,
            "away_win": round(1 / prob_away_win, 2) if prob_away_win > 0 else None,
        },
        "top_exact_scores": [
            {
                "score": s["score"],
                "probability_percent": round(s["probability"] * 100, 2),
            }
            for s in exact_scores
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)