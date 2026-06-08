from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path  # <-- IMPORTANTE: Librería nativa para manejo robusto de rutas

from infrastructure.data_loader import PandasMatchDataLoader
from infrastructure.ml_model import SklearnLinearPredictiveModel
from infrastructure.poisson_calculator import ScipyPoissonCalculator
from application.predict_match_use_case import PredictMatchUseCase

app = FastAPI(
    title="Mundial 2026 Forecasting API (Poisson Distribution)",
    description="Backend desacoplado aplicando SOLID, Machine Learning y Poisson."
)

# 🎯 SOLUCIÓN PRO: Construimos la ruta absoluta dinámica respecto a donde esté este archivo main.py
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "results.csv"

# Inyección de dependencias manual (Composición)
data_loader = PandasMatchDataLoader()
ml_model = SklearnLinearPredictiveModel()
poisson_calculator = ScipyPoissonCalculator()

# Inyectamos la configuración del Top N y la ruta corregida
use_case = PredictMatchUseCase(
    data_loader=data_loader,
    predictive_model=ml_model,
    prob_calculator=poisson_calculator,
    data_path=str(DATA_PATH),  # <-- Pasamos la ruta absoluta del archivo en string 🚀
    top_n=5
)

@app.on_event("startup")
def startup_event():
    # Entrenar el modelo al arrancar el servidor
    use_case.initialize()

class MatchRequest(BaseModel):
    home_team: str = Query(..., description="Nombre del primer equipo (ej: Argentina)"),
    away_team: str = Query(..., description="Nombre del segundo equipo (ej: France)")

@app.get("/")
def inicio():
    return {"mensaje": "¡Bienvenido a la API del Mundial 2026! Ve a /docs para probar los pronósticos."}

@app.post("/api/v1/predict")
def predict(request: MatchRequest):
    try:
        result = use_case.execute(request.home_team, request.away_team)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno del servidor predictivo.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)