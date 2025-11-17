from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API para predicción de riesgo de enfermedad cardíaca",
    version="1.0.0"
)

# Cargar el modelo entrenado al iniciar la aplicación
model = joblib.load("app/model.joblib")

# Definir el esquema de entrada usando Pydantic
class Input(BaseModel):
    Age: float
    RestingBP: float
    Cholesterol: float
    FastingBS: int
    MaxHR: float
    Oldpeak: float
    Sex_M: int
    ChestPainType_ATA: int
    ChestPainType_NAP: int
    ChestPainType_TA: int
    RestingECG_Normal: int
    RestingECG_ST: int
    ExerciseAngina_Y: int
    ST_Slope_Flat: int
    ST_Slope_Up: int

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 63.0,
                "RestingBP": 140.0,
                "Cholesterol": 260.0,
                "FastingBS": 0,
                "MaxHR": 112.0,
                "Oldpeak": 3.0,
                "Sex_M": 1,
                "ChestPainType_ATA": 0,
                "ChestPainType_NAP": 0,
                "ChestPainType_TA": 0,
                "RestingECG_Normal": 0,
                "RestingECG_ST": 1,
                "ExerciseAngina_Y": 1,
                "ST_Slope_Flat": 0,
                "ST_Slope_Up": 1
            }
        }

# Endpoint raíz para verificar que la API está funcionando
@app.get("/")
def read_root():
    return {
        "message": "Heart Disease Prediction API",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Realizar predicción de riesgo cardíaco",
            "/health": "GET - Verificar estado de salud del servicio"
        }
    }

# Endpoint de salud para monitoreo
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Endpoint de predicción
@app.post("/predict")
def predict(input_data: Input):
    """
    Realiza una predicción de riesgo de enfermedad cardíaca.
    
    Args:
        input_data: Objeto Input con 15 features explícitas
            - Variables numéricas: Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak
            - Variables categóricas codificadas (one-hot): Sex_M, ChestPainType_*, RestingECG_*, ExerciseAngina_Y, ST_Slope_*
    
    Returns:
        JSON con la probabilidad de enfermedad cardíaca y la predicción binaria
    """
    # Convertir los campos del modelo Pydantic a una lista en el orden correcto
    features_list = [
        input_data.Age,
        input_data.RestingBP,
        input_data.Cholesterol,
        input_data.FastingBS,
        input_data.MaxHR,
        input_data.Oldpeak,
        input_data.Sex_M,
        input_data.ChestPainType_ATA,
        input_data.ChestPainType_NAP,
        input_data.ChestPainType_TA,
        input_data.RestingECG_Normal,
        input_data.RestingECG_ST,
        input_data.ExerciseAngina_Y,
        input_data.ST_Slope_Flat,
        input_data.ST_Slope_Up
    ]
    
    # Convertir a array de NumPy
    features_array = np.array(features_list).reshape(1, -1)
    
    # Obtener la probabilidad de la clase positiva (índice 1)
    probability = float(model.predict_proba(features_array)[0][1])
    
    # Determinar la predicción binaria (umbral 0.5)
    prediction = 1 if probability > 0.5 else 0
    
    return {
        "heart_disease_probability": round(probability, 4),
        "prediction": prediction
    }