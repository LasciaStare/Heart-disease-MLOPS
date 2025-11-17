from fastapi.testclient import TestClient
from app.api import app

# Crear instancia del cliente de prueba
client = TestClient(app)


def test_predict_endpoint_success():
    """
    Prueba que el endpoint /predict funciona correctamente con datos válidos.
    """
    # Payload de ejemplo con 15 features válidas
    payload = {
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
    
    # Realizar solicitud POST al endpoint
    response = client.post("/predict", json=payload)
    
    # Aserciones para verificar la respuesta
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    
    # Convertir respuesta a JSON
    response_data = response.json()
    
    # Verificar que contiene las claves esperadas
    assert "heart_disease_probability" in response_data, "Missing key: heart_disease_probability"
    assert "prediction" in response_data, "Missing key: prediction"
    
    # Verificar que la probabilidad es un número entre 0.0 y 1.0
    probability = response_data["heart_disease_probability"]
    assert isinstance(probability, (float, int)), f"Probability should be a number, got {type(probability)}"
    assert 0.0 <= probability <= 1.0, f"Probability should be between 0.0 and 1.0, got {probability}"
    
    # Verificar que la predicción es 0 o 1
    prediction = response_data["prediction"]
    assert prediction in [0, 1], f"Prediction should be 0 or 1, got {prediction}"


def test_predict_endpoint_with_low_risk_patient():
    """
    Prueba el endpoint con un paciente de bajo riesgo.
    """
    payload = {
        "Age": 30.0,
        "RestingBP": 120.0,
        "Cholesterol": 180.0,
        "FastingBS": 0,
        "MaxHR": 180.0,
        "Oldpeak": 0.0,
        "Sex_M": 0,
        "ChestPainType_ATA": 1,
        "ChestPainType_NAP": 0,
        "ChestPainType_TA": 0,
        "RestingECG_Normal": 1,
        "RestingECG_ST": 0,
        "ExerciseAngina_Y": 0,
        "ST_Slope_Flat": 0,
        "ST_Slope_Up": 1
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    response_data = response.json()
    assert "heart_disease_probability" in response_data
    assert "prediction" in response_data


def test_health_endpoint():
    """
    Prueba que el endpoint /health funciona correctamente.
    """
    response = client.get("/health")
    
    assert response.status_code == 200
    response_data = response.json()
    assert "status" in response_data
    assert response_data["status"] == "healthy"
    assert "model_loaded" in response_data
    assert response_data["model_loaded"] is True


def test_root_endpoint():
    """
    Prueba que el endpoint raíz funciona correctamente.
    """
    response = client.get("/")
    
    assert response.status_code == 200
    response_data = response.json()
    assert "message" in response_data
    assert "status" in response_data
    assert response_data["status"] == "running"


def test_predict_endpoint_missing_field():
    """
    Prueba que el endpoint retorna error 422 cuando falta un campo requerido.
    """
    # Payload incompleto (falta ST_Slope_Up)
    payload = {
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
        "ST_Slope_Flat": 0
        # Falta ST_Slope_Up
    }
    
    response = client.post("/predict", json=payload)
    
    # FastAPI retorna 422 para errores de validación
    assert response.status_code == 422


def test_predict_endpoint_invalid_type():
    """
    Prueba que el endpoint retorna error cuando se envía un tipo de dato inválido.
    """
    payload = {
        "Age": "invalid_string",  # Debería ser un número
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
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 422
