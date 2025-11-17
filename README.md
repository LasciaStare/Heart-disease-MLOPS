# MLOps Project

---

## Descripción General

Este proyecto implementa un pipeline completo de **Machine Learning Operations (MLOps)** para entrenar, desplegar y servir un modelo de clasificación que predice el riesgo de enfermedad cardíaca a partir de variables clínicas.

El proyecto cubre todo el ciclo de vida del modelo de ML:
- **Análisis exploratorio** y detección de data leakage
- **Entrenamiento y optimización** de modelos con GridSearchCV
- **Exposición del modelo** mediante API REST con FastAPI
- **Contenerización** con Docker
- **Orquestación** con Kubernetes

---

## Estructura del Proyecto

```
Final/
│
├── notebooks/
│   ├── 1_model_leakage_demo.ipynb      # Demostración de data leakage
│   ├── 2_model_pipeline_cv.ipynb       # Entrenamiento y optimización del modelo
│   └── heart.csv                        # Dataset original (918 pacientes)
│
├── app/
│   ├── api.py                           # API FastAPI con endpoints de predicción
│   └── model.joblib                     # Modelo entrenado (KNeighborsClassifier)
│
├── Scripts/
│   └── generate_drift_report.py         # Script de monitoreo de deriva de datos
│
├── docker/
│   ├── Dockerfile                       # Imagen Docker optimizada
│   └── requirements.txt                 # Dependencias del proyecto
│
├── k8s/
│   ├── deployment.yaml                  # Configuración de despliegue en Kubernetes
│   └── service.yaml                     # Configuración del servicio LoadBalancer
│
├── tests/
│   ├── __init__.py                      # Paquete de pruebas
│   └── test_api.py                      # Suite de pruebas para la API
│
├── artifacts/
│   ├── X_train.csv                      # Datos de entrenamiento (generados por notebook)
│   └── X_test.csv                       # Datos de prueba (generados por notebook)
│
├── .github/
│   └── workflows/
│       ├── ci.yml                       # Pipeline de CI/CD con GitHub Actions
│       └── deploy-notebooks.yml         # Despliegue automático de notebooks con Jupyter Book
│
├── drift_report.html                    # Reporte de deriva de datos (generado)
├── _config.yml                          # Configuraciones del jbook
├── _toc.yml                             # Table of content also jbook related
├── requirements.txt                     # Librerias para el jbook
└── README.md                            # Este archivo
```

---

## Tecnologías Utilizadas

| Herramienta        | Propósito                                |
| ------------------ | ---------------------------------------- |
| **Python 3.10**    | Lenguaje principal                       |
| **FastAPI**        | Framework para la API REST               |
| **scikit-learn**   | Entrenamiento y evaluación de modelos ML |
| **Pydantic**       | Validación de datos de entrada           |
| **Joblib**         | Serialización del modelo entrenado       |
| **Uvicorn**        | Servidor ASGI para FastAPI               |
| **Docker**         | Contenerización de la aplicación         |
| **Kubernetes**     | Orquestación de contenedores             |
| **NumPy / Pandas** | Manipulación y análisis de datos         |

---

## Componentes del Proyecto

### 1. Notebooks de Análisis

#### **`1_model_leakage_demo.ipynb`**
Demuestra el impacto del **data leakage** en el rendimiento de modelos:
- Comparación de modelos entrenados **con** y **sin** fuga de datos
- Análisis exploratorio del dataset (918 pacientes, 12 variables)
- Visualizaciones de distribuciones y correlaciones
- Resultados: data leakage infla artificialmente el AUC hasta 1

#### **`2_model_pipeline_cv.ipynb`**
Pipeline completo de entrenamiento sin data leakage:
- Preprocesamiento correcto (train/test split antes de cualquier transformación)
- **GridSearchCV** con 5-fold cross-validation
- Modelos evaluados:
  - KNeighborsClassifier
  - GaussianNB
  - SVC
- **Criterio de selección**: máximo Recall (por contexto clínico), luego máximo AUC
- **Modelo ganador**: KNeighborsClassifier
  - Recall: **91.18%** (solo 9 falsos negativos de 102 casos positivos)
  - AUC: **0.9397**
  - Accuracy: **89.67%**
- Exportación del modelo a `app/model.joblib`

### 2. API REST con FastAPI

**Archivo**: `app/api.py`

Características principales:
- Carga automática del modelo al iniciar
- Validación de entrada con Pydantic
- Documentación interactiva automática (Swagger UI)
- Endpoints:
  - `GET /` — Información general de la API
  - `GET /health` — Health check del servicio
  - `POST /predict` — Predicción de riesgo cardíaco

**Ejemplo de petición**:
```json
POST /predict
{
  "features": [63, 1, 140, 260, 0, 2, 112, 1, 3.0, 1, 1, 3, 0, 0, 0, 1]
}
```

**Respuesta**:
```json
{
  "heart_disease_probability": 0.8542,
  "prediction": 1
}
```

### 3. Contenerización con Docker

**Archivo**: `docker/Dockerfile`

Imagen Docker optimizada:
- Base: `python:3.10-slim` (ligera y eficiente)
- Instalación de dependencias sin caché (`--no-cache-dir`)
- Expone puerto 8000
- Comando de inicio: `uvicorn app.api:app --host 0.0.0.0 --port 8000`

### 4. Orquestación con Kubernetes

**Archivos**: `k8s/deployment.yaml` y `k8s/service.yaml`

- **Deployment**: gestiona el pod con la imagen Docker
- **Service**: expone la API mediante LoadBalancer
- Escalable y con health checks configurados

---

## Guía de Ejecución

### Opción 1: Ejecución Local sin Docker

#### 1. Crear entorno virtual e instalar dependencias:
```bash
conda create -n heart_mlops python=3.10
conda activate heart_mlops
pip install -r docker/requirements.txt
```

#### 2. Iniciar el servidor:
```bash
uvicorn app.api:app --reload
```

#### 3. Acceder a la API:
- **Documentación interactiva**: http://localhost:8000/docs
- **Endpoint raíz**: http://localhost:8000
- **Health check**: http://localhost:8000/health

---

### Opción 2: Despliegue con Docker

#### 1. Construir la imagen:
```bash
docker build -t heart-api -f docker/Dockerfile .
```

#### 2. Ejecutar el contenedor:
```bash
docker run -p 8000:8000 heart-api
```

#### 3. Acceder a la API:
- **Documentación Swagger**: http://localhost:8000/docs
- **Predicción**: http://localhost:8000/predict
- **Health check**: http://localhost:8000/health

---

### Opción 3: Despliegue con Kubernetes

#### **Prerequisitos:**
- Docker Desktop con Kubernetes habilitado

#### **Paso 1: Construir y publicar la imagen en Docker Hub**

```bash
# Construir la imagen
docker build -t heart-api -f docker/Dockerfile .

# Etiquetar la imagen con tu usuario de Docker Hub
docker tag heart-api <TU_USUARIO_DOCKER>/heart-api:latest

# Hacer login en Docker Hub
docker login

# Publicar la imagen
docker push <TU_USUARIO_DOCKER>/heart-api:latest
```

> **Nota importante**: Reemplaza `<TU_USUARIO_DOCKER>` con tu nombre de usuario de Docker Hub en todos los comandos.

#### **Paso 2: Actualizar el manifiesto de Deployment**

Edita `k8s/deployment.yaml` y reemplaza `<TU_USUARIO_DOCKER>` en la línea de `image:` con tu usuario real de Docker Hub:

```yaml
image: miusuario/heart-api:latest  # Ejemplo
```

#### **Paso 3: Aplicar los manifiestos de Kubernetes**

```bash
# Aplicar el Deployment (crea 2 réplicas de la aplicación)
kubectl apply -f k8s/deployment.yaml

# Aplicar el Service (expone la aplicación)
kubectl apply -f k8s/service.yaml
```

#### **Paso 4: Verificar el despliegue**

```bash
# Verificar que los Pods se están ejecutando
kubectl get pods


# Verificar el estado del servicio
kubectl get service heart-model-service

```

#### **Paso 5: Acceder a la API**

**En Docker Desktop:**
```
http://localhost/docs
http://localhost/predict
http://localhost/health
```

**En Minikube:**
```bash
# Obtener la URL del servicio
minikube service heart-model-service --url

# Usar la URL retornada (ej: http://192.168.49.2:30123)
```

#### **Comandos útiles de Kubernetes**

```bash
# Ver logs de un pod específico
kubectl logs <nombre-del-pod>

# Ver logs de todos los pods del deployment
kubectl logs -l app=heart-model

# Escalar el número de réplicas
kubectl scale deployment heart-model-deployment --replicas=3

# Ver detalles del deployment
kubectl describe deployment heart-model-deployment

# Ver detalles del service
kubectl describe service heart-model-service

# Eliminar los recursos
kubectl delete -f k8s/deployment.yaml
kubectl delete -f k8s/service.yaml
```

---

## Ejemplo de Uso de la API

### Usando `curl`:
```bash
curl -X POST "http://localhost/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Usando Python:
```python
import requests

url = "http://localhost/predict"
data = {
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

response = requests.post(url, json=data)
print(response.json())
# Output: {'heart_disease_probability': 0.8542, 'prediction': 1}
```

### Interpretación de la respuesta:
- **`heart_disease_probability`**: Probabilidad de enfermedad cardíaca (0.0 a 1.0)
- **`prediction`**: Clasificación binaria (0 = Sin riesgo, 1 = Con riesgo)



## CI/CD con GitHub Actions

El proyecto incluye **dos workflows automatizados** que se ejecutan en GitHub Actions:

### **1. CI Pipeline** (`.github/workflows/ci.yml`)

Pipeline de Integración Continua que valida la calidad del código en cada push o PR a `main` y `develop`.

**Pasos del pipeline:**

1. **Checkout del código**: Obtiene la última versión del repositorio
2. **Setup de Python 3.10**: Configura el entorno de Python
3. **Instalación de dependencias**: Instala todas las librerías necesarias + Evidently
4. **Linting con Flake8**: Verifica la calidad y estilo del código
   - Detecta errores de sintaxis
   - Verifica complejidad ciclomática
   - Asegura cumplimiento de estándares PEP8
5. **Testing con Pytest**: Ejecuta la suite de pruebas
   - Tests de endpoints (`/predict`, `/health`, `/`)
   - Tests de validación de datos
   - Tests de casos límite y errores
   - Genera reporte de cobertura de código
6. **Generate Drift Report**: Genera automáticamente el reporte de deriva de datos
7. **Upload Drift Report**: Sube el reporte HTML como artefacto descargable

### **2. Deploy Notebooks** (`.github/workflows/deploy-notebooks.yml`)

Workflow que despliega automáticamente los notebooks como un libro interactivo usando **Jupyter Book** en GitHub Pages.

**Pasos del workflow:**

1. **Checkout del repositorio**
2. **Configuración de Python 3.10**
3. **Instalación de Jupyter Book** y dependencias
4. **Creación de estructura** del libro (_config.yml, _toc.yml, intro.md)
5. **Build del Jupyter Book** con todos los notebooks
6. **Despliegue a GitHub Pages** automático

**Resultado**: Los notebooks se publican en una web interactiva accesible en:
[Notebooks](https://lasciastare.github.io/Heart-disease-MLOPS/)


### **Suite de Pruebas** (`tests/test_api.py`)

Se incluyen 7 tests automatizados:

| Test                                          | Descripción                                   |
| --------------------------------------------- | --------------------------------------------- |
| `test_predict_endpoint_success`               | Verifica predicción exitosa con datos válidos |
| `test_predict_endpoint_with_low_risk_patient` | Prueba con paciente de bajo riesgo            |
| `test_health_endpoint`                        | Verifica endpoint de salud                    |
| `test_root_endpoint`                          | Verifica endpoint raíz                        |
| `test_predict_endpoint_missing_field`         | Valida error cuando falta un campo            |
| `test_predict_endpoint_invalid_type`          | Valida error con tipo de dato incorrecto      |

### **Ejecutar las pruebas localmente**

```bash
# Instalar dependencias de testing
pip install pytest pytest-cov flake8

# Ejecutar linting
flake8 app/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Ejecutar pruebas con reporte de cobertura
pytest tests/ -v --cov=app --cov-report=term-missing

# Ejecutar pruebas de un archivo específico
pytest tests/test_api.py -v

# Ejecutar una prueba específica
pytest tests/test_api.py::test_predict_endpoint_success -v
```

---

## Monitoreo de Deriva de Datos (Data Drift)

El proyecto incluye un sistema de monitoreo de deriva de datos usando **Evidently** para detectar cambios en la distribución de las features entre los datos de entrenamiento y nuevos datos.

### **Script de Monitoreo** (`Scripts/generate_drift_report.py`)

El script realiza las siguientes tareas:

1. Carga los artefactos de datos (`X_train.csv`, `X_test.csv`)
2. Compara distribuciones entre datos de referencia (train) y actuales (test)
3. Genera un reporte HTML interactivo con visualizaciones
4. Identifica features con deriva significativa

#### **Generar el reporte de deriva:**

```bash
# Desde el directorio raíz del proyecto
python Scripts/generate_drift_report.py
```

### **Visualizar el Reporte**

```bash
# En Windows
start drift_report.html
```

---

## Resultados del Modelo

| Métrica       | Valor  |
| ------------- | ------ |
| **Recall**    | 91.18% |
| **AUC**       | 0.9397 |
| **Accuracy**  | 89.67% |
| **Precision** | 90.29% |
| **F1-Score**  | 90.73% |

- El modelo prioriza **Recall** sobre otras métricas porque en contexto médico los **falsos negativos son críticos** (paciente enfermo diagnosticado como sano).
- Solo **9 falsos negativos** de 102 casos positivos en el conjunto de prueba.
- Los **falsos positivos** (10 casos) son aceptables ya que solo requieren estudios adicionales.

---

## Dataset

**Fuente**: [Heart Failure Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

- **Tamaño**: 918 pacientes
- **Features**: 12 variables clínicas (edad, sexo, presión arterial, colesterol, etc.)
- **Target**: HeartDisease (0 = No, 1 = Sí)
- **Prevalencia**: 55.3% de pacientes con enfermedad cardíaca
- **Split**: 80% entrenamiento (734), 20% prueba (184)

**Preprocesamiento**:
- Imputación de valores faltantes con la mediana
- One-hot encoding para variables categóricas
- Escalado con StandardScaler en pipeline

---


## Autores

- **José Menco**
- **Iván Ramirez**
- **Camilo Vargas**
