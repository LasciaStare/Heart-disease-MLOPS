"""
Script de Monitoreo de Deriva de Datos (Data Drift)

Este script genera un reporte HTML de deriva de datos utilizando Evidently.
Compara los datos de entrenamiento (referencia) con los datos de prueba (actuales)
para detectar cambios en la distribución de las features.

Prerequisitos:
- El notebook 2_model_pipeline_cv.ipynb debe haber sido ejecutado previamente
- Los archivos X_train.csv y X_test.csv deben existir en el directorio raíz

Salida:
- drift_report.html: Reporte interactivo con análisis de deriva de datos

Compatibilidad:
- Evidently >= 0.7.0 (API más reciente)
"""

import pandas as pd
from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift


def main():
    """
    Función principal que carga los datos y genera el reporte de deriva.
    """
    print("=" * 60)
    print("GENERANDO REPORTE DE DERIVA DE DATOS")
    print("=" * 60)
    
    # 1. Cargar artefactos de datos pre-procesados
    print("\n[1/4] Cargando datos de referencia (X_train.csv)...")
    try:
        reference_data = pd.read_csv('artifacts/X_train.csv')
        print(f"✓ Datos de referencia cargados: {reference_data.shape}")
    except FileNotFoundError:
        print("✗ Error: No se encontró X_train.csv")
        print("   Ejecuta primero el notebook 2_model_pipeline_cv.ipynb")
        return
    
    print("\n[2/4] Cargando datos actuales (X_test.csv)...")
    try:
        current_data = pd.read_csv('artifacts/X_test.csv')
        print(f"✓ Datos actuales cargados: {current_data.shape}")
    except FileNotFoundError:
        print("✗ Error: No se encontró X_test.csv")
        print("   Ejecuta primero el notebook 2_model_pipeline_cv.ipynb")
        return
    
    # 2. Crear el reporte de deriva con Evidently (API 0.7.x)
    print("\n[3/4] Generando reporte de deriva de datos...")
    
    # Crear lista de métricas: DriftedColumnsCount + ValueDrift para cada columna
    metrics_list = [
        DriftedColumnsCount()  # Cuenta cuántas columnas tienen deriva
    ]
    
    # Agregar ValueDrift para cada feature individual
    for column in reference_data.columns:
        metrics_list.append(ValueDrift(column=column))
    
    drift_report = Report(metrics=metrics_list)
    
    # Ejecutar el análisis
    snapshot = drift_report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    # 3. Guardar el reporte como HTML
    print("\n[4/4] Guardando reporte HTML...")
    output_file = 'drift_report.html'
    snapshot.save_html(output_file)
    
    print(f"\n{'=' * 60}")
    print(f"✓ Reporte generado exitosamente: {output_file}")
    print(f"{'=' * 60}")
    print("\nPara visualizar el reporte:")
    print(f"  - Abre el archivo {output_file} en tu navegador")
    print(f"  - O ejecuta: start {output_file} (Windows)")
    print(f"\nEl reporte incluye:")
    print("  • Número de columnas con deriva detectada")
    print("  • Análisis de deriva por cada feature")
    print("  • Comparación de distribuciones")
    print("  • Métricas estadísticas de cambio")
    print("  • Visualizaciones interactivas")


if __name__ == "__main__":
    main()
