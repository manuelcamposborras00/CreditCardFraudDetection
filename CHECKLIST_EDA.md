# Checklist de Validación Final: 01_eda.py

Este documento detalla los pasos de verificación manual y técnica necesarios para cerrar la fase de **Análisis Exploratorio de Datos (01 EDA)** con éxito. Completar este checklist garantiza que la base para el preprocesamiento (`02_preprocessing.py`) sea sólida y fiable.

## 1. Preparación del Entorno y Estilo
- [X] **Sincronización:** Ejecutar `pip install -r requirements.txt` para instalar las nuevas dependencias (`ruff`, `pydantic`, `pyyaml`, `pytest`).
- [X] **Linter (Estilo):** Ejecutar `ruff check src/ tests/`. El resultado debe ser de cero errores de estilo o imports.
- [X] **Formateo:** (Opcional) Ejecutar `ruff format src/ tests/` para estandarizar el espaciado y comillas.

## 2. Garantía de Calidad (QA)
- [X] **Tests Unitarios:** Ejecutar `pytest tests/test_plotting.py -v`. Deben pasar los 7 tests de visualización.
- [X] **Tests de Integración:** Ejecutar `pytest tests/test_integration.py -v`. Debe validar el flujo completo YAML -> Pydantic -> Pipeline -> Figuras.
- [X] **Acciones de GitHub:** Verificar que el flujo de CI en el repositorio (GitHub Actions) está en verde.

## 3. Ejecución con Datos Reales
- [X] **Dataset Original:** Confirmar la presencia de `data/creditcard.csv` (284,807 filas).
- [X] **Configuración de Producción:** Revisar `config.yaml`. Asegurar que `mi_sample_size` es adecuado (ej. `50000`) para un balance entre velocidad y precisión.
- [ ] **Pipeline:** Ejecutar `python src/01_eda.py` y verificar que termina sin excepciones.

## 4. Auditoría de Salidas (Inspección Visual)
- [ ] **Logs (`logs/01_eda.log`):**
    - [ ] Verificar `Shape: (284807, 31)`.
    - [ ] Confirmar `Valores nulos: 0`.
    - [ ] Revisar el top 5 de **Mutual Information** (anotar variables clave como V17, V14, V12).
- [ ] **Resultados Visuales (`results/`):**
    - [ ] `eda_distribucion_clases.png`: Confirmar escala logarítmica correcta.
    - [ ] `eda_distribucion_amount.png`: Validar que la campana `log1p` está bien formada.
    - [ ] `eda_features_discriminativas.png`: El ranking debe ser coherente con los logs.

## 5. Prueba de Robustez (Stress Testing)
- [ ] **Validación Pydantic:** Introducir un error deliberado en `config.yaml` (ej. `mi_sample_size: "texto"`) y confirmar que el script detiene la ejecución con un error de validación claro.
- [ ] **Archivo Faltante:** Renombrar temporalmente el CSV y confirmar que el script captura el error con el mensaje amigable de Kaggle.

---
**Estado Final:** Una vez completados estos puntos, el módulo 01 EDA se considera de **Alta Integridad** y está listo para alimentar la fase de **Preprocesamiento**.
