# CHANGES — 01_eda.py

## Ronda 1 (auditoría técnica inicial)

### 1. Logger: guard contra handlers duplicados
Añadido `if not logger.handlers:` antes de agregar los handlers.
**Por qué:** sin el guard, ejecutar el módulo varias veces duplicaba los mensajes en log y consola.

### 2. Métrica de discriminación → Mutual Information
Sustituida la métrica `|Δμ|/σ` por `mutual_info_classif` (sklearn).
**Por qué:** la métrica anterior era equivalente a un Cohen's d simplificado, no captura relaciones no lineales ni es robusta en distribuciones no gaussianas. MI mide dependencia real entre cada feature y la clase.
**Impacto:** el ranking de features y la selección de top features en heatmap y boxplots ahora usa MI.

### 3. Transformación log de Amount
Añadida columna `Amount_log = log1p(Amount)`. Fig 2 muestra original y transformada en dos subplots.
**Por qué:** Amount está muy sesgada a la derecha; la transformación log estabiliza la varianza y mejora modelos lineales y redes neuronales.

### 4. Time → hora del día cíclica
Añadida columna `Time_hour = (Time / 3600) % 24`. Fig 3 muestra KDE de hora por clase.
**Por qué:** Time en segundos desde la primera transacción no tiene interpretación directa. La hora del día captura patrones temporales de fraude.

### 5. Heatmap filtrado a top 15 features (MI)
Fig 4 usa solo las 15 features con mayor MI en lugar de las 30 originales.
**Por qué:** el heatmap completo de 30×30 era visualmente saturado e imposible de interpretar.

### 6. Boxplot consolidado a top 10 features (MI)
Las dos figuras de boxplot (V1–V14 y V15–V28) se sustituyen por una sola figura con las top 10 features según MI.
**Por qué:** filtrar por relevancia (MI) hace el gráfico interpretable y conectado con el análisis.

### 7. Eliminada copia innecesaria (df.copy)
Sustituido `df_vio = df.copy()` por filtrado directo `df_clip = df[df['Amount'] < 2000].copy()`.
**Por qué:** la copia completa del DataFrame era innecesaria para un filtrado puntual.

---

## Ronda 2 (segunda auditoría — script ya mejorado)

### 0. Auditoría técnica (documentada en AUDIT.md)
Realizada revisión exhaustiva del script detectando oportunidades de robustez (error handling), eficiencia (sampling en MI) y mantenibilidad (centralización de constantes).

### 8. Constantes centralizadas
Añadidas `DATA_FILE`, `CLASS_LABELS` y `BOXPLOT_YLIM` como constantes al inicio del script.
**Por qué:** el nombre del archivo CSV, el mapeo de etiquetas y los límites del boxplot estaban hardcodeados y repetidos en múltiples bloques. Centralizarlos facilita cambios en un único punto.

### 9. Error handling en la carga de datos
`pd.read_csv` envuelto en `try/except FileNotFoundError` con mensaje informativo y `SystemExit(1)`.
**Por qué:** sin control, un archivo ausente lanzaba un traceback críptico. El nuevo mensaje indica exactamente qué falta y cómo obtenerlo (Kaggle).

### 10. Sampling para Mutual Information
La llamada a `mutual_info_classif` ahora opera sobre una muestra de máx. 50 000 filas (`df.sample`).
**Por qué:** con ~284k filas el cálculo de MI es costoso. El sampling reduce el tiempo significativamente manteniendo un ranking estable.

### 11. Mapeo de etiquetas de clase centralizado
Todas las ocurrencias de `{0: 'Legítima', 1: 'Fraude'}` sustituidas por la constante `CLASS_LABELS`.
**Por qué:** el mapeo aparecía duplicado en los bloques de Fig 5 y Fig 6. Un único punto de definición evita inconsistencias.

---

## Ronda 3 (tercera auditoría — mejoras menores de mantenibilidad)

### 12. Constante `MI_SAMPLE_SIZE`
Añadida constante `MI_SAMPLE_SIZE = 50_000`. Sustituido el literal en `df.sample(n=min(50_000, len(df)))`.
**Por qué:** el tamaño de muestra estaba hardcodeado. Como constante es ajustable sin buscar en el cuerpo del script.

### 13. Constante `OUTLIER_THRESHOLD_AMOUNT`
Añadida constante `OUTLIER_THRESHOLD_AMOUNT = 2_000`. Sustituido el literal `2000` en el filtro del violin plot.
**Por qué:** el umbral de corte era un "magic number" dentro de la lógica de visualización. Centralizarlo alinea su gestión con el resto de constantes del script.

### 14. Documentación técnica en español
El archivo `AUDIT.md` ha sido traducido íntegramente al español.
**Por qué:** alineación con el idioma preferido del usuario para la documentación técnica y el flujo de auditoría.

---

## Ronda 4 (cuarta auditoría — modularización y documentación)

### 15. Modularización: creado `src/plotting.py`
Las 7 funciones de visualización extraídas de `01_eda.py` a un módulo dedicado `plotting.py`.
**Por qué:** `01_eda.py` queda enfocado exclusivamente en el flujo del pipeline (carga → ingeniería → MI → llamadas a plot). Las funciones de visualización son reutilizables y testables de forma independiente.

### 16. Docstring en `get_logger`
Añadido docstring estilo Google a la función `get_logger` con descripción de args y return.
**Por qué:** era la única función del script sin documentación interna. Estandariza la legibilidad del código.

---

## Ronda 5 (quinta auditoría — type hints y rutas de salida)

### 17. Type hints completos en `plotting.py`
Añadidos hints a todos los parámetros de las 7 funciones: `df: pd.DataFrame`, `discrimin: pd.Series`, `class_labels: dict[int, str]`, `ylim: tuple[float, float]`, `threshold: float`. Añadido `import pandas as pd`.
**Por qué:** mantiene consistencia con el tipado ya presente en `get_logger` y habilita análisis estático (mypy, Pylance) y autocompletado en el IDE.

### 18. Rutas de salida explícitas (`output_path`)
Cambiado `results_dir: Path` por `output_path: Path` en todas las funciones de `plotting.py`. Las funciones ya no construyen el nombre del archivo internamente. Las llamadas en `01_eda.py` pasan el `Path` completo.
**Por qué:** el nombre del archivo de salida es una decisión del llamador, no de la función de visualización. El cambio desacopla completamente `plotting.py` de cualquier convención de nombrado específica del proyecto.

---

## Ronda 6 (sexta auditoría — testing y configuración externa)

### 19. Tests unitarios (`tests/test_plotting.py`)
Creado `tests/test_plotting.py` con 7 tests pytest, uno por función de `plotting.py`. Cada test genera un DataFrame sintético con `numpy.random.default_rng`, llama a la función con `tmp_path` y verifica que el archivo de salida existe.
**Por qué:** valida que ninguna función de visualización lanza excepción con datos realistas, sin depender del dataset real ni del sistema de archivos del proyecto.

### 20. Configuración externa (`config.yaml`)
Creado `config.yaml` en la raíz del proyecto con todas las constantes de `01_eda.py` (`data.file`, `model.mi_sample_size`, `visualization.*`). El script las carga con `yaml.safe_load` al inicio.
**Por qué:** permite ajustar parámetros del EDA (umbral de Amount, tamaño de muestra para MI, etc.) sin modificar código fuente. Facilita experimentación y revisión por parte de personas no técnicas.

### 21. Dependencias añadidas a `requirements.txt`
Añadidos `pyyaml>=6.0` y `pytest>=7.0`.
**Por qué:** nuevas dependencias directas introducidas por los cambios 19 y 20.

---

## Ronda 7 (séptima auditoría — integración y validación de schema)

### 22. Validación de schema con Pydantic (`src/config_schema.py`)
Creado `src/config_schema.py` con modelos Pydantic v2: `DataConfig`, `ModelConfig`, `VisualizationConfig` y `AppConfig`. `01_eda.py` valida el YAML con `AppConfig.model_validate(...)` antes de iniciar el pipeline.
**Por qué:** errores en `config.yaml` (campo faltante, tipo incorrecto) ahora producen un mensaje claro de Pydantic en lugar de un `KeyError` o `TypeError` críptico a mitad del pipeline.

### 23. Refactorización a `main(project_root)` en `01_eda.py`
Todo el pipeline envuelto en `def main(project_root: Path = ...)`. Añadido `if __name__ == '__main__': main()`. Las variables de ruta y configuración se derivan dentro de `main()`.
**Por qué:** hace el script importable y testable sin efectos secundarios en la importación. Requisito previo para el test de integración.

### 24. Test de integración (`tests/test_integration.py`)
Creado `tests/test_integration.py` con un test que escribe un CSV sintético y un `config.yaml` en `tmp_path`, llama a `main(tmp_path)` y verifica que las 7 figuras existen en `results/`.
**Por qué:** valida la integración completa YAML → pipeline → figuras con datos controlados, sin tocar el dataset real ni el sistema de archivos del proyecto.

### 25. `pydantic>=2.0` añadido a `requirements.txt`
**Por qué:** dependencia directa introducida por el cambio 22.

---

## Ronda 8 (octava auditoría — CI/CD)

### 26. GitHub Actions workflow (`.github/workflows/ci.yml`)
Creado workflow que ejecuta `pytest tests/ -v` automáticamente en cada push o pull request a `main`/`master`. Usa `actions/checkout@v4`, `actions/setup-python@v5` con Python 3.11 y caché de pip.
**Por qué:** sin CI, los tests solo se ejecutan cuando alguien se acuerda. El workflow garantiza que ningún cambio rompa el pipeline sin que se detecte antes de mergear.

---

## Ronda 9 (novena auditoría — linter)

### 27. Linter `ruff` en CI
Añadido paso `ruff check src/ tests/` en el workflow, antes de `pytest`. Creado `ruff.toml` con `line-length = 100`, `target-version = "py311"` y reglas `E`, `F`, `W`, `I` (errores, warnings, imports). `E501` ignorado para no forzar cortes en líneas largas de visualización.
**Por qué:** garantiza estilo consistente (PEP 8 + orden de imports) en cada commit, sin depender de que el desarrollador lo ejecute manualmente. `ruff` se eligió sobre `flake8` por velocidad y por cubrir también `isort` en un solo paso.

### 28. `ruff>=0.4` añadido a `requirements.txt`
**Por qué:** dependencia directa introducida por el cambio 27.

---

## Ronda 10 (décima auditoría — formateo automático)

### 29. `ruff format --check` en CI
Añadido paso `ruff format --check src/ tests/` en el workflow, entre lint y tests. El flag `--check` falla si algún archivo no está formateado correctamente, sin modificar nada.
**Por qué:** `ruff check` detecta problemas de estilo lógico; `ruff format` garantiza consistencia visual (espaciado, comillas, trailing commas). Son complementarios. Sin coste adicional al ya tener `ruff` instalado.
