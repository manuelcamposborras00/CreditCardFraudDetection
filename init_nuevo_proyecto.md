# PROMPT DE INICIALIZACIÓN — CREDIT CARD FRAUD DETECTION
# Objetivo: 10/10 — sin compromisos

> Copia este archivo completo como primer mensaje en el nuevo directorio/conversación.

---

## ROL Y OBJETIVO

Actúa como un **ingeniero de machine learning senior** construyendo un proyecto académico de nivel máster.
El objetivo es un **10 sobre 10**. No es negociable.
Cada decisión técnica debe estar justificada. Cada resultado debe ser interpretado.
Cada línea de código debe ser correcta y reproducible.

Cuando recibas este prompt:
1. Crea la estructura de directorios completa
2. Escribe todos los scripts en orden (01 → 11), luego `main.py`, luego `memoria.tex`
3. **NO uses placeholders en tablas de la memoria** — los datos reales vienen de los JSON generados
4. Haz commit al final

---

## DATASET

**Credit Card Fraud Detection**
- Fuente: Kaggle — `mlg-ulb/creditcardfraud`
- Archivo: `creditcard.csv`
- **284.807 transacciones** con tarjeta de crédito (septiembre 2013, titulares europeos)
- **492 fraudes** → desbalanceo extremo: **0,172 % de clase positiva**
- Variables:
  - `Time` — segundos transcurridos desde la primera transacción del dataset
  - `Amount` — importe de la transacción en euros
  - `V1`–`V28` — 28 componentes PCA aplicado por los autores para anonimizar (ya transformadas)
  - `Class` — variable objetivo: 0 = legítima, 1 = fraude

**Por qué este dataset es perfecto para este enunciado:**

| Requisito del enunciado | Justificación natural |
|---|---|
| Detección de anomalías | El fraude **es por definición** una anomalía. No es una técnica añadida artificialmente |
| PCA | Las features **ya son componentes PCA**. Permite un análisis académicamente rico sobre interpretabilidad y reducción adicional |
| Clustering | Agrupar transacciones sin usar `Class` y analizar si el fraude se concentra en clusters |
| Paralelización | 284k filas → la paralelización tiene impacto real y cuantificable |
| Métricas no triviales | Accuracy es inútil (99,83% prediciendo siempre legítima). Hay que usar AUPRC, F1-fraud, Recall |

---

## MÉTRICAS PRIMARIAS — CRÍTICO

Con un desbalanceo del 0,172%, la **accuracy es una métrica engañosa** y no debe usarse como
criterio principal. Las métricas relevantes son:

- **AUPRC** (Area Under Precision-Recall Curve) — métrica principal para datasets desbalanceados
- **F1-score clase fraude** (F1 de la clase minoritaria)
- **Recall clase fraude** — en detección de fraude, los falsos negativos son más costosos
- **Precision clase fraude** — coste de falsas alarmas
- **ROC-AUC** — métrica complementaria

Accuracy se reporta pero **nunca** se usa como criterio de comparación entre modelos.

---

## MANEJO DEL DESBALANCEO

En **todos** los modelos supervisados, usar simultáneamente:
1. `class_weight='balanced'` o `scale_pos_weight` donde el modelo lo soporte
2. Estratificación en todos los splits: `train_test_split(..., stratify=y)`
3. `StratifiedKFold` en validación cruzada (no `KFold`)

Adicionalmente, en el script de preprocesamiento, aplicar **SMOTE** sobre el conjunto de
entrenamiento (nunca sobre test) y generar una versión oversampled para comparar.

---

## ESTRUCTURA DE ARCHIVOS

```
fraude_tarjeta/
├── main.py
├── requirements.txt
├── data/
│   └── creditcard.csv
├── logs/
│   └── (un .log por script)
├── results/
│   └── (figuras .png + JSONs de resultados)
├── src/
│   ├── 01_eda.py
│   ├── 02_preprocessing.py
│   ├── 03_hyperparameter_tuning.py   ← grid exhaustivo para TODOS los modelos
│   ├── 04_train_models.py
│   ├── 05_cross_validation.py
│   ├── 06_parallel_benchmark.py
│   ├── 07_average_times.py
│   ├── 08_dimensionality_reduction.py
│   ├── 09_clustering.py
│   └── 10_anomaly_detection.py
│   └── 11_final_analysis.py
└── memoria.tex
```

---

## ESPECIFICACIÓN DETALLADA DE CADA SCRIPT

---

### `src/01_eda.py`

Figuras en `results/`:
- `eda_distribucion_clases.png` — barplot absoluto + porcentaje (0 vs 1). Escala logarítmica en eje Y
- `eda_distribucion_amount.png` — histograma + KDE de Amount separado por clase (fraude vs legítima)
- `eda_distribucion_time.png` — densidad de Time por clase (fraudes distribuidos en el tiempo)
- `eda_heatmap_correlaciones.png` — correlación entre V1-V28 + Time + Amount
- `eda_boxplot_v_features_fraude.png` — boxplot de V1-V14 separado por clase
- `eda_boxplot_v_features_fraude2.png` — boxplot de V15-V28 separado por clase
- `eda_amount_por_clase.png` — violin plot de Amount por clase (con outliers)
- `eda_features_discriminativas.png` — barplot de diferencia de medias |μ_fraude - μ_legit| / σ

Log `logs/01_eda.log`:
```
[EDA] Shape: (284807, 31)
[EDA] Clase 0 (legítima): 284315  (99.828%)
[EDA] Clase 1 (fraude):       492   (0.172%)
[EDA] Ratio desbalanceo: 578:1
[EDA] Valores nulos: 0
[EDA] Amount — media legítima: X.XX €   media fraude: X.XX €
[EDA] Time — rango: [0, 172792] segundos
[EDA] Features más discriminativas (|Δμ|/σ): V14, V12, V10, V4, V11 ...
[EDA] Figuras guardadas: 8
```

---

### `src/02_preprocessing.py`

Pasos:
1. Verificar que no hay nulos
2. Eliminar duplicados exactos (reportar cuántos)
3. Feature engineering:
   - `Amount_log = np.log1p(Amount)` — normalizar Amount que tiene cola muy larga
   - `Hour = (Time % 86400) // 3600` — hora del día (ciclo de 24h)
   - `Is_night = (Hour >= 22) | (Hour <= 6)` — indicador de transacción nocturna
4. Escalar `Amount_log` y `Time` con StandardScaler (V1-V28 ya están escaladas)
   **CRÍTICO:** el scaler se ajusta SOLO sobre X_train después del split
5. Guardar `data/creditcard_cleaned.csv` SIN escalar (el escalado ocurre en cada script)
6. Guardar también `data/feature_names.json` con la lista de features finales

Log `logs/02_preprocessing.log`:
```
[PREP] Shape inicial: (284807, 31)
[PREP] Duplicados eliminados: X
[PREP] Features creadas: Amount_log, Hour, Is_night
[PREP] Features finales: V1-V28 + Amount_log + Hour + Is_night = 31 features
[PREP] Distribución Amount_log: media=X.XX  std=X.XX
[PREP] Transacciones nocturnas: X (X.X%)
[PREP] Fraudes nocturnos: X (X.X% de los fraudes)
[PREP] Guardado: data/creditcard_cleaned.csv
```

---

### `src/03_hyperparameter_tuning.py`

**GRIDS EXHAUSTIVOS PARA TODOS LOS MODELOS. TARDE LO QUE TARDE.**

Usar `StratifiedKFold(n_splits=5)` y `scoring='average_precision'` (AUPRC) en todos.
Usar `n_jobs=-1` en todos los `GridSearchCV`.
Guardar mejores parámetros de cada modelo en `results/best_hyperparams_{modelo}.json`.
Guardar CSV completo de resultados de cada grid en `results/grid_{modelo}.csv`.

#### Modelo 1: Logistic Regression
```python
param_grid_lr = {
    'C':       [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver':  ['saga'],          # único solver que soporta l1+elasticnet
    'l1_ratio':[0.1, 0.3, 0.5, 0.7, 0.9],  # solo para elasticnet
    'class_weight': ['balanced'],
    'max_iter': [2000]
}
# Nota: l1_ratio solo aplica cuando penalty='elasticnet'
# Usar Pipeline o filtrar combinaciones inválidas con ParameterGrid
```

#### Modelo 2: Random Forest
```python
param_grid_rf = {
    'n_estimators':      [100, 200, 400, 600],
    'max_depth':         [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2', 0.3],
    'class_weight':      ['balanced', 'balanced_subsample']
}
```

#### Modelo 3: Gradient Boosting
```python
param_grid_gb = {
    'n_estimators':  [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth':     [2, 3, 4, 5, 6],
    'subsample':     [0.6, 0.8, 1.0],
    'min_samples_split': [2, 5, 10],
    'max_features':  ['sqrt', 'log2']
}
```

#### Modelo 4: XGBoost
```python
# scale_pos_weight = count(clase 0) / count(clase 1) ≈ 578
param_grid_xgb = {
    'n_estimators':      [100, 200, 300, 500],
    'learning_rate':     [0.01, 0.05, 0.1, 0.2],
    'max_depth':         [3, 4, 5, 6, 7],
    'subsample':         [0.6, 0.7, 0.8, 1.0],
    'colsample_bytree':  [0.6, 0.7, 0.8, 1.0],
    'reg_alpha':         [0, 0.1, 1, 10],       # L1
    'reg_lambda':        [1, 2, 5, 10],          # L2
    'scale_pos_weight':  [1, 50, 100, 289, 578]  # manejo desbalanceo
}
```

#### Modelo 5: SVC
```python
param_grid_svc = {
    'C':      [0.01, 0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear'],
    'gamma':  ['scale', 'auto', 0.001, 0.01, 0.1],
    'class_weight': ['balanced'],
    'probability': [True]   # necesario para predict_proba y AUPRC
}
# SVC es lento en 284k filas. Usar una muestra estratificada de 50k para el grid:
# X_sample, _, y_sample, _ = train_test_split(X_train, y_train,
#     train_size=50000, stratify=y_train, random_state=42)
```

#### Modelo 6: MLP (Red Neuronal)
```python
param_grid_mlp = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
    'activation':         ['relu', 'tanh'],
    'alpha':              [0.0001, 0.001, 0.01, 0.1],  # L2 regularization
    'learning_rate_init': [0.001, 0.01],
    'max_iter':           [200, 500],
    'early_stopping':     [True],
    'validation_fraction':[0.1]
}
```

Figuras en `results/`:
- `ht_lr_heatmap_C_penalty.png`
- `ht_rf_heatmap_depth_estimators.png`
- `ht_gb_heatmap_lr_depth.png`
- `ht_xgb_heatmap_lr_depth.png`
- `ht_xgb_scale_pos_weight.png` — AUPRC por scale_pos_weight
- `ht_svc_heatmap_C_gamma.png`
- `ht_mlp_heatmap_layers_alpha.png`
- `ht_comparativa_mejor_auprc.png` — barplot AUPRC del mejor config de cada modelo

Log `logs/03_hyperparameter_tuning.log`:
```
[HT] scoring: average_precision (AUPRC)
[HT] CV: StratifiedKFold(5)
[HT]
[HT] --- Logistic Regression ---
[HT]   Combinaciones: X  ×  5 folds = Y modelos
[HT]   Mejor AUPRC (CV): X.XXXX
[HT]   Mejores params: {C: X, penalty: 'l2', ...}
[HT]   Tiempo: X.XX s
[HT]
[HT] --- Random Forest ---
[HT]   Combinaciones: X  ×  5 folds = Y modelos
[HT]   Mejor AUPRC (CV): X.XXXX
[HT]   Mejores params: {...}
[HT]   Tiempo: X.XX s
[HT]
... (todos los modelos)
[HT]
[HT] RESUMEN FINAL:
[HT]   Logistic Regression: AUPRC=X.XXXX
[HT]   Random Forest:       AUPRC=X.XXXX
[HT]   Gradient Boosting:   AUPRC=X.XXXX
[HT]   XGBoost:             AUPRC=X.XXXX
[HT]   SVC:                 AUPRC=X.XXXX
[HT]   MLP:                 AUPRC=X.XXXX
[HT] GANADOR GridSearch: [modelo] con AUPRC=X.XXXX
```

---

### `src/04_train_models.py`

Entrenar todos los modelos con sus **mejores hiperparámetros** (cargados desde los JSON).
Split: 80/20 estratificado. StandardScaler fit sobre X_train únicamente.

Para cada modelo calcular:
- Accuracy, Precision-fraude, Recall-fraude, F1-fraude, F1-macro, ROC-AUC, AUPRC
- Tiempo de entrenamiento
- Umbral óptimo de clasificación (maximizar F1-fraude en el conjunto de validación)

Guardar `results/model_results.json`.

Figuras:
- `modelos_comparativa_auprc.png` — barplot AUPRC de los 6 modelos
- `modelos_comparativa_f1_fraude.png` — barplot F1-fraude
- `modelos_comparativa_recall_fraude.png` — barplot recall-fraude
- `modelos_comparativa_tiempos.png` — barplot tiempos de entrenamiento (escala log)
- `modelos_curvas_pr.png` — curvas Precision-Recall de todos los modelos superpuestas
- `modelos_curvas_roc.png` — curvas ROC de todos los modelos superpuestas
- `modelo_ganador_confusion_matrix.png` — matriz de confusión del modelo ganador
- `modelo_ganador_confusion_matrix_normalizada.png` — idem normalizada

Log `logs/04_train_models.log`:
```
[TRAIN] Split: 227845 train / 56962 test (estratificado)
[TRAIN] Fraudes en test: X (X.XXX%)
[TRAIN]
[TRAIN] --- Logistic Regression (mejores params cargados) ---
[TRAIN]   AUPRC:          X.XXXX
[TRAIN]   ROC-AUC:        X.XXXX
[TRAIN]   F1-fraude:      X.XXXX
[TRAIN]   Precision-fraud:X.XXXX
[TRAIN]   Recall-fraude:  X.XXXX
[TRAIN]   Accuracy:       X.XXXX (referencia, no criterio principal)
[TRAIN]   Umbral óptimo:  X.XXX
[TRAIN]   Tiempo train:   X.XXXX s
[TRAIN]   Fraudes detectados: X / X (X.X% de los fraudes en test)
[TRAIN]   Falsas alarmas:     X
...
[TRAIN]
[TRAIN] RANKING por AUPRC:
[TRAIN]   1. XGBoost:            AUPRC=X.XXXX
[TRAIN]   2. Random Forest:      AUPRC=X.XXXX
...
```

---

### `src/05_cross_validation.py`

Implementar manualmente con `StratifiedKFold(n_splits=5)`.
Modelo: el ganador de `04_train_models.py` (leer de `model_results.json`).

**Versión secuencial:** bucle Python puro sobre los 5 folds.
**Versión paralela:** `ProcessPoolExecutor(max_workers=min(5, os.cpu_count()))`.
Cada proceso aplica su propio StandardScaler (fit solo sobre su X_train de fold).

Métricas por fold: AUPRC, F1-fraude, Recall-fraude.
Métricas agregadas: media ± std.

Figuras:
- `cv_metricas_por_fold.png` — lineplot AUPRC + F1 + Recall por fold
- `cv_comparativa_tiempos.png` — barplot tiempo secuencial vs paralelo

Log `logs/05_cross_validation.log`:
```
[CV] Modelo: [ganador]
[CV] StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
[CV]
[CV] ===== SECUENCIAL =====
[CV] Fold 1: AUPRC=X.XXXX  F1-fraud=X.XXXX  Recall=X.XXXX
[CV] Fold 2: AUPRC=X.XXXX  F1-fraud=X.XXXX  Recall=X.XXXX
[CV] Fold 3: AUPRC=X.XXXX  F1-fraud=X.XXXX  Recall=X.XXXX
[CV] Fold 4: AUPRC=X.XXXX  F1-fraud=X.XXXX  Recall=X.XXXX
[CV] Fold 5: AUPRC=X.XXXX  F1-fraud=X.XXXX  Recall=X.XXXX
[CV] Media AUPRC:    X.XXXX ± X.XXXX
[CV] Media F1-fraud: X.XXXX ± X.XXXX
[CV] Tiempo total:   X.XXXX s
[CV]
[CV] ===== PARALELO (ProcessPoolExecutor, N workers) =====
[CV] Fold 1: AUPRC=X.XXXX
...
[CV] Media AUPRC:   X.XXXX ± X.XXXX
[CV] Tiempo total:  X.XXXX s
[CV]
[CV] Speedup:              X.XXx
[CV] Eficiencia CPU:       XX.X%
[CV] Overhead coordinación:X.XXXX s
[CV] Núcleos disponibles:  N
```

---

### `src/06_parallel_benchmark.py`

Tres estrategias, 5 iteraciones cada una:
1. Secuencial (`n_jobs=1`)
2. Joblib implícito (`n_jobs=-1`)
3. `ProcessPoolExecutor` explícito (lotes de estimadores por proceso)

Calcular para cada estrategia: media, std, speedup, eficiencia, overhead.
Calcular speedup teórico Amdahl con fracción serial estimada.

Figuras:
- `benchmark_barplot_tiempos.png` — barplot media ± std de las 3 estrategias
- `benchmark_speedup_real_vs_amdahl.png` — speedup real vs curva teórica de Amdahl
- `benchmark_eficiencia_por_cores.png` — eficiencia en función del número de workers

Log `logs/06_parallel_benchmark.log`:
```
[BENCH] Hardware: N cores físicos
[BENCH] Modelo: RandomForestClassifier (n_estimators=200)
[BENCH] Dataset: X muestras train
[BENCH] Iteraciones por estrategia: 5
[BENCH]
[BENCH] --- SECUENCIAL (n_jobs=1) ---
[BENCH]   i=1: X.XXXXs  i=2: X.XXXXs  ... i=5: X.XXXXs
[BENCH]   Media: X.XXXX s  Std: X.XXXX s
[BENCH]
[BENCH] --- JOBLIB IMPLÍCITO (n_jobs=-1) ---
[BENCH]   Media: X.XXXX s  Std: X.XXXX s
[BENCH]   Speedup: X.XXx   Eficiencia: XX.X%
[BENCH]
[BENCH] --- ProcessPoolExecutor (N procs) ---
[BENCH]   Media: X.XXXX s  Std: X.XXXX s
[BENCH]   Speedup: X.XXx   Eficiencia: XX.X%
[BENCH]   Overhead estimado: X.XXXX s
[BENCH]
[BENCH] Fracción serial estimada (Amdahl): X.XX
[BENCH] Speedup teórico máximo (Amdahl, N=∞): X.XXx
[BENCH] Speedup teórico con N cores:          X.XXx
[BENCH]
[BENCH] Interpretación: ...
```

---

### `src/07_average_times.py`

5 repeticiones de entrenamiento completo para cada uno de los 6 modelos.
Calcular media y std. Guardar `results/average_times.json`.

Figura:
- `tiempos_medios_todos_modelos.png` — barplot con errorbars, escala logarítmica

Log `logs/07_average_times.log`:
```
[TIMES] Logistic Regression:  media=X.XXXXs  std=X.XXXXs  (×5 ejecuciones)
[TIMES] Random Forest:        media=X.XXXXs  std=X.XXXXs
[TIMES] Gradient Boosting:    media=X.XXXXs  std=X.XXXXs
[TIMES] XGBoost:              media=X.XXXXs  std=X.XXXXs
[TIMES] SVC:                  media=X.XXXXs  std=X.XXXXs
[TIMES] MLP:                  media=X.XXXXs  std=X.XXXXs
[TIMES]
[TIMES] Más rápido: [modelo] (X.XXXXs)
[TIMES] Más lento:  [modelo] (X.XXXXs)
[TIMES] Ratio max/min: X.Xx
```

---

### `src/08_dimensionality_reduction.py`

**Contexto académico importante:** V1-V28 ya son componentes PCA aplicado por los autores
del dataset para anonimizar. Este hecho debe discutirse en la memoria.

Dos experimentos:

**Experimento A — PCA adicional sobre V1-V28:**
Reducir las 31 features actuales a k componentes que expliquen 95% de varianza.
Comparar modelo baseline (31 features) vs modelo PCA-reducido (k features).

**Experimento B — Análisis de los componentes PCA existentes:**
Visualizar V1-V28 en 2D/3D. ¿Separan visualmente fraude de legítimas?
Scatter V1 vs V2 coloreado por clase. Scatter 3D de V1, V2, V3.

Figuras:
- `pca_scree_plot.png` — varianza explicada por componente (individual + acumulada)
- `pca_scatter_v1_v2.png` — scatter V1 vs V2 coloreado por clase
- `pca_scatter_3d.png` — scatter 3D V1-V2-V3 (usar mpl_toolkits)
- `pca_comparativa_baseline_vs_pca.png` — barplot AUPRC y F1 baseline vs reducido
- `pca_loadings.png` — loadings de las primeras 3 componentes adicionales

Log `logs/08_dimensionality_reduction.log`:
```
[PCA] Features originales: 31
[PCA] Componentes para 95% varianza: K
[PCA] Varianza acumulada con K componentes: 95.XX%
[PCA]
[PCA] --- BASELINE (31 features) ---
[PCA]   AUPRC:     X.XXXX
[PCA]   F1-fraud:  X.XXXX
[PCA]   Tiempo:    X.XXXX s
[PCA]
[PCA] --- PCA REDUCIDO (K componentes) ---
[PCA]   AUPRC:     X.XXXX
[PCA]   F1-fraud:  X.XXXX
[PCA]   Tiempo:    X.XXXX s
[PCA]
[PCA] Pérdida AUPRC: X.XXXX puntos
[PCA] Reducción dimensiones: XX.X%
[PCA]
[PCA] Nota: V1-V28 son ya componentes PCA aplicados por los autores
[PCA] para anonimizar los datos. Separación visual en V1-V2: [análisis]
```

---

### `src/09_clustering.py`

K-Means sobre las transacciones **sin usar `Class`** durante entrenamiento.
Analizar a posteriori si los clusters capturan el fraude.

Pasos:
1. Escalar features (StandardScaler sobre todo X, sin split — es no supervisado)
2. Reducir a 2D con PCA para visualización (no para clustering)
3. Evaluar k=2..10 con inercia + Silhouette + Davies-Bouldin
4. Modelo final con k óptimo
5. Calcular % de fraude por cluster
6. Calcular pureza, Adjusted Rand Index, Normalized Mutual Information

Figuras:
- `clustering_elbow.png`
- `clustering_silhouette_por_k.png`
- `clustering_davies_bouldin.png`
- `clustering_scatter_2d_clusters.png` — PCA-2D coloreado por cluster
- `clustering_scatter_2d_clase_real.png` — mismo scatter coloreado por clase real
- `clustering_fraude_por_cluster.png` — barplot % fraude por cluster

Log `logs/09_clustering.log`:
```
[CLUST] Evaluando k=2..10 (sin etiquetas de clase)...
[CLUST] k=2: silhouette=X.XXXX  DB=X.XXXX  inercia=XXXXXXX
...
[CLUST] k óptimo: K (max silhouette)
[CLUST]
[CLUST] --- Modelo final (k=K) ---
[CLUST] Silhouette:              X.XXXX
[CLUST] Adjusted Rand Index:     X.XXXX  (vs Class real)
[CLUST] Norm. Mutual Information:X.XXXX
[CLUST]
[CLUST] Distribución de fraude por cluster:
[CLUST]   Cluster 0: N transacciones  fraudes=X (X.XXX%)
[CLUST]   Cluster 1: N transacciones  fraudes=X (X.XXX%)
...
[CLUST] Cluster con mayor concentración de fraude: Cluster X (X.XXX%)
```

---

### `src/10_anomaly_detection.py`

**Este es el análisis estrella del proyecto** — el fraude es detección de anomalías por definición.

Implementar y comparar **tres algoritmos**:
1. `IsolationForest(contamination=0.00172, random_state=42, n_jobs=-1)`
2. `LocalOutlierFactor(n_neighbors=20, contamination=0.00172, novelty=False, n_jobs=-1)`
3. `OneClassSVM(nu=0.00172, kernel='rbf', gamma='scale')`
   (usar muestra de 50k por coste computacional de OneClassSVM)

Para cada algoritmo calcular: Precision-fraude, Recall-fraude, F1-fraude, ROC-AUC.
**Comparar con el mejor modelo supervisado** para contextualizar el resultado.

Figuras:
- `anomaly_comparativa_tres_algoritmos.png` — barplot F1 + Recall + Precision para los 3
- `anomaly_isolation_forest_scores.png` — histograma de scores de anomalía IF
- `anomaly_scatter_if.png` — scatter PCA-2D coloreado por predicción IF
- `anomaly_scatter_lof.png` — idem LOF
- `anomaly_lof_scores.png` — distribución de scores LOF
- `anomaly_supervisado_vs_no_supervisado.png` — comparativa final supervisado vs los 3 anomaly

Log `logs/10_anomaly_detection.log`:
```
[ANOM] contamination = 0.00172 (tasa real de fraude)
[ANOM]
[ANOM] --- Isolation Forest ---
[ANOM]   Anomalías detectadas: X
[ANOM]   Precision-fraude: X.XXXX
[ANOM]   Recall-fraude:    X.XXXX
[ANOM]   F1-fraude:        X.XXXX
[ANOM]   ROC-AUC:          X.XXXX
[ANOM]
[ANOM] --- Local Outlier Factor ---
[ANOM]   Precision-fraude: X.XXXX
[ANOM]   Recall-fraude:    X.XXXX
[ANOM]   F1-fraude:        X.XXXX
[ANOM]
[ANOM] --- One-Class SVM (muestra 50k) ---
[ANOM]   Precision-fraude: X.XXXX
[ANOM]   Recall-fraude:    X.XXXX
[ANOM]   F1-fraude:        X.XXXX
[ANOM]
[ANOM] --- Comparativa supervisado vs no supervisado ---
[ANOM]   Mejor supervisado ([modelo]): F1=X.XXXX  Recall=X.XXXX
[ANOM]   Mejor no supervisado (IF):    F1=X.XXXX  Recall=X.XXXX
[ANOM]   Diferencia F1: +X.XXXX a favor del supervisado
[ANOM]
[ANOM] Interpretación: el modelo supervisado supera a los no supervisados
[ANOM] porque dispone de ejemplos de fraude durante el entrenamiento.
[ANOM] Los métodos no supervisados son relevantes cuando no se dispone
[ANOM] de ejemplos etiquetados de fraude histórico.
```

---

### `src/11_final_analysis.py`

Análisis completo del modelo ganador + dashboard resumen.

Figuras:
- `final_feature_importance.png` — importancia de features (top 20)
- `final_curva_pr_optima.png` — PR curve con umbral óptimo marcado
- `final_curva_roc_optima.png` — ROC curve con punto óptimo marcado
- `final_analisis_umbral.png` — F1, Precision, Recall en función del umbral de decisión
- `final_confusion_matrix.png` — matriz de confusión con umbral óptimo
- `final_fraudes_por_amount.png` — distribución de Amount en fraudes detectados vs no detectados
- `final_fraudes_por_hora.png` — distribución temporal de fraudes detectados vs perdidos
- `final_dashboard_resumen.png` — figura compuesta con todas las métricas clave

Guardar `results/resumen_final.json`.

Log `logs/11_final_analysis.log`:
```
[FINAL] ============================================
[FINAL] RESUMEN EJECUTIVO — FRAUD DETECTION PROJECT
[FINAL] ============================================
[FINAL] Dataset: 284807 transacciones  492 fraudes (0.172%)
[FINAL]
[FINAL] Mejor modelo supervisado: [nombre]
[FINAL]   AUPRC:          X.XXXX
[FINAL]   ROC-AUC:        X.XXXX
[FINAL]   F1-fraude:      X.XXXX
[FINAL]   Recall-fraude:  X.XXXX  (fraudes capturados: X/X)
[FINAL]   Precisión:      X.XXXX  (falsas alarmas: X)
[FINAL]   Umbral óptimo:  X.XXX
[FINAL]
[FINAL] PCA: K componentes explican 95% varianza
[FINAL] Clustering: k=K (ARI=X.XXXX vs etiquetas reales)
[FINAL] Mejor detector anomalías: IF (F1=X.XXXX)
[FINAL] Speedup paralelización: X.XXx (N cores)
[FINAL] ============================================
```

---

### `main.py`

```python
"""
Pipeline completo — Credit Card Fraud Detection
Ejecutar desde la raíz del proyecto: python main.py
"""
import subprocess, sys, time, os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent

def run_script(script_path: Path) -> bool:
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {script_path.name}")
    print('='*60)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT)
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else "ERROR"
    print(f"[{status}] {script_path.name} — {elapsed:.1f}s")
    return result.returncode == 0

if __name__ == '__main__':
    # Verificar dataset
    if not (PROJECT_ROOT / 'data' / 'creditcard.csv').exists():
        print("ERROR: data/creditcard.csv no encontrado.")
        print("Descárgalo de: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        sys.exit(1)

    # Crear directorios
    for d in ['logs', 'results']:
        (PROJECT_ROOT / d).mkdir(exist_ok=True)

    scripts = [
        PROJECT_ROOT / 'src' / f'{i:02d}_{name}.py'
        for i, name in enumerate([
            'eda', 'preprocessing', 'hyperparameter_tuning',
            'train_models', 'cross_validation', 'parallel_benchmark',
            'average_times', 'dimensionality_reduction',
            'clustering', 'anomaly_detection', 'final_analysis'
        ], start=1)
    ]

    results = {}
    total_start = time.time()

    for script in scripts:
        results[script.name] = run_script(script)

    total = time.time() - total_start
    ok = sum(results.values())

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETADO — {ok}/{len(scripts)} scripts OK — {total:.1f}s total")
    print('='*60)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")
```

---

## REQUISITOS TÉCNICOS OBLIGATORIOS

### Sin data leakage
```python
# SIEMPRE en este orden en scripts supervisados:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
```

### Rutas absolutas desde `__file__`
```python
from pathlib import Path
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / 'data'
RESULTS_DIR  = PROJECT_ROOT / 'results'
LOGS_DIR     = PROJECT_ROOT / 'logs'
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
```

### Logging en cada script
```python
import logging
def get_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(f'[{name}] %(message)s')
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
```

### ProcessPoolExecutor — función worker a nivel de módulo
```python
# CORRECTO: a nivel de módulo, serializable por pickle
def _train_fold(args):
    X_tr, X_te, y_tr, y_te, params = args
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    # ... entrenamiento y evaluación ...
    return metrics_dict

# USO:
with ProcessPoolExecutor(max_workers=n_workers) as executor:
    futures = {executor.submit(_train_fold, args): i
               for i, args in enumerate(fold_args)}
    for future in as_completed(futures):
        results[futures[future]] = future.result()
```

### `requirements.txt`
```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
imbalanced-learn>=0.11   # para SMOTE
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
```

---

## ESPECIFICACIÓN DE LA MEMORIA LaTeX

### Paquetes
```latex
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish, es-tabla]{babel}
\usepackage{lmodern}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage[table]{xcolor}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{listings}
\usepackage{hyperref}
\usepackage{subcaption}
\usepackage{float}
\usepackage{fancyhdr}
\usepackage{geometry}
\usepackage{microtype}
```

### Estructura de capítulos

| Cap. | Título | Contenido clave |
|---|---|---|
| 1 | Introducción | Fraude con tarjeta, impacto económico, objetivo, dataset |
| 2 | Fundamentos teóricos | Todo lo necesario antes de usarlo: clasificación binaria, desbalanceo, AUPRC vs accuracy, sobreajuste, CV estratificada, PCA, K-Means, Isolation Forest, paralelización, Ley de Amdahl |
| 3 | Dataset | Origen, 284k transacciones, V1-V28 ya son PCA, Amount/Time, distribución de clases, EDA |
| 4 | Preprocesamiento | Pipeline completo, ingeniería de features, corrección data leakage explicada explícitamente |
| 5 | Ajuste de hiperparámetros | Grids exhaustivos para los 6 modelos, scoring=AUPRC, tablas con mejores params |
| 6 | Modelos supervisados | Uno por sección: fundamento matemático, ventajas/límites, config final, resultados |
| 7 | PCA | Contexto (V1-V28 ya son PCA), experimento adicional, scree plot, biplot, comparativa |
| 8 | Clustering | K-Means sin etiquetas, selección k, ARI/NMI vs etiquetas reales, interpretación |
| 9 | Detección de anomalías | IF + LOF + OneClassSVM, comparativa entre los tres, comparativa vs supervisado |
| 10 | Paralelización | ProcessPoolExecutor vs joblib vs secuencial, Amdahl aplicado a resultados reales |
| 11 | Resultados | Tablas con todos los datos reales de los JSON, todas las figuras |
| 12 | Análisis crítico | Por qué gana cada modelo, limitaciones, coste de no etiquetar datos |
| 13 | Conclusiones | Qué se consiguió, trabajo futuro |
| Bib | Bibliografía | ≥12 referencias académicas en formato correcto |
| Anexo A | Estructura del proyecto | árbol de directorios |
| Anexo B | Fragmentos clave de código | data leakage, ProcessPoolExecutor, GridSearch |
| Anexo C | JSONs de resultados | model_results.json, best_hyperparams completos |

### Ecuaciones obligatorias en la memoria
- Precision, Recall, F1 (para clase fraude)
- AUPRC (integral de la curva PR)
- Descomposición en vectores propios de la covarianza (PCA)
- Inercia de K-Means
- Coeficiente de Silhouette s(i)
- Ley de Amdahl S(N)
- StandardScaler z = (x - μ) / σ
- scale_pos_weight de XGBoost: count(clase 0) / count(clase 1)

### Tablas con datos reales (de los JSON generados)
- Tabla comparativa de los 6 modelos: AUPRC, ROC-AUC, F1-fraud, Recall, Precision, Tiempo
- Tabla mejores hiperparámetros por modelo (de best_hyperparams_*.json)
- Tabla tiempos medios ± std (de average_times.json)
- Tabla comparativa paralelización: tiempo, speedup, eficiencia, overhead
- Tabla PCA: baseline vs reducido
- Tabla clustering: k, silhouette, ARI, NMI, % fraude por cluster
- Tabla anomaly detection: Precision, Recall, F1 para IF, LOF, OC-SVM

### Figuras: incluir TODAS las de `results/`
Cada figura con: párrafo introductorio + `\includegraphics` + `\caption` descriptivo + interpretación.

---

## CHECKLIST ANTES DE ENTREGAR

### Código
- [ ] `python main.py` ejecuta sin errores de principio a fin
- [ ] No hay data leakage en ningún script supervisado
- [ ] `stratify=y` en todos los `train_test_split`
- [ ] `StratifiedKFold` en todos los CV
- [ ] `ProcessPoolExecutor` con función worker a nivel de módulo
- [ ] Grids exhaustivos para los 6 modelos en `03_hyperparameter_tuning.py`
- [ ] Mejores hiperparámetros cargados desde JSON en `04_train_models.py`
- [ ] `random_state=42` en todo
- [ ] Logs generados en `logs/`
- [ ] Figuras generadas en `results/`
- [ ] `requirements.txt` completo

### Memoria
- [ ] Datos reales en todas las tablas (no placeholders)
- [ ] Todas las figuras incluidas con caption + interpretación
- [ ] Data leakage explicado explícitamente
- [ ] Diferencia ProcessPoolExecutor vs joblib explicada
- [ ] Ley de Amdahl aplicada a resultados reales
- [ ] Por qué accuracy no es la métrica principal (explicado formalmente)
- [ ] Por qué scale_pos_weight importa en XGBoost
- [ ] Comparativa supervisado vs no supervisado en anomalías
- [ ] ≥12 referencias bibliográficas
- [ ] Compila con `pdflatex` sin errores en 2 pasadas
