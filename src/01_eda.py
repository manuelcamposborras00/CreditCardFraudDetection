"""
01_eda.py — Exploratory Data Analysis
Credit Card Fraud Detection
"""

import logging
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import yaml
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from config_schema import AppConfig
from plotting import (
    plot_class_distribution,
    plot_amount_distribution,
    plot_time_hour_distribution,
    plot_correlation_heatmap,
    plot_top_features_boxplot,
    plot_amount_violin,
    plot_mi_ranking,
)


def get_logger(name: str, log_file: Path) -> logging.Logger:
    """Configure and return a logger with file and console handlers.

    Args:
        name: Logger name, used as prefix in formatted messages.
        log_file: Path to the output log file.

    Returns:
        Configured Logger instance. Handlers are added only once to prevent
        duplication on repeated calls.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(f"[{name}] %(message)s")
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def main(project_root: Path = Path(__file__).parent.parent) -> None:
    """Run the full EDA pipeline for a given project root directory.

    Args:
        project_root: Root of the project. Must contain data/, results/,
            logs/ subdirectories and a config.yaml file.
    """
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    logs_dir = project_root / "logs"
    results_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # ── Configuración ──────────────────────────────────────────────────────────
    with open(project_root / "config.yaml", encoding="utf-8") as f:
        cfg = AppConfig.model_validate(yaml.safe_load(f))

    data_file = data_dir / cfg.data.file
    class_labels = cfg.visualization.class_labels
    boxplot_ylim = cfg.visualization.boxplot_ylim
    mi_sample_size = cfg.model.mi_sample_size
    outlier_threshold_amount = cfg.visualization.outlier_threshold_amount

    logger = get_logger("EDA", logs_dir / "01_eda.log")

    # ── Carga ──────────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        logger.error(
            f"Archivo no encontrado: {data_file}. Descarga el dataset desde Kaggle (mlg-ulb/creditcardfraud)."
        )
        raise SystemExit(1)
    logger.info(f"Shape: {df.shape}")

    n0 = (df["Class"] == 0).sum()
    n1 = (df["Class"] == 1).sum()
    ratio = n0 / n1
    logger.info(f"Clase 0 (legítima): {n0}  ({n0 / len(df) * 100:.3f}%)")
    logger.info(f"Clase 1 (fraude):   {n1}  ({n1 / len(df) * 100:.3f}%)")
    logger.info(f"Ratio desbalanceo: {ratio:.0f}:1")
    logger.info(f"Valores nulos: {df.isnull().sum().sum()}")

    amount_leg = df.loc[df["Class"] == 0, "Amount"].mean()
    amount_frau = df.loc[df["Class"] == 1, "Amount"].mean()
    logger.info(f"Amount — media legítima: {amount_leg:.2f} €   media fraude: {amount_frau:.2f} €")
    logger.info(f"Time — rango: [{df['Time'].min():.0f}, {df['Time'].max():.0f}] segundos")

    # ── Feature engineering ────────────────────────────────────────────────────
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Time_hour"] = (df["Time"] / 3600) % 24
    logger.info("Time convertido a hora del día (cíclica 0-24h)")

    # ── Mutual Information ranking ─────────────────────────────────────────────
    v_cols = [f"V{i}" for i in range(1, 29)]
    df_mi = df.sample(n=min(mi_sample_size, len(df)), random_state=42)
    mi_scores = mutual_info_classif(df_mi[v_cols], df_mi["Class"], random_state=42)
    discrimin = pd.Series(mi_scores, index=v_cols).sort_values(ascending=False)
    top5 = ", ".join(discrimin.head(5).index.tolist())
    logger.info(f"Features más discriminativas (Mutual Information): {top5} ...")

    # ── Visualizaciones ────────────────────────────────────────────────────────
    plot_class_distribution(df, results_dir / "eda_distribucion_clases.png")
    plot_amount_distribution(df, results_dir / "eda_distribucion_amount.png")
    plot_time_hour_distribution(df, results_dir / "eda_distribucion_time.png")
    plot_correlation_heatmap(df, discrimin, results_dir / "eda_heatmap_correlaciones.png")
    plot_top_features_boxplot(
        df, discrimin, results_dir / "eda_boxplot_v_features_fraude.png", class_labels, boxplot_ylim
    )
    plot_amount_violin(
        df, results_dir / "eda_amount_por_clase.png", class_labels, outlier_threshold_amount
    )
    plot_mi_ranking(discrimin, results_dir / "eda_features_discriminativas.png")

    logger.info("Figuras guardadas: 7")


if __name__ == "__main__":
    main()
