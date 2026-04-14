"""
tests/test_integration.py — Integration test for the full EDA pipeline.

Runs main() with a synthetic dataset and a temp project root,
verifying that all 7 expected output figures are produced.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

EXPECTED_FIGURES = [
    "eda_distribucion_clases.png",
    "eda_distribucion_amount.png",
    "eda_distribucion_time.png",
    "eda_heatmap_correlaciones.png",
    "eda_boxplot_v_features_fraude.png",
    "eda_amount_por_clase.png",
    "eda_features_discriminativas.png",
]


def _make_synthetic_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    v_data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    return pd.DataFrame(
        {
            "Class": rng.choice([0, 1], size=n, p=[0.95, 0.05]),
            "Amount": np.abs(rng.exponential(100, n)),
            "Time": np.linspace(0, 172_800, n),
            **v_data,
        }
    )


def _load_main():
    spec = importlib.util.spec_from_file_location("eda", SRC_DIR / "01_eda.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_full_pipeline(tmp_path: Path) -> None:
    # ── Setup temp project structure ──────────────────────────────────────────
    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "logs").mkdir()

    _make_synthetic_df().to_csv(tmp_path / "data" / "creditcard.csv", index=False)

    config = {
        "data": {"file": "creditcard.csv"},
        "model": {"mi_sample_size": 200},
        "visualization": {
            "boxplot_ylim": [-30, 30],
            "outlier_threshold_amount": 2000,
            "class_labels": {0: "Legítima", 1: "Fraude"},
        },
    }
    with open(tmp_path / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    main = _load_main()
    main(tmp_path)

    # ── Assert all figures were created ───────────────────────────────────────
    for fname in EXPECTED_FIGURES:
        assert (tmp_path / "results" / fname).exists(), f"Figura no generada: {fname}"
