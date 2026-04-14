"""
tests/test_plotting.py — Unit tests for plotting.py

Each test creates a synthetic DataFrame, calls one plotting function
with a tmp_path, and verifies the output file was created.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plotting import (
    plot_amount_distribution,
    plot_amount_violin,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_mi_ranking,
    plot_time_hour_distribution,
    plot_top_features_boxplot,
)

CLASS_LABELS = {0: "Legítima", 1: "Fraude"}
BOXPLOT_YLIM = (-30, 30)
OUTLIER_THRESHOLD_AMOUNT = 2_000


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Synthetic DataFrame mimicking creditcard.csv structure."""
    rng = np.random.default_rng(42)
    n = 300
    v_data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    df = pd.DataFrame(
        {
            "Class": rng.choice([0, 1], size=n, p=[0.95, 0.05]),
            "Amount": np.abs(rng.exponential(100, n)),
            "Time": np.linspace(0, 172_800, n),
            **v_data,
        }
    )
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Time_hour"] = (df["Time"] / 3600) % 24
    return df


@pytest.fixture
def sample_discrimin() -> pd.Series:
    """Synthetic MI ranking over V1–V28."""
    rng = np.random.default_rng(42)
    v_cols = [f"V{i}" for i in range(1, 29)]
    return pd.Series(rng.random(28), index=v_cols).sort_values(ascending=False)


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_plot_class_distribution(sample_df, tmp_path):
    out = tmp_path / "clases.png"
    plot_class_distribution(sample_df, out)
    assert out.exists()


def test_plot_amount_distribution(sample_df, tmp_path):
    out = tmp_path / "amount.png"
    plot_amount_distribution(sample_df, out)
    assert out.exists()


def test_plot_time_hour_distribution(sample_df, tmp_path):
    out = tmp_path / "time.png"
    plot_time_hour_distribution(sample_df, out)
    assert out.exists()


def test_plot_correlation_heatmap(sample_df, sample_discrimin, tmp_path):
    out = tmp_path / "heatmap.png"
    plot_correlation_heatmap(sample_df, sample_discrimin, out)
    assert out.exists()


def test_plot_top_features_boxplot(sample_df, sample_discrimin, tmp_path):
    out = tmp_path / "boxplot.png"
    plot_top_features_boxplot(sample_df, sample_discrimin, out, CLASS_LABELS, BOXPLOT_YLIM)
    assert out.exists()


def test_plot_amount_violin(sample_df, tmp_path):
    out = tmp_path / "violin.png"
    plot_amount_violin(sample_df, out, CLASS_LABELS, OUTLIER_THRESHOLD_AMOUNT)
    assert out.exists()


def test_plot_mi_ranking(sample_discrimin, tmp_path):
    out = tmp_path / "mi.png"
    plot_mi_ranking(sample_discrimin, out)
    assert out.exists()
