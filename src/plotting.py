"""
plotting.py — Visualization functions for the Credit Card Fraud EDA pipeline.

Each function receives an explicit output_path and saves the figure there,
giving the caller full control over naming and location.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

_PALETTE = {"Legítima": "steelblue", "Fraude": "tomato"}


def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart (log scale) showing class imbalance."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["Class"].value_counts().sort_index()
    bars = ax.bar(
        ["Legítima (0)", "Fraude (1)"],
        counts.values,
        color=["steelblue", "tomato"],
        edgecolor="black",
    )
    for bar, val in zip(bars, counts.values):
        pct = val / len(df) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"{val:,}\n({pct:.3f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Número de transacciones (escala log)")
    ax.set_title("Distribución de clases — Credit Card Fraud")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_amount_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Side-by-side KDE of Amount (original) and Amount_log by class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for cls, color, label in [(0, "steelblue", "Legítima"), (1, "tomato", "Fraude")]:
        sns.kdeplot(
            df.loc[df["Class"] == cls, "Amount"],
            ax=axes[0],
            color=color,
            label=label,
            fill=True,
            alpha=0.4,
        )
        sns.kdeplot(
            df.loc[df["Class"] == cls, "Amount_log"],
            ax=axes[1],
            color=color,
            label=label,
            fill=True,
            alpha=0.4,
        )
    axes[0].set_xlim(0, 500)
    axes[0].set_xlabel("Amount (€)")
    axes[0].set_title("Amount original")
    axes[0].legend()
    axes[1].set_xlabel("log1p(Amount)")
    axes[1].set_title("Amount log1p")
    axes[1].legend()
    fig.suptitle("Amount: distribución original vs. log1p")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_time_hour_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """KDE of hour-of-day (Time_hour) by class."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for cls, color, label in [(0, "steelblue", "Legítima"), (1, "tomato", "Fraude")]:
        sns.kdeplot(
            df.loc[df["Class"] == cls, "Time_hour"],
            ax=ax,
            color=color,
            label=label,
            fill=True,
            alpha=0.4,
        )
    ax.set_xlabel("Hora del día (0–24h)")
    ax.set_title("Densidad de hora del día por clase")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, discrimin: pd.Series, output_path: Path) -> None:
    """Correlation heatmap restricted to top-15 MI features + Amount."""
    top15 = discrimin.head(15).index.tolist()
    corr = df[top15 + ["Amount"]].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        xticklabels=True,
        yticklabels=True,
        linewidths=0.3,
        linecolor="grey",
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Correlación entre features — top 15 por Mutual Information + Amount")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_top_features_boxplot(
    df: pd.DataFrame,
    discrimin: pd.Series,
    output_path: Path,
    class_labels: dict[int, str],
    ylim: tuple[float, float],
) -> None:
    """Boxplot of top-10 MI features by class."""
    top10 = discrimin.head(10).index.tolist()
    df_melt = df[top10 + ["Class"]].melt(id_vars="Class", var_name="Feature", value_name="Valor")
    df_melt["Clase"] = df_melt["Class"].map(class_labels)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df_melt,
        x="Feature",
        y="Valor",
        hue="Clase",
        palette=_PALETTE,
        flierprops=dict(markersize=1),
        ax=ax,
    )
    ax.set_title("Boxplot top 10 features por Mutual Information")
    ax.set_ylim(*ylim)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_amount_violin(
    df: pd.DataFrame, output_path: Path, class_labels: dict[int, str], threshold: float
) -> None:
    """Violin plot of Amount by class, clipped at threshold."""
    df_clip = df[df["Amount"] < threshold].copy()
    df_clip["Clase"] = df_clip["Class"].map(class_labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.violinplot(data=df_clip, x="Clase", y="Amount", hue="Clase",
                   palette=_PALETTE, inner="box", legend=False, ax=ax)
    ax.set_title(f"Violin plot de Amount por clase (Amount < {threshold:,.0f} €)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_mi_ranking(discrimin: pd.Series, output_path: Path) -> None:
    """Horizontal bar chart of top-15 features by Mutual Information score."""
    fig, ax = plt.subplots(figsize=(9, 6))
    discrimin.head(15).sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
    ax.set_xlabel("Mutual Information")
    ax.set_title("Features más discriminativas entre fraude y legítima")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
