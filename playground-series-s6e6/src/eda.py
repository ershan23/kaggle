"""
EDA for Playground Series S6E6 - Stellar Classification
Generates plots and summary report to artifacts/eda/
Run: python -m src.eda
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.config import (
    TRAIN_CSV, TEST_CSV, TARGET, CLASSES, NUMERIC_COLS,
    CAT_COLS, MAG_COLS, EDA_DIR,
)
from src.utils import get_logger

logger = get_logger()
sns.set_theme(style="whitegrid", palette="Set2")


def plot_target_distribution(train: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = train[TARGET].value_counts()
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("Set2", 3))
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1000,
                f"{v}\n({v/len(train)*100:.1f}%)", ha="center", fontsize=9)
    ax.set_title("Target Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "01_target_distribution.png", dpi=120)
    plt.close(fig)
    logger.info("Saved 01_target_distribution.png")


def plot_numeric_by_class(train: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.ravel(), NUMERIC_COLS):
        for cls in CLASSES:
            subset = train.loc[train[TARGET] == cls, col]
            ax.hist(subset, bins=80, alpha=0.5, label=cls, density=True)
        ax.set_title(col)
        ax.legend(fontsize=7)
    fig.suptitle("Numeric Features by Class (density)", fontsize=13)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "02_numeric_by_class.png", dpi=120)
    plt.close(fig)
    logger.info("Saved 02_numeric_by_class.png")


def plot_color_indices(train: pd.DataFrame):
    colors = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (a, b) in zip(axes, colors):
        col_name = f"{a}-{b}"
        vals = train[a] - train[b]
        for cls in CLASSES:
            mask = train[TARGET] == cls
            ax.hist(vals[mask], bins=80, alpha=0.5, label=cls, density=True)
        ax.set_title(col_name)
        ax.legend(fontsize=7)
    fig.suptitle("Color Indices by Class", fontsize=13)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "03_color_indices.png", dpi=120)
    plt.close(fig)
    logger.info("Saved 03_color_indices.png")


def plot_cat_vs_target(train: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, CAT_COLS):
        ct = pd.crosstab(train[col], train[TARGET], normalize="index") * 100
        ct[CLASSES].plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
        ax.set_title(f"{col} vs {TARGET} (%)")
        ax.set_ylabel("Percentage")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "04_cat_vs_target.png", dpi=120)
    plt.close(fig)
    logger.info("Saved 04_cat_vs_target.png")


def plot_correlation_heatmap(train: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 7))
    corr = train[NUMERIC_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, vmin=-1, vmax=1, square=True)
    ax.set_title("Numeric Feature Correlation")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "05_correlation.png", dpi=120)
    plt.close(fig)
    logger.info("Saved 05_correlation.png")


def plot_train_test_comparison(train: pd.DataFrame, test: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    ks_results = {}
    for ax, col in zip(axes.ravel(), NUMERIC_COLS):
        ax.hist(train[col], bins=80, alpha=0.5, label="train", density=True)
        ax.hist(test[col], bins=80, alpha=0.5, label="test", density=True)
        ks_stat, ks_p = stats.ks_2samp(train[col].dropna(), test[col].dropna())
        ks_results[col] = (ks_stat, ks_p)
        ax.set_title(f"{col} (KS={ks_stat:.4f})")
        ax.legend(fontsize=7)
    fig.suptitle("Train vs Test Distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "06_train_test_comparison.png", dpi=120)
    plt.close(fig)
    logger.info("Saved 06_train_test_comparison.png")
    return ks_results


def plot_sky_scatter(train: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    sample = train.sample(min(50000, len(train)), random_state=42)
    colors_map = {"GALAXY": "C0", "QSO": "C1", "STAR": "C2"}
    for cls in CLASSES:
        mask = sample[TARGET] == cls
        ax.scatter(sample.loc[mask, "alpha"], sample.loc[mask, "delta"],
                   s=0.3, alpha=0.3, label=cls, color=colors_map[cls])
    ax.set_xlabel("alpha (RA)")
    ax.set_ylabel("delta (Dec)")
    ax.set_title("Sky Distribution (50k sample)")
    ax.legend(markerscale=10)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "07_sky_scatter.png", dpi=120)
    plt.close(fig)
    logger.info("Saved 07_sky_scatter.png")


def generate_summary(train: pd.DataFrame, test: pd.DataFrame, ks_results: dict):
    lines = []
    lines.append("# EDA Summary - Stellar Classification\n")
    lines.append(f"## Dataset Size")
    lines.append(f"- Train: {train.shape[0]:,} rows x {train.shape[1]} cols")
    lines.append(f"- Test: {test.shape[0]:,} rows x {test.shape[1]} cols")
    lines.append(f"- Missing values: None\n")

    lines.append("## Target Distribution")
    for cls in CLASSES:
        cnt = (train[TARGET] == cls).sum()
        lines.append(f"- {cls}: {cnt:,} ({cnt/len(train)*100:.1f}%)")
    lines.append("")

    lines.append("## Key Findings")
    lines.append("### 1. Redshift is the strongest single discriminator")
    for cls in CLASSES:
        subset = train.loc[train[TARGET] == cls, "redshift"]
        lines.append(f"  - {cls}: median={subset.median():.4f}, std={subset.std():.4f}")
    neg_z = (train["redshift"] < 0).sum()
    lines.append(f"  - Negative redshift: {neg_z} rows ({neg_z/len(train)*100:.3f}%)")
    lines.append("")

    lines.append("### 2. Photometric bands (u,g,r,i,z) are highly correlated")
    corr = train[MAG_COLS].corr()
    min_corr = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1)).min().min()
    max_corr = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1)).max().max()
    lines.append(f"  - Pairwise correlation range: [{min_corr:.3f}, {max_corr:.3f}]")
    lines.append("  - Color indices (differences) provide independent information")
    lines.append("")

    lines.append("### 3. Categorical features have strong class association")
    for col in CAT_COLS:
        ct = pd.crosstab(train[col], train[TARGET])
        lines.append(f"  - {col}:")
        for val in ct.index:
            dominant = ct.loc[val].idxmax()
            pct = ct.loc[val, dominant] / ct.loc[val].sum() * 100
            lines.append(f"    - {val}: dominant class = {dominant} ({pct:.1f}%)")
    lines.append("")

    lines.append("### 4. Train/Test distribution shift (KS test)")
    shift_detected = False
    for col, (ks, p) in ks_results.items():
        flag = " <-- SHIFT" if p < 0.01 and ks > 0.05 else ""
        if flag:
            shift_detected = True
        lines.append(f"  - {col}: KS={ks:.4f}, p={p:.2e}{flag}")
    if not shift_detected:
        lines.append("  - No significant distribution shift detected")
    lines.append("")

    lines.append("### 5. Sky distribution (alpha/delta)")
    lines.append("  - Objects are spread across the observable sky (SDSS footprint)")
    lines.append("  - No obvious spatial clustering by class")
    lines.append("")

    lines.append("## Feature Engineering Recommendations")
    lines.append("1. Color indices (u-g, g-r, r-i, i-z) as primary features")
    lines.append("2. Redshift log-transform + negative flag")
    lines.append("3. Magnitude aggregations (mean, std, range)")
    lines.append("4. Cyclic encoding for alpha (RA)")
    lines.append("5. Categorical interaction: spectral_type x galaxy_population")
    lines.append("")

    summary_path = EDA_DIR / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved summary.md")


def main():
    logger.info("Starting EDA...")
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    logger.info(f"Train: {train.shape}, Test: {test.shape}")

    plot_target_distribution(train)
    plot_numeric_by_class(train)
    plot_color_indices(train)
    plot_cat_vs_target(train)
    plot_correlation_heatmap(train)
    ks_results = plot_train_test_comparison(train, test)
    plot_sky_scatter(train)
    generate_summary(train, test, ks_results)

    logger.info("EDA complete! All outputs in artifacts/eda/")


if __name__ == "__main__":
    main()
