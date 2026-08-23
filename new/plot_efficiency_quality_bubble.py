from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path(r"D:\Desktop\xqy\NUS311\FYP\docs\results\metrics_summary.csv")
OUT_DIR = Path(__file__).resolve().parent / "efficiency_tradeoff"

LATENCY_SEC = {
    "SFT-50": 180.0,
    "SFT-10": 30.0,
    "DMD-4": 10.0,
    "DMD-8": 15.0,
}

METHOD_PATTERNS = {
    "SFT-50": ["sft50", "50_inference"],
    "SFT-10": ["sft10", "10_inference"],
    "DMD-8": ["8steps_inference", "8steps"],
    "DMD-4": ["4steps_inference", "dmd4", "dmd2"],
}

METHOD_COLORS = {
    "DMD-4": "#2A9D8F",
    "DMD-8": "#457B9D",
    "SFT-10": "#E76F51",
    "SFT-50": "#6D597A",
}

METRICS = [
    "LearnedPerceptualImagePatchSimilarity",
    "MeanSquaredError",
    "PeakSignaltoNoiseRatio",
    "StructuralSimilarityIndexMeasure",
]

LOWER_BETTER = {
    "LearnedPerceptualImagePatchSimilarity": True,
    "MeanSquaredError": True,
    "PeakSignaltoNoiseRatio": False,
    "StructuralSimilarityIndexMeasure": False,
}

METRIC_SHORT = {
    "LearnedPerceptualImagePatchSimilarity": "LPIPS",
    "MeanSquaredError": "MSE",
    "PeakSignaltoNoiseRatio": "PSNR",
    "StructuralSimilarityIndexMeasure": "SSIM",
}


def normalize_text(df: pd.DataFrame) -> pd.Series:
    b = df["baseline"].fillna("").astype(str).str.lower()
    c = df["config"].fillna("").astype(str).str.lower()
    m = df["model_name"].fillna("").astype(str).str.lower()
    return b + " " + c + " " + m


def detect_method(row_text: str) -> str | None:
    for method, patterns in METHOD_PATTERNS.items():
        if any(p in row_text for p in patterns):
            return method
    return None


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    x = values.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    if w.sum() <= 0:
        return float(np.mean(x))
    return float((x * w).sum() / w.sum())


def build_method_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method"] = normalize_text(df).map(detect_method)
    df = df[df["method"].isin(LATENCY_SEC.keys())].copy()
    if df.empty:
        raise RuntimeError("No target methods found in metrics_summary.csv")

    rows = []
    for method, g in df.groupby("method"):
        weights = g["num_cases"].fillna(1.0)
        row = {
            "method": method,
            "latency_sec": LATENCY_SEC[method],
            "num_rows": int(len(g)),
            "total_cases": int(weights.sum()),
        }
        for metric in METRICS:
            row[metric] = weighted_mean(g[metric], weights)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("latency_sec")
    return out


def add_normalized_scores(method_df: pd.DataFrame) -> pd.DataFrame:
    out = method_df.copy()
    for metric in METRICS:
        vals = out[metric].to_numpy(dtype=float)
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if np.isclose(lo, hi):
            norm = np.ones_like(vals, dtype=float)
        elif LOWER_BETTER[metric]:
            norm = (hi - vals) / (hi - lo)
        else:
            norm = (vals - lo) / (hi - lo)
        out[f"{metric}_norm"] = norm
    out["quality_mean_norm"] = out[[f"{m}_norm" for m in METRICS]].mean(axis=1)
    return out


def to_long(method_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in method_df.itertuples(index=False):
        for metric in METRICS:
            rows.append(
                {
                    "method": r.method,
                    "latency_sec": r.latency_sec,
                    "total_cases": r.total_cases,
                    "metric_raw": metric,
                    "metric": METRIC_SHORT[metric],
                    "score_raw": getattr(r, metric),
                    "score_norm": getattr(r, f"{metric}_norm"),
                }
            )
    return pd.DataFrame(rows)


def bubble_size_from_cases(cases: np.ndarray) -> np.ndarray:
    x = cases.astype(float)
    lo, hi = x.min(), x.max()
    if np.isclose(lo, hi):
        return np.full_like(x, 420.0)
    return 320.0 + (x - lo) / (hi - lo) * 540.0


def plot_overview(method_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 6.1), dpi=220)
    x = method_df["latency_sec"].to_numpy(dtype=float)
    y = method_df["quality_mean_norm"].to_numpy(dtype=float)
    sizes = bubble_size_from_cases(method_df["total_cases"].to_numpy(dtype=float))
    colors = [METHOD_COLORS[m] for m in method_df["method"]]

    ax.scatter(x, y, s=sizes, c=colors, alpha=0.9, edgecolors="white", linewidths=1.0, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([10, 15, 30, 60, 120, 180])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlim(8, 210)
    ax.set_ylim(0.15, 1.05)
    ax.grid(True, linestyle="--", alpha=0.22, zorder=0)
    ax.set_xlabel("Latency (seconds / video, log scale)")
    ax.set_ylabel("Composite quality score (normalized, higher is better)")
    ax.set_title("Efficiency-Quality Trade-off (Composite)")

    label_offsets = {"DMD-4": (6, 8), "DMD-8": (6, -13), "SFT-10": (6, 8), "SFT-50": (6, 8)}
    for r in method_df.itertuples(index=False):
        dx, dy = label_offsets.get(r.method, (6, 6))
        ax.annotate(r.method, (r.latency_sec, r.quality_mean_norm), textcoords="offset points", xytext=(dx, dy), fontsize=11, weight="bold")

    ax.axvspan(8, 22, ymin=0.68, ymax=1.0, color="#2A9D8F", alpha=0.08, zorder=1)
    ax.text(8.4, 1.01, "Preferred zone", fontsize=9, color="#2A9D8F", va="top")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "efficiency_quality_bubble_improved.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "efficiency_quality_bubble_improved.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_metric_matrix(long_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.3), dpi=220, sharex=True)
    axes = axes.flatten()
    metric_order = ["LPIPS", "MSE", "PSNR", "SSIM"]
    raw_name = {v: k for k, v in METRIC_SHORT.items()}

    for ax, metric in zip(axes, metric_order):
        d = long_df[long_df["metric"] == metric].copy()
        sizes = bubble_size_from_cases(d["total_cases"].to_numpy(dtype=float))
        colors = [METHOD_COLORS[m] for m in d["method"]]
        ax.scatter(d["latency_sec"], d["score_raw"], s=sizes, c=colors, alpha=0.9, edgecolors="white", linewidths=1.0)
        ax.set_xscale("log")
        ax.set_xticks([10, 15, 30, 60, 120, 180])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, linestyle="--", alpha=0.22)
        direction = "lower is better" if LOWER_BETTER[raw_name[metric]] else "higher is better"
        ax.set_title(f"{metric} ({direction})", fontsize=11)
        for r in d.itertuples(index=False):
            ax.annotate(r.method, (r.latency_sec, r.score_raw), textcoords="offset points", xytext=(4, 5), fontsize=9)

    axes[0].set_ylabel("Metric score")
    axes[2].set_ylabel("Metric score")
    axes[2].set_xlabel("Latency (seconds / video, log scale)")
    axes[3].set_xlabel("Latency (seconds / video, log scale)")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLORS[m], markeredgecolor="white", markersize=9, label=m)
        for m in ["DMD-4", "DMD-8", "SFT-10", "SFT-50"]
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Efficiency-Quality Trade-off by Metric", y=1.05, fontsize=14)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "efficiency_quality_metric_matrix.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "efficiency_quality_metric_matrix.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def write_outputs(method_df: pd.DataFrame, long_df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        method_df.to_csv(OUT_DIR / "method_weighted_metrics.csv", index=False, encoding="utf-8-sig")
    except PermissionError:
        method_df.to_csv(OUT_DIR / "method_weighted_metrics_v2.csv", index=False, encoding="utf-8-sig")
    try:
        long_df.to_csv(OUT_DIR / "method_metric_points.csv", index=False, encoding="utf-8-sig")
    except PermissionError:
        long_df.to_csv(OUT_DIR / "method_metric_points_v2.csv", index=False, encoding="utf-8-sig")
    with (OUT_DIR / "readme.txt").open("w", encoding="utf-8") as f:
        f.write("Improved efficiency-quality visualization\n")
        f.write("1) Composite bubble plot with log-scaled latency.\n")
        f.write("2) 2x2 metric matrix using raw metric scales.\n")
        f.write("Latency assumptions (sec/video):\n")
        for k, v in LATENCY_SEC.items():
            f.write(f"  {k}: {v}\n")


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    method_df = build_method_table(df)
    method_df = add_normalized_scores(method_df)
    long_df = to_long(method_df)
    write_outputs(method_df, long_df)
    plot_overview(method_df)
    plot_metric_matrix(long_df)
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
