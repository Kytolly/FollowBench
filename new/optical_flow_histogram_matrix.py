import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
INPUT_JSON = ROOT / "combined_annotations.json"
OUT_DIR = Path(__file__).resolve().parent / "flow_analysis"

SPLIT_FIELD = "所属划分集合"
PATH_FIELD = "路径"
SPLITS = ["train", "test_seen", "test_unseen"]
VIEWS = ["ego", "exo"]

# Runtime controls: for quick iteration keep this finite; set to None for full pass.
MAX_VIDEOS_PER_SPLIT = 220
FRAME_STRIDE = 3
MAX_FRAME_PAIRS_PER_VIDEO = 80
RESIZE_WIDTH = 320

NUM_BINS = 55
EXPORT_SVG = True
TITLE = "Optical Flow Magnitude Histogram Matrix (Ego/Exo x Split)"

SPLIT_COLORS = {
    "train": "#355070",
    "test_seen": "#6D597A",
    "test_unseen": "#E56B6F",
}


def load_records() -> list[dict]:
    return json.loads(INPUT_JSON.read_text(encoding="utf-8"))


def resize_gray(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ratio = width / float(w)
    new_h = int(h * ratio)
    resized = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def flow_magnitudes_for_video(video_path: Path) -> list[float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    mags: list[float] = []
    prev_gray = None
    frame_idx = 0
    used_pairs = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        gray = resize_gray(frame, RESIZE_WIDTH)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
            mags.append(float(np.mean(mag)))
            used_pairs += 1
            if used_pairs >= MAX_FRAME_PAIRS_PER_VIDEO:
                break
        prev_gray = gray
        frame_idx += 1

    cap.release()
    return mags


def collect_split_view_magnitudes(records: list[dict]) -> tuple[dict[tuple[str, str], list[float]], dict[str, int]]:
    samples_per_split = defaultdict(int)
    split_view_mags: dict[tuple[str, str], list[float]] = {(s, v): [] for s in SPLITS for v in VIEWS}

    for rec in records:
        split = str(rec.get(SPLIT_FIELD, "")).strip().lower()
        if split not in SPLITS:
            continue
        if MAX_VIDEOS_PER_SPLIT is not None and samples_per_split[split] >= MAX_VIDEOS_PER_SPLIT:
            continue

        path_info = rec.get(PATH_FIELD, {})
        if not isinstance(path_info, dict):
            continue

        for view in VIEWS:
            rel_path = path_info.get(view)
            if not rel_path:
                continue
            video_path = ROOT / rel_path
            if not video_path.exists():
                continue
            mags = flow_magnitudes_for_video(video_path)
            if mags:
                split_view_mags[(split, view)].extend(mags)

        samples_per_split[split] += 1

    return split_view_mags, dict(samples_per_split)


def safe_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def write_stats(split_view_mags: dict[tuple[str, str], list[float]], used_counts: dict[str, int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "flow_stats_summary.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Optical flow magnitude summary\n")
        f.write(f"max_videos_per_split: {MAX_VIDEOS_PER_SPLIT}\n")
        f.write(f"frame_stride: {FRAME_STRIDE}\n")
        f.write(f"max_frame_pairs_per_video: {MAX_FRAME_PAIRS_PER_VIDEO}\n")
        f.write(f"resize_width: {RESIZE_WIDTH}\n")
        for split in SPLITS:
            f.write(f"\n[{split}] videos_used={used_counts.get(split, 0)}\n")
            for view in VIEWS:
                stats = safe_stats(split_view_mags[(split, view)])
                f.write(
                    f"  {view}: n={stats['n']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                    f"p50={stats['p50']:.4f}, p90={stats['p90']:.4f}, p95={stats['p95']:.4f}, max={stats['max']:.4f}\n"
                )


def plot_hist_matrix(split_view_mags: dict[tuple[str, str], list[float]]) -> None:
    all_values = []
    for key in split_view_mags:
        all_values.extend(split_view_mags[key])

    if not all_values:
        raise RuntimeError("No optical flow values collected. Please check paths and parameters.")

    arr = np.asarray(all_values, dtype=np.float32)
    x_max = float(np.percentile(arr, 99.2))
    x_max = max(x_max, 0.5)
    bins = np.linspace(0, x_max, NUM_BINS)

    fig, axes = plt.subplots(2, 3, figsize=(17, 8), dpi=180, sharex=True, sharey=True)
    for col, split in enumerate(SPLITS):
        for row, view in enumerate(VIEWS):
            ax = axes[row, col]
            values = split_view_mags[(split, view)]
            color = SPLIT_COLORS[split]
            if values:
                ax.hist(values, bins=bins, density=True, alpha=0.88, color=color, edgecolor="white", linewidth=0.3)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.set_title(f"{split} | {view} (n={len(values)})", fontsize=10)
            if col == 0:
                ax.set_ylabel("Density")
            if row == 1:
                ax.set_xlabel("Mean optical flow magnitude")

    fig.suptitle(TITLE, fontsize=14, y=0.995)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "optical_flow_histogram_matrix.png"
    fig.savefig(png_path, bbox_inches="tight")
    if EXPORT_SVG:
        svg_path = OUT_DIR / "optical_flow_histogram_matrix.svg"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    records = load_records()
    split_view_mags, used_counts = collect_split_view_magnitudes(records)
    write_stats(split_view_mags, used_counts)
    plot_hist_matrix(split_view_mags)
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
