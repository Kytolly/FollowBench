import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
STATS_DIR = ROOT / "word_statistics_strict"
OUTPUT_SVG = ROOT / "sunburst_hierarchy_matplotlib.svg"

INPUTS = {
    "Scene": STATS_DIR / "scene_word_stats_strict.txt",
    "Action": STATS_DIR / "action_word_stats_strict.txt",
    "Appearance": STATS_DIR / "character_word_stats_strict.txt",
}

CATEGORY_COLORS = {
    "Scene": "#2a6f97",
    "Action": "#6a4c93",
    "Appearance": "#2ec4b6",
}

CHILD_COLORS = {
    "Scene": "#89c2d9",
    "Action": "#b8a1d9",
    "Appearance": "#90e0d0",
}


def parse_stats(path: Path) -> list[tuple[str, int]]:
    items = []
    pattern = re.compile(r"^\s*\d+\.\s+(.*?):\s+(\d+)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            items.append((match.group(1).strip(), int(match.group(2))))
    return items


def main() -> None:
    category_totals = []
    child_sizes = []
    child_labels = []
    child_colors = []

    for category, path in INPUTS.items():
        items = parse_stats(path)
        total = sum(count for _, count in items)
        category_totals.append(total)
        for label, count in items:
            child_sizes.append(count)
            child_labels.append(label)
            child_colors.append(CHILD_COLORS[category])

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(aspect="equal"))

    outer_sizes = category_totals
    outer_labels = list(INPUTS.keys())
    outer_colors = [CATEGORY_COLORS[name] for name in outer_labels]

    outer_wedges, _ = ax.pie(
        outer_sizes,
        radius=1.0,
        labels=outer_labels,
        labeldistance=0.78,
        colors=outer_colors,
        wedgeprops=dict(width=0.28, edgecolor="white"),
        textprops=dict(color="white", fontsize=14, weight="bold"),
        startangle=90,
    )

    ax.pie(
        child_sizes,
        radius=0.72,
        labels=None,
        colors=child_colors,
        wedgeprops=dict(width=0.34, edgecolor="white"),
        startangle=90,
    )

    plt.text(0, 0, "FollowBench\n76 Scene\n45 Action\n63 Appearance", ha="center", va="center", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
