import re
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent
STATS_DIR = ROOT / "word_statistics_strict"
OUTPUT_HTML = ROOT / "sunburst_hierarchy.html"
OUTPUT_SVG = ROOT / "sunburst_hierarchy.svg"

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


def parse_stats(path: Path) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    pattern = re.compile(r"^\s*\d+\.\s+(.*?):\s+(\d+)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            items.append((match.group(1).strip(), int(match.group(2))))
    return items


def build_sunburst() -> go.Figure:
    labels = ["FollowBench"]
    parents = [""]
    values = [0]
    colors = ["#f8f9fa"]

    total = 0
    for category, path in INPUTS.items():
        items = parse_stats(path)
        category_total = sum(count for _, count in items)
        total += category_total

        labels.append(category)
        parents.append("FollowBench")
        values.append(category_total)
        colors.append(CATEGORY_COLORS[category])

        for label, count in items:
            labels.append(label)
            parents.append(category)
            values.append(count)
            colors.append(CATEGORY_COLORS[category])

    values[0] = total

    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colors=colors),
            insidetextorientation="radial",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Parent: %{parent}<extra></extra>",
            maxdepth=3,
        )
    )

    fig.update_layout(
        title="FollowBench Hierarchy: Scene, Action, Appearance",
        margin=dict(t=60, l=10, r=10, b=10),
        font=dict(size=14),
    )
    return fig


def main() -> None:
    fig = build_sunburst()
    fig.write_html(OUTPUT_HTML)
    try:
        fig.write_image(OUTPUT_SVG)
        print(f"Saved {OUTPUT_HTML}")
        print(f"Saved {OUTPUT_SVG}")
    except Exception:
        print(f"Saved {OUTPUT_HTML}")
        print("SVG export skipped. Install kaleido if you need static vector export.")


if __name__ == "__main__":
    main()
