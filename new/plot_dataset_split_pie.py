from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager


labels = ["Training Set", "Seen Test Set", "Unseen Test Set"]
sizes = [8644, 1235, 1138]

# Requested palette: light blue, light purple, light red.
colors = ["#A9D6F5", "#D7C6F5", "#F6C1C1"]

out_dir = Path(__file__).resolve().parent / "dataset_split_visuals"
out_dir.mkdir(parents=True, exist_ok=True)
out_png = out_dir / "dataset_split_pie_transparent.png"
out_svg = out_dir / "dataset_split_pie_transparent.svg"

# Set display font
font_dir = Path(__file__).resolve().parent / "fonts"
for font_file in [font_dir / "ComicNeue-Regular.ttf", font_dir / "ComicNeue-Bold.ttf"]:
    if font_file.exists():
        font_manager.fontManager.addfont(str(font_file))
rcParams["font.family"] = "Comic Neue"


def autopct_with_label(lbls):
    idx = {"i": 0}

    def _fmt(pct):
        label = lbls[idx["i"]]
        idx["i"] += 1
        return f"{label}\n{pct:.1f}%"

    return _fmt


fig, ax = plt.subplots(figsize=(7.8, 6.6), dpi=220)
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=None,
    colors=colors,
    autopct=autopct_with_label(labels),
    pctdistance=0.62,
    startangle=95,
    wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    textprops={"fontsize": 11, "color": "#1f2937", "ha": "center", "va": "center"},
)
for t in autotexts:
    t.set_fontsize(11)
    t.set_color("#111827")
ax.axis("equal")

plt.tight_layout()
fig.savefig(out_png, transparent=True, bbox_inches="tight")
fig.savefig(out_svg, transparent=True, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {out_png}")
print(f"Saved: {out_svg}")
