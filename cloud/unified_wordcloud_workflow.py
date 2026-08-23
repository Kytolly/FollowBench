import json
import random
import re
from collections import Counter
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# =========================
# User Config
# =========================
ROOT = Path(__file__).resolve().parent.parent
INPUT_JSON = ROOT / "combined_annotations.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "unified_outputs"

EXPORT_FORMAT = "svg"  # png / pdf / svg
SHOW_TITLES = False
BACKGROUND_COLOR = "white"
FONT_PATH = "C:/Windows/Fonts/arial.ttf"

COLOR_PALETTE = [
    "#1f3b73",
    "#2a6f97",
    "#2ec4b6",
    "#7bc950",
    "#f4d35e",
    "#6a4c93",
]

OUTPUT_FILES = {
    "character": "character_wordcloud_unified",
    "scene": "scene_wordcloud_unified",
    "action": "action_wordcloud_unified",
}

TITLES = {
    "character": "Character Word Cloud",
    "scene": "Scene Word Cloud",
    "action": "Action Word Cloud",
}

MAX_WORDS = {
    "character": 200,
    "scene": 200,
    "action": 140,
}

WORDCLOUD_OPTIONS = {
    "character": {"prefer_horizontal": 0.85, "relative_scaling": 0.4, "min_font_size": 10},
    "scene": {"prefer_horizontal": 0.85, "relative_scaling": 0.4, "min_font_size": 10},
    "action": {"prefer_horizontal": 0.84, "relative_scaling": 0.35, "min_font_size": 10},
}


# =========================
# Extraction Rules
# =========================
CHAR_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "and", "with", "to", "is", "are", "for",
}
SCENE_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "and", "with", "to", "is", "are", "for",
}

ACTION_VERBS = {
    "stand", "standing", "walk", "walking", "sit", "sitting", "look", "looking", "hold",
    "holding", "carry", "carrying", "push", "pushing", "pull", "pulling", "ride", "riding",
    "cross", "crossing", "wait", "waiting", "talk", "talking", "use", "using", "browse",
    "browsing", "shop", "shopping", "run", "running", "turn", "turning", "approach",
    "approaching", "leave", "leaving", "lean", "leaning", "check", "checking", "face",
    "facing", "move", "moving", "play", "playing", "work", "working", "eat", "eating",
    "drink", "drinking", "queue", "queuing",
}
ACTION_LINKERS = {
    "at", "on", "in", "by", "near", "toward", "towards", "away", "from", "with", "into",
    "through", "across", "along", "around", "up", "down", "behind", "beside", "next", "to",
}
ACTION_OBJECTS = {
    "cart", "table", "desk", "checkout", "machine", "window", "road", "sidewalk", "store",
    "shop", "kitchen", "bike", "scooter", "phone", "bag",
}


def load_records() -> list[dict]:
    return json.loads(INPUT_JSON.read_text(encoding="utf-8"))


def palette_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    rng = random_state or random.Random(42)
    return COLOR_PALETTE[rng.randint(0, len(COLOR_PALETTE) - 1)]


def loose_token_counter(texts: list[str], stopwords: set[str], boosts: dict[str, float]) -> Counter:
    full_text = " ".join(texts)
    words = jieba.cut(full_text)
    counter = Counter()

    for raw in words:
        word = raw.strip().lower()
        if len(word) <= 1 or word in stopwords:
            continue
        counter[word] += 1

    for word, factor in boosts.items():
        if word in counter:
            counter[word] = int(counter[word] * factor)

    return counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:-[a-z]+)?", text.lower())


def build_character_counter(records: list[dict]) -> Counter:
    texts = [str(item.get("人物描述", "") or "") for item in records]
    counter = loose_token_counter(
        texts,
        CHAR_STOPWORDS,
        {
            "person": 1.35,
            "wearing": 1.45,
            "camera": 1.25,
            "walking": 1.15,
            "away": 1.05,
            "black": 1.2,
            "white": 1.15,
            "dark": 1.15,
            "hair": 1.18,
            "pants": 1.2,
            "jacket": 1.18,
            "shirt": 1.12,
            "pack": 1.12,
            "fanny": 1.08,
            "hooded": 1.08,
            "purple": 1.06,
            "grey": 1.05,
        },
    )

    for text in texts:
        tokens = tokenize(text)
        for i in range(len(tokens) - 1):
            phrase = " ".join(tokens[i:i + 2])
            if tokens[i] in {"person", "wearing", "walking", "camera", "dark", "black", "white", "hair", "pants"}:
                counter[phrase] += 1
        for i in range(len(tokens) - 2):
            tri = tokens[i:i + 3]
            phrase = " ".join(tri)
            if tri[0] == "person" and tri[1] in {"wearing", "walking", "standing", "facing"}:
                counter[phrase] += 2
            if tri[0] in {"walking", "facing", "wearing"}:
                counter[phrase] += 1

    for phrase in [
        "person wearing",
        "walking away",
        "walking away from",
        "dark hair",
        "black pants",
        "white shirt",
        "grey pants",
        "hooded jacket",
        "fanny pack",
        "camera view",
        "exo view",
    ]:
        if phrase in counter:
            counter[phrase] += max(3, counter[phrase] // 3)

    return counter


def build_scene_counter(records: list[dict]) -> Counter:
    texts = [str(item.get("场景", "") or "") for item in records]
    counter = loose_token_counter(
        texts,
        SCENE_STOPWORDS,
        {
            "scene": 1.45,
            "visible": 1.3,
            "background": 1.28,
            "takes": 1.12,
            "place": 1.12,
            "right": 1.08,
            "left": 1.08,
            "side": 1.1,
            "modern": 1.12,
            "building": 1.08,
            "buildings": 1.15,
            "shopping": 1.12,
            "mall": 1.12,
            "road": 1.12,
            "walkway": 1.1,
            "kitchen": 1.08,
            "bunk": 1.08,
            "bed": 1.08,
            "urban": 1.08,
            "environment": 1.1,
        },
    )

    for text in texts:
        tokens = tokenize(text)
        for i in range(len(tokens) - 1):
            phrase = " ".join(tokens[i:i + 2])
            if tokens[i] in {"scene", "visible", "right", "left", "shopping", "urban", "paved", "people", "large", "red"}:
                counter[phrase] += 1
        for i in range(len(tokens) - 2):
            tri = tokens[i:i + 3]
            phrase = " ".join(tri)
            if tri[0] == "scene" and tri[1] in {"appears", "takes", "outdoor", "indoor", "kitchen"}:
                counter[phrase] += 3
            if tri[1:] == ["takes", "place"]:
                counter[phrase] += 3
            if tri[0] in {"visible", "shopping", "urban", "people", "paved", "large", "red"}:
                counter[phrase] += 1

    for phrase in [
        "scene appears",
        "takes place",
        "scene takes place",
        "visible background",
        "right side",
        "left side",
        "shopping mall",
        "urban environment",
        "people walking",
        "grassy area",
        "scene outdoor",
        "scene indoor",
        "scene kitchen",
        "bunk bed",
        "paved walkway",
        "paved road",
        "large window",
        "red cabinet",
    ]:
        if phrase in counter:
            counter[phrase] += max(3, counter[phrase] // 3)

    return counter


def build_action_counter(records: list[dict]) -> Counter:
    texts = [str(item.get("人物描述", "") or "") for item in records]
    counter = Counter()

    for text in texts:
        tokens = tokenize(text)
        for i, token in enumerate(tokens):
            if token not in ACTION_VERBS:
                continue
            phrase = [token]
            j = i + 1
            while j < len(tokens) and len(phrase) < 5:
                nxt = tokens[j]
                if nxt in ACTION_LINKERS or nxt in ACTION_OBJECTS:
                    phrase.append(nxt)
                    j += 1
                    continue
                break
            built = " ".join(phrase)
            counter[built] += 1
            for word in phrase:
                counter[word] += 1

    for phrase in ["walking away from", "facing away from", "standing in", "sitting at", "looking at"]:
        if phrase in counter:
            counter[phrase] += max(3, counter[phrase] // 3)

    for word in ["walking", "standing", "facing", "looking", "holding", "carrying", "sitting"]:
        if word in counter:
            counter[word] += max(5, counter[word] // 4)

    return counter


def render_wordcloud(name: str, frequencies: Counter) -> Path:
    output_path = OUTPUT_DIR / f"{OUTPUT_FILES[name]}.{EXPORT_FORMAT}"
    options = WORDCLOUD_OPTIONS[name]
    wc = WordCloud(
        width=1600,
        height=1000,
        background_color=BACKGROUND_COLOR,
        max_words=MAX_WORDS[name],
        prefer_horizontal=options["prefer_horizontal"],
        collocations=False,
        random_state=42,
        margin=2,
        relative_scaling=options["relative_scaling"],
        min_font_size=options["min_font_size"],
        font_path=FONT_PATH,
    ).generate_from_frequencies(dict(frequencies.most_common(MAX_WORDS[name])))

    recolored = wc.recolor(color_func=palette_color_func, random_state=42)
    plt.figure(figsize=(16, 10))
    plt.imshow(recolored, interpolation="bilinear")
    plt.axis("off")
    if SHOW_TITLES:
        plt.title(TITLES[name], fontsize=18)
    plt.tight_layout(pad=0)
    save_kwargs = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0}
    if EXPORT_FORMAT == "svg":
        # matplotlib renders the image inside svg; still useful when a vector container is required
        save_kwargs.pop("dpi", None)
    plt.savefig(output_path, **save_kwargs)
    plt.close()
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()

    outputs = {
        "character": render_wordcloud("character", build_character_counter(records)),
        "scene": render_wordcloud("scene", build_scene_counter(records)),
        "action": render_wordcloud("action", build_action_counter(records)),
    }

    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
