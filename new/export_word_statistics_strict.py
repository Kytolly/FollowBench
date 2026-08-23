import json
import re
from collections import Counter
from pathlib import Path

import jieba


ROOT = Path(__file__).resolve().parent.parent
INPUT_JSON = ROOT / "combined_annotations.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "word_statistics_strict"

CHAR_FIELD = "人物描述"
SCENE_FIELD = "场景"

COMMON_STOPWORDS = {
    "a", "an", "the", "of", "and", "to", "is", "are", "in", "on", "at", "with", "for",
    "this", "that", "these", "those", "their", "there", "some", "many", "has", "have",
    "be", "it", "as", "from", "or", "by", "into", "onto", "over", "under", "while",
}

CHAR_STOPWORDS = COMMON_STOPWORDS | {
    "person", "people", "someone", "subject", "camera", "view", "frame", "visible", "seen",
    "appears", "another", "center", "middle", "front", "back", "exo", "ego",
    "wearing", "standing", "walking", "sitting", "facing", "looking",
}

SCENE_STOPWORDS = COMMON_STOPWORDS | {
    "scene", "background", "visible", "appears", "showing", "shows", "environment", "place",
    "overall", "impression", "distance", "possibly", "suggests", "suggesting", "indicating",
    "there", "several", "various", "other", "some", "many", "one", "also",
}

ACTION_STOPWORDS = COMMON_STOPWORDS | {
    "person", "people", "someone", "subject", "camera", "view", "frame", "back",
    "near", "around", "through", "across", "along", "towards", "toward", "away", "from",
    "into", "onto", "with", "by", "next", "behind", "beside", "down", "up", "at", "on", "in",
}

CHARACTER_VOCAB = {
    "black", "white", "blue", "red", "green", "yellow", "orange", "purple", "pink", "brown",
    "gray", "grey", "dark", "light", "blonde", "beige", "colored", "dyed", "hooded",
    "hair", "jacket", "coat", "shirt", "hoodie", "sweater", "vest", "top", "dress", "skirt",
    "pants", "jeans", "shorts", "leggings", "shoes", "boots", "sneakers", "hat", "cap",
    "helmet", "mask", "glasses", "scarf", "backpack", "bag", "pack", "fanny", "ponytail",
    "braid", "braids", "bun", "denim", "plaid", "striped", "patterned", "fluffy", "curly",
    "straight", "long", "short", "tied", "waist", "streak", "streaks", "harness",
}

SCENE_VOCAB = {
    "road", "street", "sidewalk", "walkway", "crosswalk", "lane", "highway", "intersection",
    "supermarket", "store", "shop", "mall", "checkout", "station", "platform", "office",
    "room", "hallway", "corridor", "kitchen", "restaurant", "cafe", "park", "parking",
    "lot", "building", "buildings", "window", "windows", "wall", "walls", "floor", "shelf",
    "shelves", "aisle", "table", "desk", "bridge", "vegetation", "trees", "grass", "hedge",
    "streetlight", "bus", "lobby", "playground", "plaza", "campus", "market", "bed", "bunk",
    "urban", "residential", "paved", "busy", "modern", "tiled", "large", "small", "open",
    "overcast", "daytime", "nighttime", "commercial", "public", "elevated", "green", "grassy",
    "indoor", "outdoor", "cabinet", "cabinets", "counter", "counters", "sky", "traffic", "cars",
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

ACTION_OBJECTS = {
    "cart", "table", "desk", "checkout", "machine", "window", "road", "sidewalk", "store",
    "shop", "kitchen", "bike", "scooter", "phone", "bag", "bed", "chair", "counter",
}


def load_records() -> list[dict]:
    return json.loads(INPUT_JSON.read_text(encoding="utf-8"))


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in jieba.cut(text):
        token = token.strip().lower()
        if re.fullmatch(r"[a-z]+(?:-[a-z]+)?", token):
            tokens.append(token)
    return tokens


def build_character_counter(records: list[dict]) -> Counter:
    counter = Counter()
    for record in records:
        tokens = tokenize(str(record.get(CHAR_FIELD, "") or ""))
        for token in tokens:
            if token in CHARACTER_VOCAB and token not in CHAR_STOPWORDS:
                counter[token] += 1
    return counter


def build_scene_counter(records: list[dict]) -> Counter:
    counter = Counter()
    for record in records:
        tokens = tokenize(str(record.get(SCENE_FIELD, "") or ""))
        for token in tokens:
            if token in SCENE_VOCAB and token not in SCENE_STOPWORDS:
                counter[token] += 1
    return counter


def build_action_counter(records: list[dict]) -> Counter:
    counter = Counter()
    for record in records:
        tokens = tokenize(str(record.get(CHAR_FIELD, "") or ""))
        for i, token in enumerate(tokens):
            if token not in ACTION_VERBS:
                continue
            counter[token] += 1

            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if nxt in ACTION_OBJECTS:
                    counter[f"{token} {nxt}"] += 1

            if i + 2 < len(tokens):
                nxt1 = tokens[i + 1]
                nxt2 = tokens[i + 2]
                if nxt1 not in ACTION_STOPWORDS and nxt2 in ACTION_OBJECTS:
                    counter[f"{token} {nxt1} {nxt2}"] += 1

    for key in list(counter.keys()):
        parts = key.split()
        if any(part in ACTION_STOPWORDS for part in parts[1:]):
            del counter[key]
    return counter


def write_counter(name: str, counter: Counter) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{name}_word_stats_strict.txt"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"{name}_strict\n")
        f.write(f"unique_items: {len(counter)}\n")
        f.write(f"total_occurrences: {sum(counter.values())}\n\n")
        for rank, (word, count) in enumerate(counter.most_common(), start=1):
            f.write(f"{rank:>5}. {word}: {count}\n")
    return output_path


def main() -> None:
    records = load_records()
    outputs = {
        "character": write_counter("character", build_character_counter(records)),
        "scene": write_counter("scene", build_scene_counter(records)),
        "action": write_counter("action", build_action_counter(records)),
    }
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
