import json
import re
from collections import Counter
from pathlib import Path

import jieba


ROOT = Path(__file__).resolve().parent.parent
INPUT_JSON = ROOT / "combined_annotations.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "word_statistics"

CHAR_FIELD = "人物描述"
SCENE_FIELD = "场景"

COMMON_STOPWORDS = {
    "a", "an", "the", "of", "and", "to", "is", "are", "in", "on", "at", "with", "for",
    "this", "that", "these", "those", "their", "there", "some", "many", "has", "have",
    "be", "it", "as", "from",
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


def load_records() -> list[dict]:
    return json.loads(INPUT_JSON.read_text(encoding="utf-8"))


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in jieba.cut(text):
        token = token.strip().lower()
        if not token:
            continue
        if re.fullmatch(r"[a-z]+(?:-[a-z]+)?", token):
            tokens.append(token)
    return tokens


def build_character_counter(records: list[dict]) -> Counter:
    counter = Counter()
    for record in records:
        text = str(record.get(CHAR_FIELD, "") or "")
        for token in tokenize(text):
            if len(token) > 1 and token not in COMMON_STOPWORDS:
                counter[token] += 1
    return counter


def build_scene_counter(records: list[dict]) -> Counter:
    counter = Counter()
    for record in records:
        text = str(record.get(SCENE_FIELD, "") or "")
        for token in tokenize(text):
            if len(token) > 1 and token not in COMMON_STOPWORDS:
                counter[token] += 1
    return counter


def build_action_counter(records: list[dict]) -> Counter:
    counter = Counter()
    for record in records:
        text = str(record.get(CHAR_FIELD, "") or "")
        tokens = tokenize(text)
        for i, token in enumerate(tokens):
            if token not in ACTION_VERBS:
                continue
            counter[token] += 1
            if i + 1 < len(tokens) and tokens[i + 1] not in COMMON_STOPWORDS:
                counter[f"{token} {tokens[i + 1]}"] += 1
            if i + 2 < len(tokens) and tokens[i + 1] not in COMMON_STOPWORDS and tokens[i + 2] not in COMMON_STOPWORDS:
                counter[f"{token} {tokens[i + 1]} {tokens[i + 2]}"] += 1
    return counter


def write_counter(name: str, counter: Counter) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{name}_word_stats.txt"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"{name}\n")
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
