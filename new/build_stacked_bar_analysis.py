import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
NEW_DIR = Path(__file__).resolve().parent
INPUT_JSON = ROOT / "combined_annotations.json"
STRICT_DIR = NEW_DIR / "word_statistics_strict"
OUT_DIR = NEW_DIR / "stacked_bar_analysis"

SPLIT_FIELD = "所属划分集合"
SCENE_FIELD = "场景"
CHAR_FIELD = "人物描述"
VALID_SPLITS = ["train", "test_seen", "test_unseen"]

MIN_FREQ = {
    "action": 2,
    "scene": 1,
    "appearance": 1,
}

TOP_K_READABLE = 30
EXPORT_SVG = True

COLORS = {
    "train": "#355070",
    "test_seen": "#6D597A",
    "test_unseen": "#E56B6F",
}

TERM_FILE_MAP = {
    "action": STRICT_DIR / "action_word_stats_strict.txt",
    "scene": STRICT_DIR / "scene_word_stats_strict.txt",
    "appearance": STRICT_DIR / "character_word_stats_strict.txt",
}

FIELD_MAP = {
    "action": CHAR_FIELD,
    "scene": SCENE_FIELD,
    "appearance": CHAR_FIELD,
}

STOP_IN_PHRASE = {
    "his", "her", "their", "its", "the", "a", "an", "another", "this", "that",
}

CANONICAL_MAP = {
    "gray": "grey",
    "buildings": "building",
    "walls": "wall",
    "windows": "window",
    "cabinets": "cabinet",
    "counters": "counter",
    "shelves": "shelf",
    "cars": "car",
    "trees": "tree",
}

DOMAIN_EXCLUDE_TERMS = {
    # Weak scene adjectives that are common but not category-discriminative.
    "scene": {
        "large",
        "small",
        "modern",
        "open",
        "busy",
    },
    "action": set(),
    "appearance": set(),
}

TOKEN_PATTERN = re.compile(r"[a-z]+(?:-[a-z]+)?")
LINE_PATTERN = re.compile(r"^\s*\d+\.\s+(.*?):\s+(\d+)\s*$")


def load_records() -> list[dict]:
    return json.loads(INPUT_JSON.read_text(encoding="utf-8"))


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def parse_strict_terms(path: Path) -> list[tuple[str, int]]:
    terms: list[tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LINE_PATTERN.match(line)
        if not match:
            continue
        term = " ".join(match.group(1).strip().lower().split())
        freq = int(match.group(2))
        terms.append((term, freq))
    return terms


def clean_term(domain: str, term: str) -> str:
    parts = term.split()
    if domain == "action" and len(parts) > 1:
        parts = [p for p in parts if p not in STOP_IN_PHRASE]
    if len(parts) == 1:
        return CANONICAL_MAP.get(parts[0], parts[0])
    if domain in {"scene", "appearance"} and parts[-1] in CANONICAL_MAP:
        parts[-1] = CANONICAL_MAP[parts[-1]]
    return " ".join(parts).strip()


def build_cleaned_vocab(domain: str) -> tuple[list[str], dict[str, int]]:
    raw_terms = parse_strict_terms(TERM_FILE_MAP[domain])
    merged_freq = Counter()
    for term, freq in raw_terms:
        cleaned = clean_term(domain, term)
        if cleaned:
            merged_freq[cleaned] += freq

    min_freq = MIN_FREQ[domain]
    excluded = DOMAIN_EXCLUDE_TERMS.get(domain, set())
    cleaned_terms = [t for t, f in merged_freq.items() if f >= min_freq and t not in excluded]
    cleaned_terms.sort(key=lambda x: (-merged_freq[x], x))
    return cleaned_terms, dict(merged_freq)


def count_term_occurrence(tokens: list[str], term_tokens: list[str]) -> int:
    n = len(term_tokens)
    if n == 1:
        target = term_tokens[0]
        return sum(1 for token in tokens if token == target)
    count = 0
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == term_tokens:
            count += 1
    return count


def compute_split_counts(records: list[dict], domain: str, terms: list[str]) -> pd.DataFrame:
    term_token_map = {t: t.split() for t in terms}
    matrix = defaultdict(lambda: {split: 0 for split in VALID_SPLITS})

    for rec in records:
        split = str(rec.get(SPLIT_FIELD, "")).strip().lower()
        if split not in VALID_SPLITS:
            continue
        text = str(rec.get(FIELD_MAP[domain], "") or "")
        tokens = tokenize(text)
        for term, tks in term_token_map.items():
            c = count_term_occurrence(tokens, tks)
            if c > 0:
                matrix[term][split] += c

    rows = []
    for term in terms:
        row = {"term": term}
        total = 0
        for split in VALID_SPLITS:
            value = matrix[term][split]
            row[split] = value
            total += value
        row["total"] = total
        row["unseen_only"] = int(row["train"] == 0 and row["test_unseen"] > 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[df["total"] > 0].copy()
    df = df.sort_values(["total", "test_unseen"], ascending=[False, False])
    return df


def write_frequency_outputs(domain: str, df: pd.DataFrame, merged_freq: dict[str, int]) -> None:
    freq_dir = OUT_DIR / "frequencies"
    vocab_dir = OUT_DIR / "cleaned_vocab"
    freq_dir.mkdir(parents=True, exist_ok=True)
    vocab_dir.mkdir(parents=True, exist_ok=True)

    csv_path = freq_dir / f"{domain}_split_counts.csv"
    txt_path = freq_dir / f"{domain}_split_counts.txt"
    vocab_path = vocab_dir / f"{domain}_cleaned_terms.txt"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with txt_path.open("w", encoding="utf-8") as f:
        unseen_only = int(df["unseen_only"].sum())
        f.write(f"{domain}_stacked_bar_stats\n")
        f.write(f"num_terms: {len(df)}\n")
        f.write(f"unseen_only_terms: {unseen_only}\n")
        f.write(f"sum_train: {int(df['train'].sum())}\n")
        f.write(f"sum_test_seen: {int(df['test_seen'].sum())}\n")
        f.write(f"sum_test_unseen: {int(df['test_unseen'].sum())}\n\n")
        for i, row in enumerate(df.itertuples(index=False), start=1):
            f.write(
                f"{i:>4}. {row.term}: "
                f"train={row.train}, test_seen={row.test_seen}, test_unseen={row.test_unseen}, total={row.total}\n"
            )

    with vocab_path.open("w", encoding="utf-8") as f:
        f.write(f"{domain}_cleaned_vocab\n")
        for i, term in enumerate(df["term"].tolist(), start=1):
            f.write(f"{i:>4}. {term}: merged_freq={merged_freq.get(term, 0)}\n")


def plot_stacked_bar(domain: str, df: pd.DataFrame, top_k: int | None = None) -> None:
    vis_dir = OUT_DIR / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)

    data = df.copy()
    suffix = "full"
    if top_k is not None and len(data) > top_k:
        head = data.head(top_k).copy()
        tail = data.iloc[top_k:]
        other = {
            "term": "other_terms",
            "train": int(tail["train"].sum()),
            "test_seen": int(tail["test_seen"].sum()),
            "test_unseen": int(tail["test_unseen"].sum()),
            "total": int(tail["total"].sum()),
            "unseen_only": 0,
        }
        data = pd.concat([head, pd.DataFrame([other])], ignore_index=True)
        suffix = f"top{top_k}"

    x = range(len(data))
    fig_w = max(14, int(len(data) * 0.42))
    fig_h = 7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)

    train = data["train"].to_list()
    seen = data["test_seen"].to_list()
    unseen = data["test_unseen"].to_list()
    labels = data["term"].to_list()

    ax.bar(x, train, color=COLORS["train"], label="Training Set")
    ax.bar(x, seen, bottom=train, color=COLORS["test_seen"], label="Seen Test Set")
    bottom2 = [a + b for a, b in zip(train, seen)]
    ax.bar(x, unseen, bottom=bottom2, color=COLORS["test_unseen"], label="Unseen Test Set")

    ax.set_xlabel(f"{domain.capitalize()} categories")
    ax.set_ylabel("Count of term occurrences")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="upper right")

    # Annotate unseen-only categories for direct zero-shot evidence.
    for idx, row in enumerate(data.itertuples(index=False)):
        if getattr(row, "unseen_only", 0) == 1:
            ax.text(idx, row.total + max(1, row.total * 0.01), "*", ha="center", va="bottom", fontsize=10, color="#9b2226")

    plt.tight_layout()

    png_path = vis_dir / f"{domain}_stacked_bar_{suffix}.png"
    fig.savefig(png_path, bbox_inches="tight")
    if EXPORT_SVG:
        svg_path = vis_dir / f"{domain}_stacked_bar_{suffix}.svg"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def write_summary(all_dfs: dict[str, pd.DataFrame]) -> None:
    out_path = OUT_DIR / "summary.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Stacked bar analysis summary\n")
        for domain in ["scene", "action", "appearance"]:
            df = all_dfs[domain]
            unseen_only = int(df["unseen_only"].sum())
            f.write(
                f"\n[{domain}]\n"
                f"categories={len(df)}\n"
                f"unseen_only_categories={unseen_only}\n"
                f"train_total={int(df['train'].sum())}\n"
                f"test_seen_total={int(df['test_seen'].sum())}\n"
                f"test_unseen_total={int(df['test_unseen'].sum())}\n"
            )


def main() -> None:
    records = load_records()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dfs: dict[str, pd.DataFrame] = {}
    for domain in ["scene", "action", "appearance"]:
        terms, merged_freq = build_cleaned_vocab(domain)
        df = compute_split_counts(records, domain, terms)
        write_frequency_outputs(domain, df, merged_freq)
        plot_stacked_bar(domain, df, top_k=None)
        plot_stacked_bar(domain, df, top_k=TOP_K_READABLE)
        all_dfs[domain] = df

    write_summary(all_dfs)
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
