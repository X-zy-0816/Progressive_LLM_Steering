"""
Generate BeShift prompts by combining templates with movie, book, and music lists.
Produces positive prompts (from curated positive lists) and negative prompts
(from negative/low-quality lists).
"""

import csv
import os
import random
from typing import List, Dict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "beshift"))
TEMPLATE_FILE = os.path.join(BASE_DIR, "template.csv")

MOVIE_LIST_FILE = os.path.join(BASE_DIR, "posMovielist.csv")
BOOK_LIST_FILE = os.path.join(BASE_DIR, "posBooklist.csv")

NEG_MOVIE_LIST_FILE = os.path.join(BASE_DIR, "negMovielist.csv")
NEG_BOOK_LIST_FILE = os.path.join(BASE_DIR, "negBooklist.csv")
NEG_MUSIC_LIST_FILE = os.path.join(BASE_DIR, "negMusiclist.csv")

OUTPUT_FILE = os.path.join(BASE_DIR, "template_filled.csv")
OUTPUT_FILE_NEG = os.path.join(BASE_DIR, "template_filled_neg.csv")

POS_TARGET_COUNT = 340
NEG_TARGET_COUNT = 300


def _normalize_header(name: str) -> str:
    return (name or "").lstrip("\ufeff").strip().lower()


def read_templates(filepath: str) -> Dict[str, List[str]]:
    templates_by_type: Dict[str, List[str]] = {"Movie": [], "Book": [], "Music": []}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header_map: Dict[str, str] = {}
        if reader.fieldnames:
            for h in reader.fieldnames:
                header_map[_normalize_header(h)] = h
        type_key = header_map.get("type", "Type")
        prompt_key = header_map.get("prompt", "Prompt")

        for row in reader:
            ttype = (row.get(type_key) or "").strip()
            prompt = (row.get(prompt_key) or "").strip()
            if not prompt:
                continue
            tnorm = ttype.lower()
            if tnorm.startswith("movie"):
                templates_by_type["Movie"].append(prompt)
            elif tnorm.startswith("book"):
                templates_by_type["Book"].append(prompt)
            elif tnorm.startswith("music"):
                templates_by_type["Music"].append(prompt)
    return templates_by_type


def read_list(filepath: str) -> List[str]:
    items: List[str] = []
    if not os.path.exists(filepath):
        return items
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header_map: Dict[str, str] = {}
        if reader.fieldnames:
            for h in reader.fieldnames:
                header_map[_normalize_header(h)] = h
        name_field = header_map.get("name", "name")
        for row in reader:
            name_val = (row.get(name_field) or "").strip()
            if name_val:
                items.append(name_val)
    return items


def fill_movie_prompt(template: str, title_with_year: str) -> str:
    year = None
    title = title_with_year
    if title_with_year.endswith(")") and "(" in title_with_year:
        left_paren = title_with_year.rfind("(")
        maybe_year = title_with_year[left_paren + 1 : -1]
        if maybe_year.isdigit() and len(maybe_year) == 4:
            year = maybe_year
            title = title_with_year[: left_paren].strip()
    prompt = template.replace("[Movie Title]", title)
    if "[Year]" in prompt:
        prompt = prompt.replace("[Year]", year or "")
        prompt = prompt.replace("()", "").replace("( )", "")
    return prompt


def fill_book_prompt(template: str, book_entry: str) -> str:
    return template.replace("[Book Title]", book_entry)


def fill_music_prompt(template: str, music_entry: str) -> str:
    return template.replace("[Music Title]", music_entry)


def _sample_to_target(pool: List[str], target: int, rng: random.Random) -> List[str]:
    """Sample from pool to reach target count (with replacement if pool is smaller)."""
    if not pool:
        return []
    if len(pool) >= target:
        return rng.sample(pool, target)
    else:
        return [rng.choice(pool) for _ in range(target)]


def _generate_generic(
    movie_list: List[str],
    book_list: List[str],
    music_list: List[str],
    templates: Dict[str, List[str]],
    target_count: int,
) -> List[Dict[str, str]]:
    rng = random.Random(42)
    movie_templates = templates.get("Movie", [])
    book_templates = templates.get("Book", [])
    music_templates = templates.get("Music", [])

    candidates: List[str] = []

    for m in movie_list:
        for t in movie_templates:
            candidates.append(fill_movie_prompt(t, m))
    for b in book_list:
        for t in book_templates:
            candidates.append(fill_book_prompt(t, b))
    for mu in music_list:
        for t in music_templates:
            candidates.append(fill_music_prompt(t, mu))

    candidates = list(set(candidates))
    rng.shuffle(candidates)

    sampled = _sample_to_target(candidates, target_count, rng)
    return [{"question": s} for s in sampled]


def generate_prompts() -> List[Dict[str, str]]:
    templates = read_templates(TEMPLATE_FILE)
    movies = read_list(MOVIE_LIST_FILE)
    books = read_list(BOOK_LIST_FILE)
    musics: List[str] = []
    return _generate_generic(movies, books, musics, templates, POS_TARGET_COUNT)


def generate_neg_prompts() -> List[Dict[str, str]]:
    templates = read_templates(TEMPLATE_FILE)
    movies = read_list(NEG_MOVIE_LIST_FILE)
    books = read_list(NEG_BOOK_LIST_FILE)
    musics = read_list(NEG_MUSIC_LIST_FILE)
    return _generate_generic(movies, books, musics, templates, NEG_TARGET_COUNT)


def write_csv(rows: List[Dict[str, str]], outfile: str) -> None:
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows_pos = generate_prompts()
    write_csv(rows_pos, OUTPUT_FILE)
    print(f"Generated {len(rows_pos)} positive prompts -> {OUTPUT_FILE}")

    rows_neg = generate_neg_prompts()
    write_csv(rows_neg, OUTPUT_FILE_NEG)
    print(f"Generated {len(rows_neg)} negative prompts -> {OUTPUT_FILE_NEG}")


if __name__ == "__main__":
    main()
