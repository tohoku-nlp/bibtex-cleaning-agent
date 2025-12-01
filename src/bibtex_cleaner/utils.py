import re
import sys
from typing import Any, Dict, List, Optional
import yaml
from pathlib import Path


def split_bibtex_entries(text: str) -> List[str]:
    parts, current = [], []
    for line in text.splitlines():
        if line.strip().startswith("@") and current:
            parts.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("\n".join(current))
    return [p for p in parts if p.strip().startswith("@")]


def grab(block: str, field: str) -> Optional[str]:
    m = re.search(rf"\b{field}\s*=\s*([^\n,]+)", block, flags=re.IGNORECASE)
    if not m:
        return None
    v = m.group(1).strip().rstrip(",")
    return v.strip('{}"')


def parse_entry(block: str) -> Dict[str, Any]:
    mkey = re.search(r"@\w+\s*{\s*([^,]+)\s*,", block)
    key = mkey.group(1).strip() if mkey else "entry"
    title = grab(block, "title")
    year = grab(block, "year")
    doi = grab(block, "doi")
    authors_raw = grab(block, "author")
    authors: List[str] = []
    if authors_raw:
        for a in re.split(r"\band\b", authors_raw, flags=re.IGNORECASE):
            a = a.strip().strip('{}"')
            if a:
                authors.append(a)
    return {
        "key": key,
        "raw": block,
        "title": title,
        "year": year,
        "doi": doi,
        "authors": authors,
    }


def generate_citation_key(entry: Dict[str, Any], format_str: str) -> str:
    """
    Generate a citation key based on a format string.
    Supported placeholders:
    - {author}: First author's surname
    - {year}: Publication year
    - {title}: Full title (sanitized)
    - {shorttitle}: First 3 meaningful words of title (default)
    - {veryshorttitle}: First 1 meaningful word of title
    - {mediumtitle}: First 5 meaningful words of title
    - {venue}: Venue/journal name
    - {doi}: DOI (sanitized)
    """
    if not format_str:
        return entry.get("key", "unknown")

    def sanitize(s: str) -> str:
        if not s:
            return ""
        # Remove accents/special chars, keep alphanumeric
        s = re.sub(r"[^a-zA-Z0-9]", "", s)
        return s

    # Extract fields
    authors = entry.get("authors", [])
    author = authors[0].split(",")[-1].split()[-1] if authors else "Unknown"
    if "," in (
        authors[0] if authors else ""
    ):  # handle "Last, First" format better if present in list
        parts = authors[0].split(",")
        author = parts[0].strip()

    year = str(entry.get("year") or "Wait")

    title_raw = entry.get("title") or ""
    title = sanitize(title_raw)

    # Smart short titles
    # Filter stop words
    STOP_WORDS = {
        "the",
        "a",
        "an",
        "of",
        "for",
        "in",
        "on",
        "with",
        "to",
        "at",
        "by",
        "and",
        "or",
        "is",
        "are",
        "from",
        "via",
        "using",
    }

    # Split and clean words
    raw_words = re.findall(r"\w+", title_raw.lower())
    meaningful_words = [w for w in raw_words if w not in STOP_WORDS]

    # Fallback if all words were stop words (unlikely but possible)
    if not meaningful_words and raw_words:
        meaningful_words = raw_words

    def get_short_title(num_words: int) -> str:
        selected = meaningful_words[:num_words]
        return "".join(selected) if selected else "untitled"

    shorttitle = get_short_title(3)  # Standard: 3 words
    veryshorttitle = get_short_title(1)  # Very short: 1 word
    mediumtitle = get_short_title(5)  # Medium: 5 words

    venue = sanitize(
        entry.get("venue") or entry.get("journal") or entry.get("booktitle") or ""
    )
    doi = sanitize(entry.get("doi") or "")

    # Replace placeholders
    key = format_str.format(
        author=sanitize(author),
        year=sanitize(year),
        title=title,
        shorttitle=shorttitle,
        veryshorttitle=veryshorttitle,
        mediumtitle=mediumtitle,
        venue=venue,
        doi=doi,
    )

    # Final cleanup
    return key if key else "unknown"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not config_path:
        # Try default locations
        defaults = ["bibtex_cleaner_config.yaml", ".bibtex_cleaner_config.yaml"]
        for d in defaults:
            if Path(d).exists():
                config_path = d
                break

    if not config_path or not Path(config_path).exists():
        return {}

    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(
            f"Warning: Failed to load config file {config_path}: {e}", file=sys.stderr
        )
        return {}
