import random
import re
import sys
import time
import unicodedata
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from requests.exceptions import ConnectionError

PROCEEDINGS_TYPES = {"proceedings-article"}
JOURNAL_TYPES = {"journal-article"}
ARXIV_LIKE_TYPES = {"posted-content", "report"}


def _confidence(
    title_ok: bool,
    doi_ok: bool,
    authors_overlap: bool,
    year_close: bool,
) -> float:
    """
    Simple confidence score to help the LLM pick among candidates.
    Tunable; weights chosen to prioritize title/DOI match.
    """
    return (
        (1.0 if title_ok else 0.0)
        + (1.0 if doi_ok else 0.0)
        + (0.6 if authors_overlap else 0.0)
        + (0.3 if year_close else 0.0)
    )


class OnlineMetadataResolver:
    def __init__(self, user_agent: str = "BibAI/1.0"):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent})
        self.timeout = (3, 10)

    def retry_with_exponential_backoff(
        self,
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 5,
        errors: tuple = (ConnectionError,),
    ):
        def wrapper(*args, **kwargs):
            num_retries = 0
            delay = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except errors:
                    num_retries += 1
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    delay *= exponential_base * (1 + jitter * random.random())
                    print(
                        f"Connection error occurred. Retrying in {delay:.2f} seconds...",
                        file=sys.stderr
                    )
                    time.sleep(delay)
                except Exception as e:
                    raise e

        return wrapper

    @staticmethod
    def _norm(s: Optional[str]) -> str:
        if not s:
            return ""
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"[{}\"]", "", s)
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _surname_set(names: List[str]) -> set:
        out = set()
        for n in names or []:
            n = n.strip()
            if not n:
                continue
            if "," in n:
                out.add(n.split(",", 1)[0].strip().lower())
            else:
                toks = n.split()
                if toks:
                    out.add(toks[-1].lower())
        return out

    @staticmethod
    def simple_match(original: Dict[str, Any], cand: Dict[str, Any]) -> bool:
        """
        Mark candidate as matched if:
        - exact title OR exact DOI, OR
        - overlapping surnames AND year within ±1
        Also attach a 'confidence' float to cand for downstream use.
        """
        ot = OnlineMetadataResolver._norm(original.get("title")).lower()
        ct = OnlineMetadataResolver._norm(cand.get("title")).lower()
        title_ok = bool(ot and ct and ot == ct)

        od = (original.get("doi") or "").lower()
        cd = (cand.get("doi") or "").lower()
        doi_ok = bool(od and cd and od == cd)

        oy_raw = original.get("year")
        cy_raw = cand.get("year")
        year_close = False
        try:
            if oy_raw and cy_raw:
                oy = int(str(oy_raw))
                cy = int(str(cy_raw))
                year_close = abs(oy - cy) <= 1
        except Exception:
            year_close = bool(oy_raw and cy_raw and str(oy_raw) == str(cy_raw))

        osn = OnlineMetadataResolver._surname_set(original.get("authors") or [])
        csn = OnlineMetadataResolver._surname_set(cand.get("authors") or [])
        authors_overlap = bool(osn and csn and (osn & csn))

        cand["confidence"] = _confidence(
            title_ok=title_ok,
            doi_ok=doi_ok,
            authors_overlap=authors_overlap,
            year_close=year_close,
        )
        return bool(title_ok or doi_ok or (authors_overlap and year_close))

    def crossref_first(
        self, title: str, year: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        from urllib.parse import urlencode

        select = "title,author,DOI,container-title,issued,page,URL,type,abstract"
        params = {"query.title": title, "rows": 1, "select": select}
        if year:
            params["filter"] = f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"
        url = "https://api.crossref.org/works?" + urlencode(params)

        r = self.s.get(url, timeout=self.timeout)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        if not items:
            return None
        it = items[0]

        ctype = (it.get("type") or "").lower()
        if ctype in PROCEEDINGS_TYPES:
            kind = "proceedings"
        elif ctype in JOURNAL_TYPES:
            kind = "journal"
        elif ctype in ARXIV_LIKE_TYPES:
            kind = "arxiv"
        else:
            kind = "other"

        t = (it.get("title") or [""])[0]
        y = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        venue = (it.get("container-title") or [""])[0] or None
        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in it.get("author", [])
            if a.get("family")
        ]

        abstract = it.get("abstract")

        return {
            "title": t or None,
            "authors": authors,
            "year": str(y) if y else None,
            "venue": venue,
            "doi": it.get("DOI"),
            "url": it.get("URL"),
            "pages": it.get("page"),
            "type": ctype,
            "kind": kind,
            "source": "crossref",
            "abstract": abstract,
        }

    def dblp_search(self, title: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search DBLP for papers matching the title, returning multiple results."""
        r = self.s.get(
            "https://dblp.org/search/publ/api",
            params={"q": title, "h": max_results, "format": "json"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        hits = r.json().get("result", {}).get("hits", {}).get("hit", []) or []

        results = []
        for hit in hits:
            info = hit.get("info", {})
            t = info.get("title") or None
            y = info.get("year") or None
            venue = info.get("venue") or None

            authors = info.get("authors", {}).get("author", [])
            if isinstance(authors, dict):
                authors = [authors.get("text") or ""]
            else:
                authors = [a.get("text") or "" for a in authors]

            # Flag arXiv-style venues
            is_arxiv = False
            if venue and ("CoRR" in venue or "arXiv" in venue):
                is_arxiv = True
                kind = "arxiv"
            else:
                kind = "proceedings" if venue else "other"

            results.append(
                {
                    "title": t,
                    "authors": authors,
                    "year": y,
                    "venue": venue,
                    "doi": None,
                    "url": info.get("ee"),
                    "pages": None,
                    "type": "proceedings-article"
                    if venue and not is_arxiv
                    else "posted-content",
                    "kind": kind,
                    "source": "dblp",
                }
            )

        return results

    def dblp_first(self, title: str) -> Optional[Dict[str, Any]]:
        """Get first DBLP result (backward compatibility)."""
        results = self.dblp_search(title, max_results=1)
        return results[0] if results else None

    def semantic_scholar_search(
        self, title: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search Semantic Scholar for papers."""
        params = {
            "query": title,
            "limit": max_results,
            "fields": "title,venue,year,authors,publicationVenue,externalIds,url,abstract",
        }
        try:
            r = self.s.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()

            results = []
            for item in data.get("data", []):
                t = item.get("title")
                y = item.get("year")

                # Prefer explicit publication venue metadata
                venue = item.get("venue")
                pub_venue = item.get("publicationVenue")
                if pub_venue and isinstance(pub_venue, dict):
                    venue = pub_venue.get("name", venue)

                authors = [
                    a.get("name") for a in item.get("authors", []) if a.get("name")
                ]

                ext_ids = item.get("externalIds", {})
                doi = ext_ids.get("DOI")
                arxiv_id = ext_ids.get("ArXiv")

                kind = "other"
                if venue:
                    if "arxiv" in venue.lower() or arxiv_id:
                        kind = "arxiv"
                    else:
                        kind = "proceedings"  # Treat named venues as proceedings
                elif arxiv_id:
                    kind = "arxiv"
                    venue = f"arXiv:{arxiv_id}"

                results.append(
                    {
                        "title": t,
                        "authors": authors,
                        "year": str(y) if y else None,
                        "venue": venue,
                        "doi": doi,
                        "url": item.get("url"),
                        "pages": None,
                        "type": "proceedings-article"
                        if kind == "proceedings"
                        else "posted-content",
                        "kind": kind,
                        "source": "semantic_scholar",
                        "abstract": item.get("abstract"),
                        "arxiv_id": arxiv_id,
                    }
                )
            return results
        except Exception:
            return []

    def arxiv_search(self, title: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search arXiv for papers matching the title."""
        from urllib.parse import quote

        api_url = f"http://export.arxiv.org/api/query?search_query=ti:{quote(title)}&max_results={max_results}"

        try:
            r = self.s.get(api_url, timeout=self.timeout)
            r.raise_for_status()

            papers = BeautifulSoup(r.content, "xml").find_all("entry")
            results = []

            for paper in papers:
                title_text = " ".join(paper.find("title").get_text(strip=True).split())
                authors = [
                    author.find("name").get_text(strip=True)
                    for author in paper.find_all("author")
                ]
                abstract = paper.find("summary").get_text(strip=True)
                year = paper.find("published").get_text(strip=True)[:4]
                arxiv_id = paper.find("id").get_text(strip=True).split("/")[-1]

                results.append(
                    {
                        "title": title_text,
                        "authors": authors,
                        "year": year,
                        "venue": f"arXiv {year}",
                        "doi": None,
                        "url": f"https://arxiv.org/abs/{arxiv_id}",
                        "pages": None,
                        "type": "posted-content",
                        "kind": "arxiv",
                        "source": "arxiv",
                        "abstract": abstract,
                        "arxiv_id": arxiv_id,
                    }
                )

            return results
        except Exception:
            return []

    def generate_bibtex(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Generate BibTeX entry from metadata."""
        if not metadata.get("title") or not metadata.get("authors"):
            return None

        source = metadata.get("source", "")
        kind = metadata.get("kind", "")

        if source == "arxiv" or kind == "arxiv":
            entry_type = "article"
            key_prefix = "arxiv"
        elif kind == "proceedings":
            entry_type = "inproceedings"
            key_prefix = "conf"
        elif kind == "journal":
            entry_type = "article"
            key_prefix = "journal"
        else:
            entry_type = "misc"
            key_prefix = "misc"

        first_author = metadata["authors"][0] if metadata["authors"] else "Unknown"
        if ", " in first_author:
            surname = first_author.split(",")[0]
        else:
            surname = first_author.split()[-1] if first_author.split() else "Unknown"
        surname = re.sub(r"[^a-zA-Z]", "", surname).lower()
        year = metadata.get("year", "????")
        cite_key = f"{surname}{year}{key_prefix}"

        authors_str = " and ".join(metadata["authors"])

        bibtex = f"@{entry_type}{{{cite_key},\n"
        bibtex += f"    title = {{{{{metadata['title']}}}}},\n"
        bibtex += f"    author = {{{authors_str}}},\n"
        bibtex += f"    year = {{{year}}},\n"

        if metadata.get("venue"):
            if kind == "proceedings":
                bibtex += f"    booktitle = {{{metadata['venue']}}},\n"
            else:
                bibtex += f"    journal = {{{metadata['venue']}}},\n"

        if metadata.get("pages"):
            bibtex += f"    pages = {{{metadata['pages']}}},\n"

        if metadata.get("doi"):
            bibtex += f"    doi = {{{metadata['doi']}}},\n"

        if metadata.get("url"):
            bibtex += f"    url = {{{metadata['url']}}},\n"

        if source == "arxiv" and metadata.get("arxiv_id"):
            bibtex += f"    eprint = {{{metadata['arxiv_id']}}},\n"
            bibtex += "    archivePrefix = {arXiv},\n"

        bibtex = bibtex.rstrip(",\n") + "\n}"
        return bibtex

    def resolve(
        self,
        original: Dict[str, Any],
        sources: List[str],
        stop_on_match: bool = True,
        include_abstract: bool = False,
        generate_bibtex: bool = False,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Try sources in order; append each candidate (with matched/ confidence).
        If stop_on_match=True, stop only when the match is a peer-reviewed venue
        (kind in {'proceedings','journal'}). This avoids stopping early on arXiv.

        Args:
            original: Original metadata to match against
            sources: List of sources to search (e.g., ['crossref', 'dblp', 'arxiv', 'semantic_scholar'])
            stop_on_match: Whether to stop on first peer-reviewed match
            include_abstract: Whether to include abstracts (already included for CrossRef and arXiv)
            generate_bibtex: Whether to generate BibTeX entries
            max_results: Maximum results per source (only for sources supporting multiple results)
        """
        out: List[Dict[str, Any]] = []
        title = original.get("title") or ""
        year = original.get("year")

        for src in sources:
            try:
                if src == "crossref":
                    cand = self.crossref_first(title, year)
                    candidates = [cand] if cand else []
                elif src == "dblp":
                    if max_results > 1:
                        candidates = self.dblp_search(title, max_results=max_results)
                    else:
                        cand = self.dblp_first(title)
                        candidates = [cand] if cand else []
                elif src == "arxiv":
                    candidates = self.arxiv_search(title, max_results=max_results)
                elif src == "semantic_scholar":
                    candidates = self.semantic_scholar_search(
                        title, max_results=max_results
                    )
                else:
                    candidates = []
            except Exception:
                candidates = []

            for cand in candidates:
                cand["matched"] = self.simple_match(original, cand)

                if generate_bibtex and cand.get("matched"):
                    cand["bibtex"] = self.generate_bibtex(cand)

                out.append(cand)

                if stop_on_match and cand["matched"]:
                    # Stop only when the match comes from a peer-reviewed venue.
                    if cand.get("kind") in {"proceedings", "journal"}:
                        return out

        return out

    def collect_detailed_info(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Collect detailed paper information from a URL (similar to collect_paper_info).
        Currently supports arXiv URLs with potential for expansion.
        """
        # Normalize URL
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        if "arxiv.org" in url:
            # Extract arXiv ID
            match = re.search(
                r"arxiv.org/(?:abs|pdf)/(\d{4}\.\d+|[a-z-]+/\d+)(?:v\d+)?/?$", url
            )
            if match:
                arxiv_id = match.group(1)
                # Use arXiv API to get detailed info
                api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

                try:
                    r = self.s.get(api_url, timeout=self.timeout)
                    r.raise_for_status()

                    paper = BeautifulSoup(r.content, "xml").find("entry")
                    if paper:
                        title = " ".join(
                            paper.find("title").get_text(strip=True).split()
                        )
                        authors = [
                            author.find("name").get_text(strip=True)
                            for author in paper.find_all("author")
                        ]
                        abstract = paper.find("summary").get_text(strip=True)
                        year = paper.find("published").get_text(strip=True)[:4]

                        metadata = {
                            "title": title,
                            "authors": authors,
                            "year": year,
                            "venue": f"arXiv {year}",
                            "doi": None,
                            "url": url,
                            "abstract": abstract,
                            "source": "arxiv",
                            "kind": "arxiv",
                            "arxiv_id": arxiv_id,
                        }

                        # Generate BibTeX
                        metadata["bibtex"] = self.generate_bibtex(metadata)

                        return metadata
                except Exception:
                    pass

        return None
