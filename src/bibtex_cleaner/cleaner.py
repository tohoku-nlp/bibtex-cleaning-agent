"""
LangChain-based BibTeX Cleaner

This module provides advanced BibTeX cleaning functionality using LangChain
with support for OpenAI.
"""

import json
import os
import re
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_openai import ChatOpenAI
except ImportError as e:
    print(
        "Error: LangChain dependencies not installed. Run 'uv sync' to install dependencies.",
        file=sys.stderr,
    )
    print(f"Missing: {e}", file=sys.stderr)
    sys.exit(1)

from .metadata import OnlineMetadataResolver
from .utils import grab, parse_entry, split_bibtex_entries, generate_citation_key


class TokenTracker:
    """Track token usage for OpenAI."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self._lock = threading.Lock()

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 chars for English)."""
        return len(text) // 4

    def add_request(self, input_text: str, output_text: str):
        """Add a request to tracking."""
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)

        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_requests += 1

        return input_tokens, output_tokens

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "provider": "openai",
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "avg_input_tokens_per_request": self.total_input_tokens
            / max(self.total_requests, 1),
            "avg_output_tokens_per_request": self.total_output_tokens
            / max(self.total_requests, 1),
        }


class LangChainBibTeXCleaner:
    """LangChain-based BibTeX cleaner using OpenAI."""

    def __init__(
        self,
        custom_prompt: Optional[str] = None,
        batch_size: int = 1,
        delay_between_requests: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the LangChain BibTeX cleaner.

        Args:
            custom_prompt: Custom system prompt (optional)
            batch_size: Number of entries to process at once (default: 1)
            delay_between_requests: Delay in seconds between API calls (default: 1.0)
            **kwargs: OpenAI specific configuration (api_key, model)
        """
        self.batch_size = batch_size
        self.delay_between_requests = delay_between_requests
        self.llm = self._initialize_llm(**kwargs)
        self.prompt_template = self._create_prompt_template(custom_prompt)
        self.token_tracker = TokenTracker()

    def _initialize_llm(self, **kwargs):
        """Initialize ChatOpenAI."""
        api_key = kwargs.get("api_key")
        default_model = os.environ.get("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
        model = kwargs.get("model") or default_model

        print("\n" * 2, file=sys.stderr)
        print(f"Using OpenAI Model: {model}", file=sys.stderr)

        return ChatOpenAI(api_key=api_key, model=model, temperature=0, seed=42)

    def _create_prompt_template(
        self, custom_prompt: Optional[str] = None
    ) -> ChatPromptTemplate:
        """Create the prompt template for BibTeX cleaning."""
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            system_prompt = self._load_default_prompt()

        human_prompt = """Clean the following BibTeX entries:

{bibtex_content}"""

        return ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )

    @staticmethod
    def _entry_type(block: str) -> str:
        m = re.match(r"\s*@(\w+)", block)
        return m.group(1).lower() if m else ""

    def _collect_missing_required_fields(
        self, entries: List[str]
    ) -> List[Dict[str, Any]]:
        expectations = {
            "inproceedings": ["author", "title", "booktitle", "pages", "year"],
            "article": ["author", "title", "journal", "year"],
            "book": ["author", "title", "publisher", "year"],
            "misc": ["author", "title", "year", "url"],
        }
        report: List[Dict[str, Any]] = []
        for block in entries:
            entry_type = self._entry_type(block)
            required = expectations.get(entry_type)
            if not required:
                continue
            missing = []
            for field in required:
                val = grab(block, field)
                if not val or not val.strip():
                    missing.append(field)

            if missing:
                key = parse_entry(block).get("key", "unknown")
                report.append({"key": key, "entry_type": entry_type, "fields": missing})
        return report

    def _load_default_prompt(self) -> str:
        """Load the default system prompt from file."""
        try:
            prompt_path = (
                Path(__file__).parent / "prompts" / "bibtex_cleaning_prompt.md"
            )
            with open(prompt_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
                content_lines = []
                for line in lines[2:]:
                    content_lines.append(line)
                return "\n".join(content_lines)
        except FileNotFoundError:
            return self._get_fallback_prompt()

    def _get_fallback_prompt(self) -> str:
        """Get fallback prompt when prompt file is not available."""
        return """You are a BibTeX expert. Clean and format the provided BibTeX entries following these instructions:

**Overall Instructions:**
- Do not add any new contents to the entries
- Do not remove any entire entries, even when cleaning multiple entries
- Do not change each entry name
- Do not add missing fields with fabricated information
- Remove fields: "month", "note", "key", "abstract", "editor", "address", "publisher"
- Use curly braces instead of double quotes
- Use double curly braces for "title" field to protect the entire title
- Output only the cleaned BibTeX entries without any other text"""

    def _load_conference_enhancement_prompt(self) -> str:
        """Load the conference enhancement prompt from file."""
        try:
            prompt_path = (
                Path(__file__).parent / "prompts" / "conference_enhancement_prompt.md"
            )
            with open(prompt_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
                content_lines = []
                for line in lines[2:]:
                    content_lines.append(line)
                return "\n".join(content_lines)
        except FileNotFoundError:
            return self._get_fallback_conference_prompt()

    def _get_fallback_conference_prompt(self) -> str:
        """Get fallback conference enhancement prompt when prompt file is not available."""
        return """You are a research publication expert. Your task is to enhance conference and journal information in BibTeX entries.

**Instructions:**
- Focus specifically on improving conference/journal names and venues
- Standardize conference names to their official forms
- Add proper abbreviations for well-known conferences
- Fix incomplete or inconsistent venue information
- Do not modify other fields unless directly related to venue information
- Output only the enhanced BibTeX entries"""

    def _process_single_entry(
        self,
        block: str,
        resolver: OnlineMetadataResolver,
        sources: List[str],
        retry_count: int = 3,
        entry_index: int = 0,
    ) -> Optional[str]:
        """
        Process a single BibTeX entry with retry logic and rate limiting.

        Args:
            block: Single BibTeX entry text
            resolver: OnlineMetadataResolver instance
            sources: List of metadata sources
            retry_count: Number of retries for failed requests
            entry_index: Index of current entry for logging

        Returns:
            Processed entry text or None if processing fails
        """
        for attempt in range(retry_count + 1):
            try:
                pe = parse_entry(block)
                original = {
                    "title": pe.get("title"),
                    "year": pe.get("year"),
                    "doi": pe.get("doi"),
                    "authors": pe.get("authors") or [],
                }

                candidates = resolver.resolve(original, sources, stop_on_match=True)

                human_payload = (
                    "ORIGINAL_ENTRY:\n"
                    f"{block}\n\n"
                    "METADATA_CANDIDATES (JSON):\n"
                    f"{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
                    "Return the corrected single BibTeX entry only (no commentary)."
                )

                if attempt > 0:
                    delay_time = self.delay_between_requests * (2**attempt)
                    time.sleep(delay_time)
                elif self.delay_between_requests > 0:
                    time.sleep(self.delay_between_requests)

                chain = self.prompt_template | self.llm
                result = chain.invoke({"bibtex_content": human_payload})
                entry_text = (
                    result.content if hasattr(result, "content") else str(result)
                )

                self.token_tracker.add_request(human_payload, entry_text)

                return entry_text.strip()

            except Exception as e:
                if attempt == retry_count:
                    tqdm.write(f"Entry {entry_index + 1} failed: {str(e)}")
                    return None

                if "ThrottlingException" in str(e) or "Too many requests" in str(e):
                    if attempt < retry_count:
                        wait_time = self.delay_between_requests * (3 ** (attempt + 1))
                        tqdm.write(f"Rate limit hit, waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue

        return None

    def clean_bibtex(
        self,
        bibtex_content: str,
        key_format: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Iteratively clean each BibTeX entry using the existing system prompt,
        plus online metadata (Crossref -> DBLP -> S2 -> ArXiv) for each entry. Always on.
        """
        try:
            overall_start_time = time.time()

            resolver = OnlineMetadataResolver(user_agent="BibAI/1.0")
            if sources is None:
                sources = ["dblp", "semantic_scholar", "arxiv"]

            blocks = split_bibtex_entries(bibtex_content)
            corrected_entries: List[str] = []
            total_blocks = len(blocks)

            print(
                f"Detected {total_blocks} entries. Starting cleaning...",
                file=sys.stderr,
            )

            # Initialize seen_fps as a dict mapping footprint -> canonical_key
            seen_fps: Dict[tuple, str] = {}

            # Collect deduplication/renaming stats
            # Format: {"original_key": "new_key", "duplicate_of": "primary_key_if_dupe"}
            key_mapping: Dict[str, Any] = {}

            pbar = tqdm(total=total_blocks, desc="Cleaning entries", unit="entry")

            def process_func(item):
                res = self._process_single_entry(
                    item["block"], resolver, sources, entry_index=item["index"]
                )
                pbar.update(1)
                return res

            inputs = [{"block": block, "index": i} for i, block in enumerate(blocks)]

            cleaned_results = RunnableLambda(process_func).batch(
                inputs, config={"max_concurrency": self.batch_size}
            )
            pbar.close()

            for i, (block, entry_text) in enumerate(zip(blocks, cleaned_results)):
                original_parsed = parse_entry(block)
                original_key = original_parsed.get("key", "unknown")

                if entry_text is None:
                    tqdm.write(f"Skipping entry {i + 1} due to processing failure")
                    key_mapping[original_key] = {"status": "failed"}
                    continue

                parsed = parse_entry(entry_text)
                current_key = parsed.get("key", "unknown")

                if key_format and entry_text:
                    try:
                        new_key = generate_citation_key(parsed, key_format)
                        import re

                        entry_text = re.sub(
                            r"(@\w+\s*{\s*)([^,]+)(\s*,)",
                            rf"\1{new_key}\3",
                            entry_text,
                            count=1,
                        )
                        current_key = new_key
                    except Exception as e:
                        tqdm.write(f"Warning: Failed to apply key format: {e}")

                doi_fp = (grab(entry_text, "doi") or "").strip().lower()
                title_fp = (
                    (grab(entry_text, "title") or "")
                    .replace("{", "")
                    .replace("}", "")
                    .strip()
                    .lower()
                )
                year_fp = (grab(entry_text, "year") or "").strip()
                fp = (doi_fp, title_fp, year_fp)

                # Determine if duplicate
                if fp in seen_fps:
                    tqdm.write(f"Skipping duplicate entry {i + 1}")
                    # Map this original key to the primary key that was already kept
                    primary_key = seen_fps[fp]
                    key_mapping[original_key] = {
                        "status": "duplicate",
                        "new_key": primary_key,
                    }
                    continue

                seen_fps[fp] = current_key
                key_mapping[original_key] = {
                    "status": "success",
                    "new_key": current_key,
                }
                corrected_entries.append(entry_text)

            cleaned_content = "\n\n".join(corrected_entries)
            success_count = len(corrected_entries)
            overall_time = time.time() - overall_start_time
            missing_fields = self._collect_missing_required_fields(corrected_entries)

            token_stats = self.token_tracker.get_stats()

            print(
                f"\nSuccessfully processed {success_count}/{total_blocks} entries in {overall_time:.2f}s",
                file=sys.stderr,
            )
            print(
                f"Tokens: {token_stats['total_input_tokens']} input + {token_stats['total_output_tokens']} output ({token_stats['total_requests']} requests)",
                file=sys.stderr,
            )

            return {
                "original": bibtex_content,
                "cleaned": cleaned_content,
                "provider": "openai",
                "success": True,
                "error": None,
                "missing_fields": missing_fields,
                "key_mapping": key_mapping,
                "stats": {
                    "total_entries": total_blocks,
                    "processed_entries": success_count,
                    "skipped_entries": total_blocks - success_count,
                },
            }

        except Exception as e:
            return {
                "original": bibtex_content,
                "cleaned": None,
                "provider": "openai",
                "success": False,
                "error": str(e),
                "missing_fields": [],
            }

    def _process_single_entry_for_enhancement(
        self,
        block: str,
        resolver: OnlineMetadataResolver,
        sources: List[str],
        enhancement_prompt: ChatPromptTemplate,
        retry_count: int = 3,
        entry_index: int = 0,
    ) -> Optional[str]:
        """
        Process a single BibTeX entry for conference enhancement with retry logic.

        Args:
            block: Single BibTeX entry text
            resolver: OnlineMetadataResolver instance
            sources: List of metadata sources
            enhancement_prompt: ChatPromptTemplate for enhancement
            retry_count: Number of retries for failed requests
            entry_index: Index of current entry for logging

        Returns:
            Enhanced entry text or None if processing fails
        """
        for attempt in range(retry_count + 1):
            try:
                pe = parse_entry(block)
                original = {
                    "title": pe.get("title"),
                    "year": pe.get("year"),
                    "doi": pe.get("doi"),
                    "authors": pe.get("authors") or [],
                }

                candidates = resolver.resolve(original, sources, stop_on_match=True)

                human_payload = (
                    "ORIGINAL_ENTRY:\n"
                    f"{block}\n\n"
                    "METADATA_CANDIDATES (JSON):\n"
                    f"{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
                    "Enhance only the venue-related fields (booktitle/journal etc.). "
                    "Return the corrected single BibTeX entry only (no commentary)."
                )

                if attempt > 0:
                    delay_time = self.delay_between_requests * (2**attempt)
                    time.sleep(delay_time)
                elif self.delay_between_requests > 0:
                    time.sleep(self.delay_between_requests)

                chain = enhancement_prompt | self.llm
                result = chain.invoke({"payload": human_payload})
                entry_text = (
                    result.content if hasattr(result, "content") else str(result)
                )

                return entry_text.strip()

            except Exception as e:
                if attempt == retry_count:
                    tqdm.write(f"Enhancement entry {entry_index + 1} failed: {str(e)}")
                    return None

                if "ThrottlingException" in str(e) or "Too many requests" in str(e):
                    if attempt < retry_count:
                        wait_time = self.delay_between_requests * (3 ** (attempt + 1))
                        tqdm.write(f"Rate limit hit, waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue

        return None

    def enhance_conference_info(
        self,
        bibtex_content: str,
        custom_prompt: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Iteratively enhance conference/journal details per entry, using the
        enhancement system prompt + online metadata for each entry.
        """
        if custom_prompt:
            enhancement_system_prompt = custom_prompt
        else:
            enhancement_system_prompt = self._load_conference_enhancement_prompt()

        try:
            resolver = OnlineMetadataResolver(user_agent="BibAI/1.0")
            if sources is None:
                sources = ["dblp", "semantic_scholar", "arxiv"]

            blocks = split_bibtex_entries(bibtex_content)
            enhanced_entries: List[str] = []
            seen_fps = set()
            total_blocks = len(blocks)

            print(
                f"Detected {total_blocks} entries. Starting enhancement...",
                file=sys.stderr,
            )

            per_entry_prompt = ChatPromptTemplate.from_messages(
                [("system", enhancement_system_prompt), ("human", "{payload}")]
            )

            pbar = tqdm(total=total_blocks, desc="Enhancing entries", unit="entry")

            def process_func(item):
                res = self._process_single_entry_for_enhancement(
                    item["block"],
                    resolver,
                    sources,
                    per_entry_prompt,
                    entry_index=item["index"],
                )
                pbar.update(1)
                return res

            inputs = [{"block": block, "index": i} for i, block in enumerate(blocks)]

            enhanced_results = RunnableLambda(process_func).batch(
                inputs, config={"max_concurrency": self.batch_size}
            )
            pbar.close()

            for i, (block, entry_text) in enumerate(zip(blocks, enhanced_results)):
                if entry_text is None:
                    tqdm.write(
                        f"Skipping enhancement for entry {i + 1} due to processing failure"
                    )
                    continue

                doi_fp = (grab(entry_text, "doi") or "").strip().lower()
                title_fp = (
                    (grab(entry_text, "title") or "")
                    .replace("{", "")
                    .replace("}", "")
                    .strip()
                    .lower()
                )
                year_fp = (grab(entry_text, "year") or "").strip()
                fp = (doi_fp, title_fp, year_fp)

                if fp in seen_fps:
                    tqdm.write(f"Skipping duplicate entry {i + 1}")
                    continue

                seen_fps.add(fp)
                enhanced_entries.append(entry_text)

            enhanced_content = "\n\n".join(enhanced_entries)
            success_count = len(enhanced_entries)
            missing_fields = self._collect_missing_required_fields(enhanced_entries)

            print(
                f"\nSuccessfully enhanced {success_count}/{total_blocks} entries",
                file=sys.stderr,
            )

            return {
                "original": bibtex_content,
                "enhanced": enhanced_content,
                "provider": "openai",
                "success": True,
                "error": None,
                "missing_fields": missing_fields,
                "stats": {
                    "total_entries": total_blocks,
                    "processed_entries": success_count,
                    "skipped_entries": total_blocks - success_count,
                },
            }

        except Exception as e:
            return {
                "original": bibtex_content,
                "enhanced": None,
                "provider": "openai",
                "success": False,
                "error": str(e),
                "missing_fields": [],
            }
