## BibTeX Cleaner

![BibTeX Cleaner overview](https://raw.githubusercontent.com/tohoku-nlp/bibtex-cleaning-agent/main/assets/cover.png)

BibTeX Cleaner is an LLM-powered BibTeX cleaning agent and command-line tool that integrates with DBLP, Semantic Scholar, and arXiv to automatically clean, normalize, enrich, and deduplicate BibTeX entries, while optionally standardizing citation keys and conference/journal information.

## Features

- **Cleaning & Normalization**: Standardizes BibTeX entries, fixes formatting, and ensures consistent fields.
- **Metadata Enhancement**: Fetches accurate metadata from DBLP, Semantic Scholar, and ArXiv.
- **Conference Info**: Updates venue information to official conference titles (e.g., "The 37th International Conference on Machine Learning (ICML)").
- **Custom Citation Keys**: Generates citation keys based on customizable patterns (e.g., `{author}{year}{shorttitle}`).
- **Deduplication**: Detects and removes duplicate entries.
- **Key Mapping JSON**: Writes a JSON sidecar file that maps original BibTeX keys to their cleaned/deduplicated keys.
- **Missing Field Diagnostics**: Prints a summary of entries that are missing required core fields (e.g., `pages` for `@inproceedings`) so you can fix them manually.

## Installation

You can install the package via pip (or uv):

```bash
pip install bibtex-cleaner
# or
uv pip install bibtex-cleaner
```

## Environment Variables

You need to create a `.env` file in your project root with your API key first:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Cleaning

Clean a BibTeX file using default settings:

```bash
bib-cleaner input.bib output.bib
```

### Configuration

The tool uses a `bibtex_cleaner_config.yaml` file for configuration. You can specify a custom config path with the `--config` flag.

Example `bibtex_cleaner_config.yaml`:

```yaml
# BibTeX Cleaner Configuration

# LLM Settings
model: "gpt-4o-mini"

# Processing Settings
batch_size: 5
delay: 1.0
retry_count: 3

# Feature Flags
enhance_conferences: false  # Set to true for deep conference metadata enhancement
key_format: "{author}{year}{shorttitle}"

# Output Settings
json: false
output: "cleaned.bib"
```

### CLI Options

```bash
bib-cleaner input.bib [output.bib] [--config path/to/config.yaml]
```

### Key Mapping JSON Output

When you specify an `output.bib` file (either via CLI or config), the cleaner also generates a JSON file with the same stem:

- **Output**: For `output.bib`, a `output.json` file is created alongside it.
- **Content**: A mapping from original keys to their final keys and status, for example:
  - original key that was successfully cleaned and kept -> `{ "status": "success", "new_key": "<final_key>" }`
  - original key that was detected as a duplicate and removed -> `{ "status": "duplicate", "new_key": "<canonical_key>" }`
  - original key that failed cleaning -> `{ "status": "failed" }`

This makes it easy to update citations in your LaTeX project after renaming and deduplication.

### Diagnostics and Known Limitations

- **Missing field warnings**: After cleaning, the CLI prints a short report to stderr listing entries that are missing important fields (for example, `pages` for `@inproceedings`, or `journal`/`year` for `@article`). These warnings do not stop the run; they are there to help you manually patch edge cases.
- **Upstream metadata gaps (e.g., pages)**: Sometimes DBLP / Semantic Scholar / arXiv do not provide complete metadata (most commonly page ranges for conference papers). In those cases the tool will warn about missing `pages`, but it will not fabricate values. 

## Citation Key Formatting

You can customize the citation key format in the configuration file using placeholders:

- `{author}`: First author's surname
- `{year}`: Publication year
- `{title}`: Full sanitized title
- `{shorttitle}`: First 3 meaningful words of title (default)
- `{veryshorttitle}`: First 1 meaningful word of title
- `{mediumtitle}`: First 5 meaningful words of title
- `{venue}`: Venue/journal name
- `{doi}`: DOI

Example: `key_format: "{author}-{year}-{veryshorttitle}"` -> `Smith-2023-Attention`


## Contributor of This Project
This idea was born from discussions between Kai Kamijo, [Mengyu Ye](https://muyo8692.com/), [Ryosuke Takahashi](https://r-takahashi.webflow.io/), and Taïga Gonçalves (in alphabetical order).

[Ryosuke Takahashi](https://r-takahashi.webflow.io/) implemented the core LangChain-based cleaning pipeline, and Taïga Gonçalves developed the initial conference-metadata resolver prototype. [Mengyu Ye](https://muyo8692.com/) refactored and streamlined these components, added parallel cleaning support, integrated them into a unified Python package, and finalized the current release.