## BibTeX Cleaning Agent

[![PyPI version](https://badge.fury.io/py/bibtex-cleaner.svg)](https://badge.fury.io/py/bibtex-cleaner)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)

![BibTeX Cleaner overview](https://raw.githubusercontent.com/tohoku-nlp/bibtex-cleaning-agent/main/assets/cover.png)

BibTeX Cleaner is an LLM-powered BibTeX cleaning agent and command-line tool. It connects to DBLP, Semantic Scholar, and arXiv to fix formatting issues, fill in missing information, standardize fields, remove duplicates, and automatically search for and update a paper’s official publication details. 

## Features

- **Cleaning & Normalization**: Standardizes BibTeX entries, fixes formatting, and ensures consistent fields.
- **Metadata Enhancement**: Fetches accurate metadata from DBLP, Semantic Scholar, and ArXiv.
- **Conference Info**: Updates venue information to official conference titles (e.g., "The 37th International Conference on Machine Learning (ICML)").
- **Custom Citation Keys**: Generates citation keys based on customizable patterns (e.g., `{author}{year}{shorttitle}`).
- **Deduplication**: Detects and removes duplicate entries.
- **Key Mapping JSON**: Writes a JSON sidecar file that maps original BibTeX keys to their cleaned/deduplicated keys.
- **Missing Field Diagnostics**: Prints a summary of entries that are missing required core fields (e.g., `pages` for `@inproceedings`) so you can fix them manually.

## Installation

You can install the package via pip or uv:

```bash
pip install bibtex-cleaner
# or
uv pip install bibtex-cleaner
```

## API Key and Environment Variables

BibTeX Cleaner uses the `OPENAI_API_KEY` environment variable. There are two common ways to set it:

- **Using a `.env` file** (recommended for per-project setup):

  Create a `.env` file in your LaTeX project root (where your `.bib` file lives):

  ```bash
  echo 'OPENAI_API_KEY=your_api_key_here' > .env
  ```

  The CLI automatically loads `.env` when you run `bibtex-cleaner`.

- **Using a shell environment variable**:

  ```bash
  export OPENAI_API_KEY=your_api_key_here
  # or one-shot:
  OPENAI_API_KEY=your_api_key_here bibtex-cleaner input.bib output.bib
  ```

## Usage

### Typical Workflow

- **From your LaTeX project root** (where `myrefs.bib` is located):

  ```bash
  bibtex-cleaner myrefs.bib cleaned_myrefs.bib
  ```

- **With default output from config** (see below), you can omit the output path:

  ```bash
  bibtex-cleaner myrefs.bib
  ```

### Basic Cleaning

Clean a BibTeX file using default settings:

```bash
bibtex-cleaner input.bib output.bib
```

### Configuration

The tool uses a `bibtex_cleaner_config.yaml` file for configuration. By default, it looks for this file in the current working directory, but you can also specify a custom path with the `--config` flag.

- **Create a config file in your LaTeX project** (recommended):

  - In the same directory as your `.bib` file, create `bibtex_cleaner_config.yaml`.
  - Either:
    - Copy the example below into that file, or
    - Copy the template from the GitHub repository (`bibtex_cleaner_config.yaml` in the project root).

- **Run the cleaner with your config**:

  ```bash
  # Uses bibtex_cleaner_config.yaml in the current directory
  bibtex-cleaner myrefs.bib cleaned_myrefs.bib

  # Or specify a custom config path explicitly
  bibtex-cleaner myrefs.bib cleaned_myrefs.bib --config path/to/bibtex_cleaner_config.yaml
  ```

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
bibtex-cleaner input.bib [output.bib] [--config path/to/config.yaml]
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

Example: `key_format: "{author}-{year}-{veryshorttitle}"` -> `Vaswani-2017-Attention`


## Contributor of This Project
This idea was born from discussions between Kai Kamijo, [Mengyu Ye](https://muyo8692.com/), [Ryosuke Takahashi](https://r-takahashi.webflow.io/), and Taïga Gonçalves (in alphabetical order).

[Ryosuke Takahashi](https://r-takahashi.webflow.io/) implemented the core LangChain-based cleaning pipeline, and Taïga Gonçalves developed the initial conference-metadata resolver prototype. [Mengyu Ye](https://muyo8692.com/) refactored and streamlined these components, added parallel cleaning support, integrated them into a unified Python package, and finalized the current release.
