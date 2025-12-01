import json
import os
import sys
import fire
from .cleaner import LangChainBibTeXCleaner
from .utils import load_config

from dotenv import load_dotenv


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()


def run_cleaner(
    input: str = None,
    output: str = None,
    config: str = "bibtex_cleaner_config.yaml",
):
    """
    BibTeX Cleaner CLI.

    Args:
        input: Input BibTeX file path (or '-' for stdin).
        output: Output file path (or '-' for stdout).
        config: Path to YAML configuration file.
    """
    load_environment()

    conf = load_config(config)

    def get_conf(name, default=None):
        return conf.get(name, default)

    if input and input != "-":
        try:
            with open(input, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: File '{input}' not found", file=sys.stderr)
            sys.exit(1)
        except UnicodeDecodeError:
            print(f"Error: Could not read file '{input}' as UTF-8", file=sys.stderr)
            sys.exit(1)
    else:
        if sys.stdin.isatty():
            print(
                "Error: No input provided. Usage: bibtex-cleaner <input_file> or cat input.bib | bibtex-cleaner",
                file=sys.stderr,
            )
            sys.exit(1)
        content = sys.stdin.read()

    kwargs = {}

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key

    model_val = get_conf("model")
    if model_val:
        kwargs["model"] = model_val

    try:
        batch_size = int(get_conf("batch_size", 1))
        delay = float(get_conf("delay", 1.0))
        custom_prompt = get_conf("custom_prompt")

        cleaner = LangChainBibTeXCleaner(
            custom_prompt=custom_prompt,
            batch_size=batch_size,
            delay_between_requests=delay,
            **kwargs,
        )

        enhance_conferences = get_conf("enhance_conferences", False)
        if enhance_conferences:
            result = cleaner.enhance_conference_info(
                content, custom_prompt=custom_prompt
            )
            output_key = "enhanced"
        else:
            key_format = get_conf("key_format")
            result = cleaner.clean_bibtex(content, key_format=key_format)
            output_key = "cleaned"

        json_output = get_conf("json", False)
        if json_output:
            output_text = json.dumps(result, ensure_ascii=False, indent=2)
        else:
            if result["success"]:
                output_text = result[output_key]
            else:
                print(f"Error: {result['error']}", file=sys.stderr)
                sys.exit(1)

        output_file = output or get_conf("output")
        if output_file and output_file != "-":
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(output_text)

                # Also generate JSON key mapping file
                if result.get("key_mapping"):
                    json_path = os.path.splitext(output_file)[0] + ".json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(
                            result["key_mapping"], f, ensure_ascii=False, indent=2
                        )

            except IOError as e:
                print(f"Error writing to file '{output_file}': {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print(output_text, end="")

        missing_fields = (
            result.get("missing_fields") if isinstance(result, dict) else None
        )
        if missing_fields:
            print(
                "\nMissing required BibTeX fields detected:",
                file=sys.stderr,
            )
            for entry in missing_fields:
                fields = ", ".join(entry.get("fields", []))
                key = entry.get("key", "unknown")
                entry_type = entry.get("entry_type", "entry")
                print(
                    f"- {key} ({entry_type}): {fields}",
                    file=sys.stderr,
                )

    except Exception as e:
        print(f"Error initializing cleaner: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    fire.Fire(run_cleaner)


if __name__ == "__main__":
    main()
