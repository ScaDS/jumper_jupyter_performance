"""Print the composed system prompts for inspection.

    python -m jumper_extension.adapters.ai_reviewer.prompts [--strategy S] [--note TEXT]

Shows exactly what ``PromptLibrary`` assembles for each prompt (the system
half of what the LLM receives). For the full input including the human
message, see ``agent/preview.py``.
"""
from __future__ import annotations

import argparse

from jumper_extension.adapters.ai_reviewer.prompts import PromptLibrary
from jumper_extension.adapters.ai_reviewer.strategy import get_strategy, strategy_ids
from jumper_extension.config.loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Print composed AI-reviewer system prompts.")
    parser.add_argument("--strategy", default="faster", choices=strategy_ids())
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    overrides = get_strategy(args.strategy).overrides
    library = PromptLibrary.load()
    variables = {"suggestion_count": load_config().ai.suggestions.count}
    for prompt_id in library.prompt_ids():
        print("=" * 72)
        print(f"# {prompt_id}   (strategy={args.strategy}, note={args.note!r})")
        print("=" * 72)
        print(
            library.render(
                prompt_id,
                overrides=overrides,
                note=args.note,
                variables=variables,
            )
        )
        print()


if __name__ == "__main__":
    main()
