"""Modal launcher for the GSM8K prompting baselines.

This is the *orchestration* layer only: it dispatches your evaluation script to
Modal GPU containers (one container per prompt style) using the helpers in
``cs336_alignment/modal_utils.py``. It contains no evaluation logic itself.

What you still have to write (the actual ``prompting_baselines`` deliverable):
    scripts/eval_gsm8k.py  -- loads OLMo-2-0425-1B with vLLM, builds the prompt
                              from the chosen template, generates over GSM8K,
                              grades with the reward fns in drgrpo_grader.py,
                              and buckets the results.

This launcher calls that script like:
    python scripts/eval_gsm8k.py --prompt <style> [your other args...]

Prereqs (one-time):
    1. Set SUNET_ID in cs336_alignment/modal_utils.py
    2. uv run modal token new
    3. Create the Modal secret used by modal_utils:  modal secret create wandb WANDB_API_KEY=...

Run (from inside assignment5-alignment/):
    uv run modal run scripts/eval_modal.py
    uv run modal run scripts/eval_modal.py --prompts r1_zero          # single style
    uv run modal run scripts/eval_modal.py --max-examples 8           # quick smoke test

The OLMo weights download *inside the container* (vLLM pulls from HF on the GPU
box), so nothing heavy lands on your Mac. Only code + data are uploaded by the
image definition in modal_utils.py.
"""

from __future__ import annotations

import argparse
import sys
import json
import statistics
from collections import Counter
from pathlib import Path

from cs336_alignment.modal_utils import app, submit_commands

# The three prompt styles the prompting_baselines problem asks you to evaluate.
# Templates live in cs336_alignment/prompts/.
PROMPT_STYLES = ["question_only", "r1_zero", "r1_zero_three_shot"]

# Path (inside the Modal container, cwd = /root) to the eval script you write.
EVAL_SCRIPT = "scripts/eval_gsm8k.py"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch GSM8K prompting baselines on Modal.")
    parser.add_argument(
        "--prompts",
        default=",".join(PROMPT_STYLES),
        help="Comma-separated prompt styles to run (one Modal container each). "
        f"Default: all of {PROMPT_STYLES}.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of GSM8K examples (forwarded to your eval script for quick runs).",
    )
    return parser


def build_run_commands(args: argparse.Namespace) -> list[list[str]]:
    """Map each requested prompt style to a command run inside a Modal container."""
    styles = [s.strip() for s in args.prompts.split(",") if s.strip()]
    commands: list[list[str]] = []
    for style in styles:
        command = ["python", "-u", EVAL_SCRIPT, "--prompt", style]
        if args.max_examples is not None:
            command += ["--max-examples", str(args.max_examples)]
        commands.append(command)
    return commands


@app.local_entrypoint()
def modal_main(*argv: str) -> None:
    args = make_parser().parse_args(list(argv))
    commands = build_run_commands(args)
    submit_commands(commands)



    
