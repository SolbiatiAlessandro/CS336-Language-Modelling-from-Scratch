import argparse
import sys
import json
import statistics
from collections import Counter
from pathlib import Path
import wandb
from random import random

from cs336_alignment.modal_utils import app, submit_commands
from cs336_alignment.vllm_utils import VLLMServer, generate_completions
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True,
                        choices=["question_only", "r1_zero", "r1_zero_three_shot"])
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()        # reads sys.argv automatically
    DATA_DIR = Path("data/gsm8k")

    if args.prompt == "question_only":
        prompt_style="question_only"
        PROMPT_DIR = Path("cs336_alignment/prompts/question_only.prompt")
    if args.prompt == "r1_zero":
        prompt_style="r1_zero"
        PROMPT_DIR = Path("cs336_alignment/prompts/r1_zero.prompt")
    if args.prompt == "r1_zero_three_shot":
        prompt_style="r1_zero_three_shot"
        PROMPT_DIR = Path("cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt")
    prompt = PROMPT_DIR.read_text()
    def load_jsonl(path):
        """Each line is an independent JSON object: {'question': str, 'answer': str}."""
        with open(path) as f:
            return [json.loads(line) for line in f]

    def gold_answer(record):
        return record["answer"].split("####")[-1].strip()

    test = load_jsonl(DATA_DIR / "test.jsonl")
    wandb.init(
        entity="lessandro",            # your W&B entity (same as assignment 1)
        project="cs336-a5-baselines",
        name=prompt_style,             # e.g. "question_only" — so the 3 runs are distinguishable
        config={                       # logs your hyperparams for reproducibility
            "model": "allenai/OLMo-2-0425-1B",
            "prompt_style": prompt_style,
            "temperature": 1.0,
            "max_tokens": 512,
            "n_examples": len(test),
        },
    )
    train = load_jsonl(DATA_DIR / "train.jsonl")
    print(f"test:  {len(test):>5} examples")
    print(f"train: {len(train):>5} examples")
    server = VLLMServer(model_id="allenai/OLMo-2-0425-1B", gpu=0)
    server.start()

    sampling_params = {
            "temperature": 1,
            "top_p": 1,
            "max_tokens": 512,
            "n": 1,
            "seed" : 0
            }
    sampling_params['stop'] = ["</answer>"]
    sampling_params['include_stop_str_in_output'] = True

    prompts = [prompt.format(question=row['question']) for row in test]
    completions = server.generate_completions(prompts, sampling_params)

    cat1table = wandb.Table(columns=["prompt", "response", "reward"])
    cat2table = wandb.Table(columns=["prompt", "response", "reward"])
    cat3table = wandb.Table(columns=["prompt", "response", "reward"])
    cat1, cat2, cat3 = 0,0,0
    for i, row in enumerate(test):
        question = row['question']
        response = completions[i].text
        ground_truth = row["answer"].split("####")[-1].strip()
        if args.prompt == "question_only":
            grading = question_only_reward_fn(completions[i].text, ground_truth)
        if "r1" in args.prompt:
            grading = r1_zero_reward_fn(completions[i].text, ground_truth)
        is_cat1 = grading['format_reward'] == 1 and grading['answer_reward'] == 1
        cat1 += is_cat1
        if random() < 0.05 and is_cat1:
            cat1table.add_data(question, response, is_cat1)
        is_cat2 = grading['format_reward'] == 1 and grading['answer_reward'] == 0
        cat2 += is_cat2
        if random() < 0.05 and is_cat2:
            cat2table.add_data(question, response, is_cat2)
        is_cat3 = grading['format_reward'] == 0
        cat3 += is_cat3
        if random() < 0.05 and is_cat3:
            cat3table.add_data(question, response, is_cat3)
        wandb.log({
            "correct":            cat1,                 # format=1, answer=1
            "incorrect":    cat2,                 # format=1, answer=0
            "bad_format":         cat3,                 # format=0
            "accuracy":           cat1 / len(test),
        })
    wandb.log({"correct": cat1table})
    wandb.log({"incorrect": cat2table})
    wandb.log({"misformat": cat3table})
    wandb.finish()
