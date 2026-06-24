"""Tokenize an instruction-following SFT dataset (default: tatsu-lab/alpaca) with
*our* BPE tokenizer, on Modal, and write the tokens to the `cs336-sft-data` volume.

Output format (per dataset, under <volume>/<out_subdir>/):
- input_ids.uint16   flat concatenation of every example's token ids
- offsets.npy        int64 [N+1]; example i = input_ids[offsets[i]:offsets[i+1]]
- prompt_lens.npy    int32 [N]; tokens of the prompt (everything before the response)
- manifest.json      counts, template, tokenizer + special-token info

SFT loss masking (done at training time, not here): for example i, the loss mask is
1 for positions >= prompt_lens[i] (the response + trailing <|endoftext|>) and 0 for the
prompt. We tokenize prompt and response separately and concatenate, so prompt_lens is
exact regardless of BPE merges across the boundary.

Run:
    uv run modal run cs336_basics/tokenize_sft.py
    uv run modal run cs336_basics/tokenize_sft.py --dataset-name tatsu-lab/alpaca
"""

import json
from pathlib import Path

import modal

app = modal.App("cs336-sft-tokenize")
LOCAL_DIR = Path(__file__).parent
REMOTE_DIR = Path("/root/cs336_basics")

sft_volume = modal.Volume.from_name("cs336-sft-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "regex", "datasets", "huggingface-hub")
    .add_local_file(LOCAL_DIR / "__init__.py", str(REMOTE_DIR / "__init__.py"), copy=True)
    .add_local_file(LOCAL_DIR / "tokenizer.py", str(REMOTE_DIR / "tokenizer.py"), copy=True)
    .add_local_dir(
        LOCAL_DIR / "tokenizer_artifacts",
        str(REMOTE_DIR / "tokenizer_artifacts"),
        copy=True,
    )
)

# Standard Alpaca prompt templates (plain text; no new special tokens). The prompt ends
# at "### Response:\n" so the model learns to generate the response after it.
PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
)

# Per-worker tokenizer (built once in each process by the pool initializer).
_TOK = None
_EOT = None


def _init_worker(vocab_path, merges_path):
    global _TOK, _EOT
    import sys
    sys.path.insert(0, "/root")
    from cs336_basics.tokenizer import Tokenizer

    _TOK = Tokenizer.from_files(
        Path(vocab_path), Path(merges_path), special_tokens=["<|endoftext|>"]
    )
    _EOT = _TOK.encode("<|endoftext|>")[0]


def _encode_example(pair):
    """pair = (prompt_text, response_text) -> (prompt_ids, response_ids_with_eot)."""
    prompt_text, response_text = pair
    prompt_ids = _TOK.encode(prompt_text)
    response_ids = _TOK.encode(response_text) + [_EOT]
    return prompt_ids, response_ids


def _format(example):
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output_text = (example.get("output") or "").strip()
    if input_text:
        prompt = PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text)
    else:
        prompt = PROMPT_NO_INPUT.format(instruction=instruction)
    return prompt, output_text


@app.function(image=image, volumes={"/data": sft_volume}, cpu=8.0, timeout=60 * 60)
def tokenize(dataset_name: str = "tatsu-lab/alpaca",
             split: str = "train",
             out_subdir: str = "alpaca_vocab32000"):
    import sys
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    from datasets import load_dataset

    sys.path.insert(0, "/root")

    vocab_path = REMOTE_DIR / "tokenizer_artifacts" / "vocab.json"
    merges_path = REMOTE_DIR / "tokenizer_artifacts" / "merges.json"

    print(f"loading dataset {dataset_name} [{split}] ...")
    ds = load_dataset(dataset_name, split=split)
    print(f"  {len(ds)} examples; fields = {ds.column_names}")

    pairs = [_format(ex) for ex in ds]

    print("tokenizing with our BPE (8 workers) ...")
    with ProcessPoolExecutor(
        max_workers=8, initializer=_init_worker, initargs=(str(vocab_path), str(merges_path))
    ) as pool:
        encoded = list(pool.map(_encode_example, pairs, chunksize=256))

    # Flatten into the on-disk layout.
    input_ids = []
    offsets = [0]
    prompt_lens = []
    for prompt_ids, response_ids in encoded:
        input_ids.extend(prompt_ids)
        input_ids.extend(response_ids)
        offsets.append(len(input_ids))
        prompt_lens.append(len(prompt_ids))

    input_ids = np.asarray(input_ids, dtype=np.uint16)
    offsets = np.asarray(offsets, dtype=np.int64)
    prompt_lens = np.asarray(prompt_lens, dtype=np.int32)

    seq_lens = np.diff(offsets)
    response_lens = seq_lens - prompt_lens

    out_dir = Path("/data") / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    input_ids.tofile(out_dir / "input_ids.uint16")
    np.save(out_dir / "offsets.npy", offsets)
    np.save(out_dir / "prompt_lens.npy", prompt_lens)

    manifest = {
        "dataset": dataset_name,
        "split": split,
        "tokenizer": "cs336 BPE vocab_size=32000 (owt_prefix_100000_vocab_32000)",
        "special_tokens": {"<|endoftext|>": _init_eot(vocab_path, merges_path)},
        "template": "alpaca (plain text, response after '### Response:\\n', <|endoftext|> appended)",
        "num_examples": int(len(prompt_lens)),
        "total_tokens": int(input_ids.size),
        "total_response_tokens": int(response_lens.sum()),
        "seq_len": {
            "mean": float(seq_lens.mean()),
            "p50": int(np.percentile(seq_lens, 50)),
            "p95": int(np.percentile(seq_lens, 95)),
            "max": int(seq_lens.max()),
        },
        "files": {
            "input_ids": "input_ids.uint16 (np.uint16, flat)",
            "offsets": "offsets.npy (int64, [N+1])",
            "prompt_lens": "prompt_lens.npy (int32, [N])",
        },
        "loss_mask_rule": "mask=1 for positions >= prompt_lens[i] (response + <|endoftext|>)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    sft_volume.commit()

    mb = input_ids.nbytes / 1024 ** 2
    print(f"wrote {out_dir} :  {len(prompt_lens)} examples, "
          f"{input_ids.size:,} tokens ({mb:.1f} MiB), "
          f"mean seq {seq_lens.mean():.0f} tok (p95 {np.percentile(seq_lens,95):.0f})")
    print(f"response tokens (the part we train on): {int(response_lens.sum()):,}")
    return manifest


def _init_eot(vocab_path, merges_path):
    """Resolve the <|endoftext|> id in the driver process (for the manifest)."""
    import sys
    sys.path.insert(0, "/root")
    from cs336_basics.tokenizer import Tokenizer

    tok = Tokenizer.from_files(Path(vocab_path), Path(merges_path),
                               special_tokens=["<|endoftext|>"])
    return int(tok.encode("<|endoftext|>")[0])


@app.local_entrypoint()
def main(dataset_name: str = "tatsu-lab/alpaca",
         split: str = "train",
         out_subdir: str = "alpaca_vocab32000"):
    manifest = tokenize.remote(dataset_name=dataset_name, split=split, out_subdir=out_subdir)
    print(json.dumps(manifest, indent=2))
