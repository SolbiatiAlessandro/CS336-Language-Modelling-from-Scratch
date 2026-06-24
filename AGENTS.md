# AGENTS.md — working guide for AI agents on this repo

This repo is a reproducible snapshot of Alessandro's CS336 language model: a 67M-param
Transformer pretrained from scratch on OpenWebText (final val loss 3.42), plus the
tooling to train, run inference, and now post-train (SFT) it.

## Knowledge base — READ THIS FIRST

`wiki/research-logs.md` is the project's knowledge base: a running, dated log of
experiments, findings, and **decisions** (newest on top). Before proposing or starting
work, read it — it explains *why* things are the way they are (e.g. why we use a
plain-text chat template instead of new special tokens, why the model is considered
trained ~optimally for its size, the SFT plan, cost references).

When you finish a meaningful piece of work or reach a decision, **append a new dated
entry** to `wiki/research-logs.md`. Keep plots/assets in `wiki/assets/`.

## HARD RULE: nothing big on the local machine

The working machine is a laptop and must stay clean. **Do not write, download, or
copy any data artifact larger than 100 MB to the local filesystem** — not to the repo,
not to `~/Downloads`, not to `/tmp`. This includes datasets, tokenized corpora, and
model checkpoints (each checkpoint is ~767 MB).

**Everything big goes through Modal:**
- **Datasets / tokenization / training** run as Modal jobs; outputs are written to
  **Modal volumes**, never downloaded locally.
- **Checkpoints** live on the `cs336-model-checkpoints` volume. `inference.py` *streams*
  them into memory (`BytesIO` → `torch.load`); it never saves a local `.pt`.
- If you need to inspect big data, do it inside a Modal function and print/return only
  small summaries.
- Small text artifacts are fine locally: source code, configs, the tokenizer
  vocab/merges (~1.5 MB), JSON manifests, plots.

If a task seems to require a large local file, stop and find the Modal-based way instead.

## Modal layout

- **App(s):** training is `cs336-basics-training` (`cs336_basics/train_model.py`);
  SFT tokenization is `cs336-sft-tokenize` (`cs336_basics/tokenize_sft.py`).
- **Volumes:**
  - `cs336-owt-artifacts` — raw + tokenized OpenWebText (pretraining data), ~28 GiB
  - `cs336-model-checkpoints` — saved checkpoints, ~19 GiB
  - `cs336-sft-data` — tokenized SFT datasets (created by the SFT tokenize job)
- **GPU:** training uses B200 ($6.25/hr; a 45-min run ≈ $4.69). SFT of a 67M model is
  far lighter — an A100-40GB or smaller is plenty.
- **Storage cost:** Modal volumes are $0.09/GiB/mo with 1 TiB/mo free; current usage is
  well under the free tier.
- **Auth:** Modal via `~/.modal.toml` (`uv run modal token new` once); W&B via `~/.netrc`.

## Common commands

```bash
# Inference (streams checkpoint from Modal; default = final checkpoint)
uv run cs336_basics/inference.py --prompt "..." --max-tokens 256 --temperature 0.7 --top-p 0.9

# List checkpoints / volumes
uv run modal volume ls cs336-model-checkpoints
uv run modal volume ls cs336-sft-data

# Tokenize the Alpaca SFT dataset on Modal (writes tokens to cs336-sft-data)
uv run modal run cs336_basics/tokenize_sft.py

# Pretraining (B200)
uv run main.py
```

## Model / code facts worth knowing

- Architecture: d_model 768, 12 heads, d_ff 2048, 6 layers, vocab 32000, context 512.
  **Tied embeddings** (`head` shares `embedding.embeddings`). Defined in
  `cs336_basics/model.py`; loaded for inference with the `vanilla` attention backend
  (the Triton flash kernel is CUDA-only).
- Tokenizer: `cs336_basics/tokenizer.py` + `cs336_basics/tokenizer_artifacts/{vocab,merges}.json`.
  Only special token is `<|endoftext|>` (id 256).
- SFT approach (see research log): plain-text Alpaca template, reuse `<|endoftext|>` as
  the stop token, **loss-masked on the response span only** (prompt tokens are
  conditioning, not targets). No new special tokens for v1.

## CS336 academic-integrity note

The pretraining code here is Alessandro's own completed work. This is his personal
reproduction/experimentation repo (not a live graded submission), so building inference
and post-training tooling on top of it is in scope. If a request ever maps onto an
unsubmitted graded CS336 deliverable, prefer explaining/reviewing over writing it.
