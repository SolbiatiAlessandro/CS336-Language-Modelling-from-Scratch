# CS336 Assignment 1 Leaderboard Submission

Minimal reproducible snapshot for Alessandro Solbiati's OpenWebText leaderboard run.

Final reported run:

- W&B run: https://wandb.ai/lessandro/cs336/runs/zzd4j6h7
- W&B report: https://wandb.ai/lessandro/cs336/reports/Loss-Validation-26-06-21-22-23-46---VmlldzoxNzI5ODQwMw
- Final validation loss: 3.3998279571533203
- Wallclock: 44.999 minutes on one B200

Run:

```bash
uv run main.py
```

The script defaults to the final hyperparameters and expects the tokenized OpenWebText memmaps at:

```text
/root/owt_artifacts/tokens/owt_prefix_100000_vocab_32000/train.uint16
/root/owt_artifacts/tokens/owt_prefix_100000_vocab_32000/valid.uint16
```

Override paths if needed:

```bash
TRAIN_DATA=/path/to/train.uint16 VALID_DATA=/path/to/valid.uint16 uv run main.py
```

This repo also includes the original Modal training entrypoint in `cs336_basics/train_model.py`.

## Inference

Generate text from a trained checkpoint with `cs336_basics/inference.py`. Checkpoints
are not stored in the repo — they are streamed on demand from the Modal volume
`cs336-model-checkpoints` straight into memory (each is ~767 MB), so nothing heavy
lands on disk. This needs Modal auth (`uv run modal token new` once).

Run with the default final checkpoint:

```bash
uv run cs336_basics/inference.py --prompt "The history of the Roman Empire"
```

Pick a different checkpoint from the volume and tune sampling:

```bash
uv run cs336_basics/inference.py \
  B200_owt_32k_45min_deep6_clip0p5_ckpts_1782191080.0134425_step1480_frac10.pt \
  --prompt "Once upon a time" \
  --max-tokens 256 --temperature 0.7 --top-p 0.9
```

List the available checkpoints on the volume:

```bash
uv run modal volume ls cs336-model-checkpoints
```

The script prints the generated text plus the average entropy and average max
probability (`p_max`) across the generated tokens. Useful flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `checkpoint` (positional) | final run | Filename on the volume, or a local `.pt` path |
| `--volume` | `cs336-model-checkpoints` | Modal volume holding the checkpoints |
| `--prompt` | `"Just testing"` | Prompt to complete |
| `--max-tokens` | `256` | Max tokens to generate |
| `--temperature` | `0.7` | Softmax temperature |
| `--top-p` | `0.9` | Nucleus (top-p) threshold |
| `--max-seq-len` | `512` | Context window fed to the model each step |
| `--debug` | off | Print each token with its entropy / p_max |

Generation runs on Apple Silicon (`mps`). If you pass a path to a local file
instead of a volume filename, it loads from disk directly.

## Web UI

`serve.py` wraps the same `inference.py` code in a small local web app — a
ChatGPT-style chat on the left, a live model/training panel on the right, and a
per-token `p_max` / `entropy` plot for every generation.

```bash
uv run serve.py            # → http://127.0.0.1:8000
uv run serve.py --port 8080
```

It's plumbing only: `serve.py` streams the checkpoint from the Modal volume into
memory via `inference.load_model`, then drives the verbatim `inference.generate`
decoding loop — no model code is duplicated. On startup the page renders
immediately and shows a `streaming → loading → ready` status while the checkpoint
loads on `mps` in the background.

The right-hand panel shows:

- **Model** — parameter count, architecture (layers, `d_model`, heads, `d_ff`,
  vocab, context), device, and the checkpoint's train step.
- **Training · W&B** — best and final train/val losses and the loss curve, pulled
  live from the leaderboard run [`lessandro/cs336/zzd4j6h7`](https://wandb.ai/lessandro/cs336/runs/zzd4j6h7).
- **Last generation** — average `p_max` / `entropy` plus a per-token chart of both
  (watch confidence rise as the model commits to a continuation).

The checkpoint dropdown in the header switches between the final, 10%, and
untrained snapshots of the run (each streamed on demand) — a quick way to *see*
the entropy collapse from ~10.3 nats (untrained, uniform over 32k vocab) to ~2
nats as the model learns.

Credentials come from your environment — `~/.modal.toml` for Modal and `~/.netrc`
for W&B. No keys are stored in the repo.
