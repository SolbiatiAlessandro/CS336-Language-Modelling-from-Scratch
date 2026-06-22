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
