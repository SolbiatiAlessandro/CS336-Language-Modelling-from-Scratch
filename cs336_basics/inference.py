"""Run inference from a trained checkpoint.

Loads a checkpoint (model + optimizer state saved by ``save_checkpoint``),
generates a completion for a prompt with temperature + top-p (nucleus) sampling,
and prints the generated text along with the average entropy and average max
probability of the per-step next-token distributions.

The ``generate`` function below is the user's own decoding implementation, kept
verbatim. It references the module-level globals ``model`` and ``tokenizer`` and
runs on ``mps``; main() sets those globals up before calling it.

Example:
    uv run cs336_basics/inference.py /path/to/checkpoint_step16612_final.pt \\
        --prompt "The history of the Roman Empire" --max-tokens 256 \\
        --temperature 0.7 --top-p 0.9
"""

import argparse
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch._dynamo.eval_frame import OptimizedModule

import cs336_basics.model as model_lib
from cs336_basics.tokenizer import Tokenizer

REPO_DIR = Path(__file__).resolve().parent
CONFIG_PATH = REPO_DIR / "training_configs" / "final_config.json"
TOKENIZER_DIR = REPO_DIR / "tokenizer_artifacts"

# Checkpoints live on a Modal volume rather than on disk (each is ~767 MB).
DEFAULT_VOLUME = "cs336-model-checkpoints"
DEFAULT_CHECKPOINT = (
    "B200_owt_32k_45min_deep6_clip0p5_ckpts_1782191080.0134425_step16612_final.pt"
)

# Architecture fields read from the training config (keeps inference in sync
# with the run that produced the checkpoint).
ARCH_KEYS = ("d_model", "num_heads", "d_ff", "vocab_size", "num_layers")

# Globals the verbatim generate() below closes over. Populated in main().
model = None
tokenizer = None


# ----------------------------------------------------------------------------
# User's decoding implementation — kept character-for-character. Do not edit.
# ----------------------------------------------------------------------------
def generate(
    prompt= "Just testing",
    max_tokens=50,
    debug=False,
    temperature=0.5,
    p_threshold=0.3,
    max_sequence_length=512):

    EOT_ID = tokenizer.encode("<|endoftext|>")[0]
    index = 0
    tokens = tokenizer.encode(prompt)
    entropies = []
    p_maxes = []
    token_id = -1

    while token_id != EOT_ID and index < max_tokens:

        X = torch.tensor(tokens[-max_sequence_length:], device='mps')
        X = X.reshape(1, -1)
        y = model(X)[0, -1]
        m, _ = y.max(axis=-1, keepdims=True)
        p = torch.exp((y - m)/temperature) / torch.exp((y - m)/temperature).sum(axis=-1, keepdims=True)
        vals, idx = p.sort(descending=True)
        mask = (vals.cumsum(dim=-1) - vals) > p_threshold
        vals[mask] = 0
        vals = vals.cpu().detach()
        vals = vals / vals.sum()
        entropy = -(p * torch.log(p + 1e-12)).sum().item()
        entropies.append(entropy)
        p_max = p.max()
        p_maxes.append(p_max.cpu().detach().item())

        pos = torch.multinomial(vals, num_samples=1).item()
        token_id = idx[pos].item()
        tokens.append(token_id)
        decoded_token = tokenizer.decode([token_id])
        index += 1
        if debug: print(f"decoded_token={decoded_token}, entropy={entropy:.3}, pmax={p_max:.2%}")
    return tokenizer.decode(tokens), entropies, p_maxes
# ----------------------------------------------------------------------------
# End user's code.
# ----------------------------------------------------------------------------


def uncompile(module):
    """Recursively unwrap ``torch.compile`` wrappers so the model runs in plain
    eager mode (portable to MPS/CPU) and its state-dict keys lose the inner
    ``_orig_mod.`` segments added by compilation."""
    for name, child in list(module.named_children()):
        if isinstance(child, OptimizedModule):
            orig = child._orig_mod
            setattr(module, name, orig)
            uncompile(orig)
        else:
            uncompile(child)
    return module


def load_checkpoint_obj(checkpoint, volume_name, device):
    """Load a checkpoint dict. If ``checkpoint`` is an existing local file, read
    it from disk; otherwise stream it from the Modal volume into memory (no local
    copy is written)."""
    local = Path(checkpoint)
    if local.is_file():
        print(f"loading checkpoint from local file: {local}")
        return torch.load(local, map_location=device)

    import modal
    print(f"streaming checkpoint from modal volume {volume_name!r}: {checkpoint}")
    volume = modal.Volume.from_name(volume_name)
    buffer = io.BytesIO()
    for chunk in volume.read_file(checkpoint):
        buffer.write(chunk)
    buffer.seek(0)
    return torch.load(buffer, map_location=device)


def load_model(checkpoint, config, device, volume_name):
    arch = {k: config[k] for k in ARCH_KEYS}
    context_length = config["max_context_length"]
    # The Triton flash kernel is CUDA-only; vanilla attention has identical
    # weights and runs everywhere.
    net = model_lib.Transformer(
        **arch,
        context_length=context_length,
        device=device,
        dtype=torch.float32,
        attention_backend="vanilla",
    )
    uncompile(net)
    obj = load_checkpoint_obj(checkpoint, volume_name, device)
    state = {k.replace("_orig_mod.", ""): v for k, v in obj["model"].items()}
    net.load_state_dict(state, strict=True)
    net.eval()
    return net, obj.get("iteration"), context_length


def main():
    global model, tokenizer

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", default=DEFAULT_CHECKPOINT,
                        help="Checkpoint filename on the Modal volume, or a path "
                             "to a local .pt file. Defaults to the final run.")
    parser.add_argument("--volume", default=DEFAULT_VOLUME,
                        help="Modal volume holding the checkpoints.")
    parser.add_argument("--prompt", default="Just testing",
                        help="Prompt to complete.")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Softmax temperature.")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold in (0, 1] (generate's p_threshold).")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Context window fed to the model each step.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH,
                        help="Training config JSON providing the architecture.")
    parser.add_argument("--debug", action="store_true",
                        help="Print each generated token with its entropy/pmax.")
    args = parser.parse_args()

    # generate() is hardcoded to device='mps', so the model must live there too.
    device = "mps"
    with open(args.config) as f:
        config = json.load(f)

    tokenizer = Tokenizer.from_files(
        TOKENIZER_DIR / "vocab.json",
        TOKENIZER_DIR / "merges.json",
        special_tokens=["<|endoftext|>"],
    )
    model, iteration, context_length = load_model(
        args.checkpoint, config, device, args.volume)

    print(f"device={device}  params={model.num_params / 1e6:.1f}M  "
          f"iteration={iteration}  ctx={context_length}")
    print(f"temperature={args.temperature}  p_threshold={args.top_p}  "
          f"max_tokens={args.max_tokens}")
    if args.debug:
        print("--- per-token ---")

    text, entropies, p_maxes = generate(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        debug=args.debug,
        temperature=args.temperature,
        p_threshold=args.top_p,
        max_sequence_length=args.max_seq_len,
    )

    avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    avg_p_max = sum(p_maxes) / len(p_maxes) if p_maxes else 0.0

    print("\n=== generated ===")
    print(text)
    print("\n=== stats ===")
    print(f"average entropy : {avg_entropy:.3f} nats")
    print(f"average p_max   : {avg_p_max:.2%}")


if __name__ == "__main__":
    main()
