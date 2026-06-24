import numpy as np
from pathlib import Path
import torch
import time
import json
import argparse
import modal
import wandb

import cs336_basics.model as model
import cs336_basics.data as data


class WandbWriter:
    def add_scalars(self, group, scalars, step):
        wandb.log(
            {
                f"{group}/{name}": value.detach().item() if isinstance(value, torch.Tensor) else value
                for name, value in scalars.items()
            },
            step=step,
        )


def get_checkpoint_path(
        checkpoint_dir,
        run_name,
        step):
    return checkpoint_dir / f"{run_name}_step{step}.pt"


def get_gpu_memory_metrics():
    gib = 1024 ** 3
    return {
        "Allocated GiB": torch.cuda.memory_allocated() / gib,
        "Reserved GiB": torch.cuda.memory_reserved() / gib,
        "Peak Allocated GiB": torch.cuda.max_memory_allocated() / gib,
        "Peak Reserved GiB": torch.cuda.max_memory_reserved() / gib,
    }


def get_torch_dtype(dtype):
    if dtype is None:
        return None
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    raise ValueError(f"unknown dtype: {dtype}")


app = modal.App("cs336-basics-training")
LOCAL_DIR = Path(__file__).parent
SYSTEMS_DIR = LOCAL_DIR.parent / "cs336_systems"
REMOTE_PROJECT_DIR = Path("/root/cs336_basics")
MODAL_CHECKPOINT_DIR = REMOTE_PROJECT_DIR / "model_checkpoints"
checkpoint_volume = modal.Volume.from_name("cs336-model-checkpoints", create_if_missing=True)
owt_artifact_volume = modal.Volume.from_name("cs336-owt-artifacts", create_if_missing=True)
sft_data_volume = modal.Volume.from_name("cs336-sft-data", create_if_missing=True)
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "tensorboard", "torch", "triton", "wandb")
    .add_local_file(LOCAL_DIR / "__init__.py", "/root/cs336_basics/__init__.py", copy=True)
    .add_local_file(LOCAL_DIR / "data.py", "/root/cs336_basics/data.py", copy=True)
    .add_local_file(LOCAL_DIR / "model.py", "/root/cs336_basics/model.py", copy=True)
    .add_local_file(SYSTEMS_DIR / "__init__.py", "/root/cs336_systems/__init__.py", copy=True)
    .add_local_file(SYSTEMS_DIR / "flash_attention.py", "/root/cs336_systems/flash_attention.py", copy=True)
    .add_local_file(SYSTEMS_DIR / "flash_attention_triton.py", "/root/cs336_systems/flash_attention_triton.py", copy=True)
    .add_local_file(
        SYSTEMS_DIR / "flash_attention_triton_backward.py",
        "/root/cs336_systems/flash_attention_triton_backward.py",
        copy=True,
    )
    .add_local_dir(LOCAL_DIR / "tokenizer_artifacts", "/root/cs336_basics/tokenizer_artifacts", copy=True)
)


@app.function(
    image=gpu_image,
    gpu=["B200"],
    timeout=50 * 60,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={
        str(MODAL_CHECKPOINT_DIR): checkpoint_volume,
        "/root/owt_artifacts": owt_artifact_volume,
        "/root/sft_data": sft_data_volume,
    },
)
def train(arguments):
   arguments = {**arguments, "device": "cuda", "project_dir": str(REMOTE_PROJECT_DIR)}
   run_name = f"{arguments['run_name']}_{time.time()}"
   if arguments.get("logging_infra") == "wandb":
       wandb.init(entity="lessandro", project="cs336", name=run_name, config=arguments)
       writer = WandbWriter()
   else:
       from torch.utils.tensorboard import SummaryWriter
       writer = SummaryWriter(f"runs/{run_name}")

   torch.set_float32_matmul_precision("high")
   transformer = model.Transformer(
           arguments['d_model'],
           arguments['num_heads'],
           arguments['d_ff'],
           arguments['vocab_size'],
           arguments['max_context_length'],
           arguments['num_layers'],
           arguments['device'],
           dtype=torch.float32,
           attention_backend=arguments.get("attention_backend", "vanilla"))
   transformer = torch.compile(transformer)
   print(f"Initialized Transformer with {transformer.num_params:.2e} parameters")

   project_dir = Path(arguments['project_dir'])
   checkpoint_dir = project_dir / arguments["checkpoint_dir"]
   checkpoint_dir.mkdir(parents=True, exist_ok=True)
   SFT = bool(arguments.get("sft", False))
   if not SFT:
       train_tokens = np.memmap(
          project_dir / arguments['train_data'],
          dtype=np.uint16,
          mode="r",
        )
       validation_tokens = np.memmap(
          project_dir / arguments['validation_data'],
          dtype=np.uint16,
          mode="r",
        )
   else:
       sft_dir = Path("/root/sft_data") / arguments.get("sft_subdir", "alpaca_vocab32000")
       input_ids = np.fromfile(sft_dir / "input_ids.uint16", dtype=np.uint16)
       offsets = np.load(sft_dir / "offsets.npy")
       prompt_lens = np.load(sft_dir / "prompt_lens.npy")
       # Hold out the last `sft_val_examples` examples for validation.
       num_examples = offsets.shape[0] - 1
       sft_val_examples = arguments.get("sft_val_examples", 1000)
       sft_train_hi = num_examples - sft_val_examples
       print(f"SFT data: {num_examples} examples -> {sft_train_hi} train / {sft_val_examples} val")

   if SFT and arguments.get("init_checkpoint"):
       init_path = checkpoint_dir / arguments["init_checkpoint"]
       obj = torch.load(init_path, map_location=arguments['device'])
       transformer.load_state_dict(obj['model'])
       print(f"SFT: loaded base weights from {init_path} (iteration {obj.get('iteration')})")

   optimizer = model.AdamW(
           transformer.parameters(),
           arguments['learning_rate'],
           arguments['betas'],
           1e-6,
           arguments['weight_decay'])

   started_at = time.perf_counter()
   max_wall_clock_seconds = arguments.get("max_wall_clock_seconds")
   total_tokens = 0
   tokens_per_seconds_measure_every = arguments.get("tokens_per_second_measure_every", 50)

   for step in range(arguments['training_steps']):
       elapsed_seconds = time.perf_counter() - started_at
       if max_wall_clock_seconds is not None and elapsed_seconds >= max_wall_clock_seconds:
           print(f"Stopping at step {step}: elapsed_seconds={elapsed_seconds:.2f}")
           break
       optimizer.zero_grad(set_to_none=True)
       if step % tokens_per_seconds_measure_every == 0:
           torch.cuda.synchronize()
           start = time.perf_counter()
       if not SFT:
           x, y_label = data.data_loading(
                   train_tokens, 
                   arguments['batch_size'], 
                   arguments['context_length'],
                   arguments['device'])
       else:
           x, y_label, loss_mask = data.data_loading_with_masking(
                   input_ids,
                   offsets,
                   prompt_lens,
                   arguments['batch_size'],
                   arguments['context_length'],
                   arguments['device'],
                   index_hi=sft_train_hi)
       with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
           z = transformer(x)
           if SFT:
               z = z[loss_mask]
               y_label = y_label[loss_mask]
           loss = model.cross_entropy(z, y_label)

       total_tokens += x.numel()
       loss.backward()

       if "gradient_clip_norm" in arguments:
           model.gradient_clipping(transformer.parameters(), arguments["gradient_clip_norm"])

       lr = model.cosine_learning_rate(
               step,
               arguments["learning_rate"],
               arguments.get("min_learning_rate", arguments["learning_rate"] * 0.1),
               arguments.get("warmup_steps", 0),
               arguments.get("lr_decay_steps", arguments["training_steps"]))
       for group in optimizer.param_groups:
           group["alpha"] = lr
       optimizer.step()
       if step % arguments.get("print_steps", 1) == 0:
           print(step, loss.detach())
       writer.add_scalars("Loss", {'Training': loss.detach()}, step)
       writer.add_scalars("LR", {'Learning Rate': lr}, step)
       writer.add_scalars("Time", {'Wall clock minutes': elapsed_seconds / 60}, step)
       writer.add_scalars("Time", {'Total Tokens Processed': total_tokens}, step)
       writer.add_scalars("GPU Memory", get_gpu_memory_metrics(), step)
       if step % tokens_per_seconds_measure_every == 0:
           torch.cuda.synchronize()
           step_seconds = time.perf_counter() - start
           tokens_per_second = x.numel() / step_seconds
           writer.add_scalars("Time", {'Tokens/Second': tokens_per_second}, step)

       if step % arguments['validation_steps'] == 0:
           with torch.no_grad():
               validation_losses = []
               for _ in range(arguments["num_validation_batches"]):
                   if not SFT:
                       x, y_label = data.data_loading(
                               validation_tokens,
                               arguments['batch_size'],
                               arguments['context_length'],
                               arguments['device'])
                       with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                           z = transformer(x)
                           loss = model.cross_entropy(z, y_label)
                   else:
                       x, y_label, loss_mask = data.data_loading_with_masking(
                               input_ids,
                               offsets,
                               prompt_lens,
                               arguments['batch_size'],
                               arguments['context_length'],
                               arguments['device'],
                               index_lo=sft_train_hi,
                               index_hi=num_examples - 1)
                       with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                           z = transformer(x)
                           z = z[loss_mask]
                           y_label = y_label[loss_mask]
                           loss = model.cross_entropy(z, y_label)
                   validation_losses.append(loss.detach())
               validation_loss = torch.stack(validation_losses).mean()
               print("validation loss: ", step, validation_loss.detach())
               writer.add_scalars("Loss", {'Validation': validation_loss.detach()}, step)

       if step > 0 and step % arguments['checkpoint_steps'] == 0:
           out = get_checkpoint_path(
                   checkpoint_dir,
                   run_name,
                   step)
           print(f"saving {(3 * transformer.num_params):.2e} parameters model to checkpoint {out}")
           model.save_checkpoint(
                   transformer,
                   optimizer,
                   step,
                   out)
           checkpoint_volume.commit()

   final_out = checkpoint_dir / f"{run_name}_step{step}_final.pt"
   print(f"saving final model to checkpoint {final_out}")
   model.save_checkpoint(transformer, optimizer, step, final_out)
   checkpoint_volume.commit()

   if arguments.get("logging_infra") == "wandb":
       wandb.finish()

@app.local_entrypoint()
def sweep(config: str):
  with open(config) as file:
      base = json.load(file)

  configs = []
  for learning_rate in [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00005]:
      configs.append({
          **base,
          "learning_rate": learning_rate,
          "run_name": f"b200_lr{learning_rate}",
          "logging_infra": "wandb",
          "training_steps": 50,
      })

  list(train.map(configs))


@app.local_entrypoint()
def final_sweep(
        max_wall_clock_seconds: int = 300,
        batch_size: int = 256,
        wave_size: int = 4,
        limit: int = 0,
        start_index: int = 0):
  with open(LOCAL_DIR / "training_configs" / "B200_owt_32k_45min.json") as file:
      base = json.load(file)

  architectures = [
      {"d_model": 384, "num_heads": 12, "d_ff": 1024, "num_layers": 4},
      {"d_model": 512, "num_heads": 16, "d_ff": 1344, "num_layers": 4},
      {"d_model": 640, "num_heads": 20, "d_ff": 1728, "num_layers": 4},
      {"d_model": 768, "num_heads": 24, "d_ff": 2048, "num_layers": 4},
      {"d_model": 512, "num_heads": 8, "d_ff": 1344, "num_layers": 4},
      {"d_model": 768, "num_heads": 12, "d_ff": 2048, "num_layers": 4},
      {"d_model": 512, "num_heads": 16, "d_ff": 1344, "num_layers": 6},
      {"d_model": 768, "num_heads": 24, "d_ff": 2048, "num_layers": 6},
  ]
  learning_rates = [0.001, 0.0007, 0.0003]

  configs = []
  for architecture in architectures:
      for learning_rate in learning_rates:
          head_dim = architecture["d_model"] // architecture["num_heads"]
          lr_label = f"{learning_rate:g}".replace(".", "p")
          configs.append({
              **base,
              **architecture,
              "context_length": 512,
              "max_context_length": 512,
              "batch_size": batch_size,
              "learning_rate": learning_rate,
              "min_learning_rate": learning_rate * 0.1,
              "max_wall_clock_seconds": max_wall_clock_seconds,
              "run_name": (
                  "final_sweep_r1"
                  f"_dm{architecture['d_model']}"
                  f"_h{architecture['num_heads']}"
                  f"_hd{head_dim}"
                  f"_l{architecture['num_layers']}"
                  f"_ff{architecture['d_ff']}"
                  f"_b{batch_size}"
                  f"_lr{lr_label}"
              ),
          })

  if start_index:
      configs = configs[start_index:]
  if limit:
      configs = configs[:limit]

  print(f"Prepared {len(configs)} final sweep runs")
  for wave_index in range(0, len(configs), wave_size):
      wave = configs[wave_index:wave_index + wave_size]
      print(f"Starting wave {wave_index // wave_size + 1}: {len(wave)} runs")
      for config in wave:
          print("  ", config["run_name"])
      list(train.map(wave))


@app.local_entrypoint()
def deep6_quick_sweep(
        max_wall_clock_seconds: int = 180,
        wave_size: int = 4,
        limit: int = 0):
  with open(LOCAL_DIR / "training_configs" / "B200_owt_32k_45min_best_sweep.json") as file:
      base = json.load(file)

  variants = [
      {"label": "base", "learning_rate": 0.001, "warmup_steps": 100, "betas": [0.9, 0.99], "weight_decay": 0.1, "gradient_clip_norm": 1.0, "lr_decay_steps": 10000},
      {"label": "lr7e4", "learning_rate": 0.0007, "warmup_steps": 100, "betas": [0.9, 0.99], "weight_decay": 0.1, "gradient_clip_norm": 1.0, "lr_decay_steps": 10000},
      {"label": "lr13e4", "learning_rate": 0.0013, "warmup_steps": 100, "betas": [0.9, 0.99], "weight_decay": 0.1, "gradient_clip_norm": 1.0, "lr_decay_steps": 10000},
      {"label": "warm50", "learning_rate": 0.001, "warmup_steps": 50, "betas": [0.9, 0.99], "weight_decay": 0.1, "gradient_clip_norm": 1.0, "lr_decay_steps": 10000},
      {"label": "warm200", "learning_rate": 0.001, "warmup_steps": 200, "betas": [0.9, 0.99], "weight_decay": 0.1, "gradient_clip_norm": 1.0, "lr_decay_steps": 10000},
      {"label": "b2p95", "learning_rate": 0.001, "warmup_steps": 100, "betas": [0.9, 0.95], "weight_decay": 0.1, "gradient_clip_norm": 1.0, "lr_decay_steps": 10000},
      {"label": "b2p999", "learning_rate": 0.001, "warmup_steps": 100, "betas": [0.9, 0.999], "weight_decay": 0.1, "gradient_clip_norm": 1.0, "lr_decay_steps": 10000},
      {"label": "clip0p5", "learning_rate": 0.001, "warmup_steps": 100, "betas": [0.9, 0.99], "weight_decay": 0.1, "gradient_clip_norm": 0.5, "lr_decay_steps": 10000},
      {"label": "wd0p05_decay6k", "learning_rate": 0.001, "warmup_steps": 100, "betas": [0.9, 0.99], "weight_decay": 0.05, "gradient_clip_norm": 1.0, "lr_decay_steps": 6000},
  ]

  configs = []
  for variant in variants:
      configs.append({
          **base,
          **variant,
          "d_model": 768,
          "num_heads": 12,
          "d_ff": 2048,
          "num_layers": 6,
          "batch_size": 128,
          "context_length": 512,
          "max_context_length": 512,
          "max_wall_clock_seconds": max_wall_clock_seconds,
          "min_learning_rate": variant["learning_rate"] * 0.1,
          "run_name": f"deep6_quick_sweep_{variant['label']}",
      })

  if limit:
      configs = configs[:limit]

  print(f"Prepared {len(configs)} deep6 quick sweep runs")
  for wave_index in range(0, len(configs), wave_size):
      wave = configs[wave_index:wave_index + wave_size]
      print(f"Starting wave {wave_index // wave_size + 1}: {len(wave)} runs")
      for config in wave:
          print("  ", config["run_name"])
      list(train.map(wave))


@app.local_entrypoint()
def main(config: str):
    with open(config) as file:
        arguments = json.load(file)
    train.remote(arguments)
