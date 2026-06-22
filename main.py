import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import wandb

import cs336_basics.data as data
import cs336_basics.model as model


CONFIG_PATH = Path(__file__).parent / "cs336_basics" / "training_configs" / "final_config.json"


class WandbWriter:
    def __init__(self, enabled: bool, run_name: str, config: dict):
        self.enabled = enabled
        if enabled:
            wandb.init(entity="lessandro", project="cs336", name=run_name, config=config)

    def add_scalars(self, group, scalars, step):
        if not self.enabled:
            return
        wandb.log(
            {
                f"{group}/{name}": value.detach().item() if isinstance(value, torch.Tensor) else value
                for name, value in scalars.items()
            },
            step=step,
        )

    def close(self):
        if self.enabled:
            wandb.finish()


def gpu_memory_metrics():
    gib = 1024 ** 3
    return {
        "Allocated GiB": torch.cuda.memory_allocated() / gib,
        "Reserved GiB": torch.cuda.memory_reserved() / gib,
        "Peak Allocated GiB": torch.cuda.max_memory_allocated() / gib,
        "Peak Reserved GiB": torch.cuda.max_memory_reserved() / gib,
    }


def resolve_data_path(config_value: str, env_name: str) -> Path:
    override = os.environ.get(env_name)
    if override:
        return Path(override)
    return Path(config_value)


def main():
    with CONFIG_PATH.open() as file:
        arguments = json.load(file)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this reproduction script.")

    arguments["device"] = "cuda"
    arguments["train_data"] = str(resolve_data_path(arguments["train_data"], "TRAIN_DATA"))
    arguments["validation_data"] = str(resolve_data_path(arguments["validation_data"], "VALID_DATA"))

    run_name = f"{arguments['run_name']}_{time.time()}"
    logging_enabled = arguments.get("logging_infra") == "wandb" and bool(os.environ.get("WANDB_API_KEY"))
    writer = WandbWriter(logging_enabled, run_name, arguments)

    torch.set_float32_matmul_precision("high")
    transformer = model.Transformer(
        arguments["d_model"],
        arguments["num_heads"],
        arguments["d_ff"],
        arguments["vocab_size"],
        arguments["max_context_length"],
        arguments["num_layers"],
        arguments["device"],
        dtype=torch.float32,
        attention_backend=arguments.get("attention_backend", "vanilla"),
    )
    transformer = torch.compile(transformer)
    print(f"Initialized Transformer with {transformer.num_params:.2e} parameters")

    train_tokens = np.memmap(arguments["train_data"], dtype=np.uint16, mode="r")
    validation_tokens = np.memmap(arguments["validation_data"], dtype=np.uint16, mode="r")

    optimizer = model.AdamW(
        transformer.parameters(),
        arguments["learning_rate"],
        arguments["betas"],
        1e-6,
        arguments["weight_decay"],
    )

    started_at = time.perf_counter()
    total_tokens = 0
    max_wall_clock_seconds = arguments.get("max_wall_clock_seconds")
    tokens_per_second_measure_every = arguments.get("tokens_per_second_measure_every", 50)

    try:
        for step in range(arguments["training_steps"]):
            elapsed_seconds = time.perf_counter() - started_at
            if max_wall_clock_seconds is not None and elapsed_seconds >= max_wall_clock_seconds:
                print(f"Stopping at step {step}: elapsed_seconds={elapsed_seconds:.2f}")
                break

            optimizer.zero_grad(set_to_none=True)
            if step % tokens_per_second_measure_every == 0:
                torch.cuda.synchronize()
                step_started_at = time.perf_counter()

            x, y_label = data.data_loading(
                train_tokens,
                arguments["batch_size"],
                arguments["context_length"],
                arguments["device"],
            )
            total_tokens += x.numel()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = transformer(x)
                loss = model.cross_entropy(z, y_label)
            loss.backward()

            if "gradient_clip_norm" in arguments:
                model.gradient_clipping(transformer.parameters(), arguments["gradient_clip_norm"])

            lr = model.cosine_learning_rate(
                step,
                arguments["learning_rate"],
                arguments.get("min_learning_rate", arguments["learning_rate"] * 0.1),
                arguments.get("warmup_steps", 0),
                arguments.get("lr_decay_steps", arguments["training_steps"]),
            )
            for group in optimizer.param_groups:
                group["alpha"] = lr
            optimizer.step()

            if step % arguments.get("print_steps", 1) == 0:
                print(step, loss.detach())

            writer.add_scalars("Loss", {"Training": loss.detach()}, step)
            writer.add_scalars("LR", {"Learning Rate": lr}, step)
            writer.add_scalars("Time", {"Wall clock minutes": elapsed_seconds / 60}, step)
            writer.add_scalars("Time", {"Total Tokens Processed": total_tokens}, step)
            writer.add_scalars("GPU Memory", gpu_memory_metrics(), step)

            if step % tokens_per_second_measure_every == 0:
                torch.cuda.synchronize()
                step_seconds = time.perf_counter() - step_started_at
                writer.add_scalars("Time", {"Tokens/Second": x.numel() / step_seconds}, step)

            if step % arguments["validation_steps"] == 0:
                with torch.no_grad():
                    validation_losses = []
                    for _ in range(arguments["num_validation_batches"]):
                        x, y_label = data.data_loading(
                            validation_tokens,
                            arguments["batch_size"],
                            arguments["context_length"],
                            arguments["device"],
                        )
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            z = transformer(x)
                            loss = model.cross_entropy(z, y_label)
                        validation_losses.append(loss.detach())
                    validation_loss = torch.stack(validation_losses).mean()
                    print("validation loss: ", step, validation_loss.detach())
                    writer.add_scalars("Loss", {"Validation": validation_loss.detach()}, step)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
