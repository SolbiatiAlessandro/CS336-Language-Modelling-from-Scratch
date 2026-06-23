"""Local web UI for chatting with a trained CS336 checkpoint on MPS.

This is *plumbing only* — all model / decoding logic lives in
``cs336_basics/inference.py`` and is used verbatim:

* ``inference.load_model``  streams a checkpoint from the Modal volume into
  memory and loads it into a ``Transformer`` on MPS (nothing lands on disk).
* ``inference.generate``    is the user's own decoding loop (temperature +
  nucleus sampling) that returns the completion together with the per-token
  entropy and max-probability traces.

The server:
  1. Streams the requested checkpoint from the Modal ``cs336-model-checkpoints``
     volume straight into memory (no local copy) and loads it on ``mps`` in a
     background thread, so the page renders instantly and shows a live
     "streaming / loading / ready" status.
  2. Serves a ChatGPT-style chat UI (``webui/index.html``).
  3. Exposes a tiny JSON API the UI talks to:
       GET  /api/status      -> loader state machine + model card once ready
       GET  /api/model-info  -> architecture, param count, W&B losses + curves
       POST /api/generate     -> runs inference.generate, returns text + traces
       POST /api/load         -> switch to a different checkpoint

W&B "best training losses" are pulled from the leaderboard run referenced in
README.md (``lessandro/cs336/zzd4j6h7``) using the credentials in ~/.netrc.
Modal access uses ~/.modal.toml. No keys are embedded here.

Run:
    uv run serve.py                       # serve final checkpoint on :8000
    uv run serve.py --port 8080
"""

import argparse
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
WEBUI_DIR = REPO_DIR / "webui"

MODAL_VOLUME = "cs336-model-checkpoints"
WANDB_RUN = "lessandro/cs336/zzd4j6h7"  # leaderboard run from README.md

# Checkpoints from the final leaderboard run, at three points in training.
# (Same architecture; nice for watching the model "wake up" across steps.)
CHECKPOINTS = [
    {
        "file": "B200_owt_32k_45min_deep6_clip0p5_ckpts_1782191080.0134425_step16612_final.pt",
        "label": "Final · step 16,612",
        "step": 16612,
    },
    {
        "file": "B200_owt_32k_45min_deep6_clip0p5_ckpts_1782191080.0134425_step1480_frac10.pt",
        "label": "10% · step 1,480",
        "step": 1480,
    },
    {
        "file": "B200_owt_32k_45min_deep6_clip0p5_ckpts_1782191080.0134425_step5_early.pt",
        "label": "Untrained · step 5",
        "step": 5,
    },
]
DEFAULT_CHECKPOINT = CHECKPOINTS[0]["file"]

DEVICE = "mps"  # inference.generate() is hardcoded to mps

# ---------------------------------------------------------------------------
# Shared loader state (read by the HTTP threads, written by the loader thread).
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_gen_lock = threading.Lock()  # serialize MPS inference across requests
STATE = {
    "status": "starting",  # starting | streaming | loading | ready | error
    "detail": "",
    "checkpoint": DEFAULT_CHECKPOINT,
    "model_info": None,
    "error": None,
}

_WANDB_CACHE = None  # populated once; reused across checkpoint switches

# Imported lazily inside the loader thread so the web server starts even if a
# heavy import is slow.
inference = None
Tokenizer = None


def _set_state(**kw):
    with _state_lock:
        STATE.update(kw)


def _get_state():
    with _state_lock:
        return dict(STATE)


# ---------------------------------------------------------------------------
# W&B: best / final training losses + loss curves for the right-hand panel.
# ---------------------------------------------------------------------------
def fetch_wandb_info():
    global _WANDB_CACHE
    if _WANDB_CACHE is not None:
        return _WANDB_CACHE
    try:
        import wandb

        api = wandb.Api()
        run = api.run(WANDB_RUN)
        hist = run.history(keys=["Loss/Training", "Loss/Validation", "_step"], pandas=True)
        hist = hist.dropna(subset=["Loss/Training"], how="all")

        steps = [int(s) for s in hist["_step"].tolist()]
        train = [None if v != v else float(v) for v in hist["Loss/Training"].tolist()]
        val = [None if v != v else float(v) for v in hist["Loss/Validation"].tolist()]

        tr = hist["Loss/Training"].dropna()
        va = hist["Loss/Validation"].dropna()
        info = {
            "run_name": run.name,
            "run_url": run.url,
            "final_train": float(run.summary.get("Loss/Training")),
            "final_val": float(run.summary.get("Loss/Validation")),
            "min_train": float(tr.min()),
            "min_train_step": int(hist.loc[tr.idxmin(), "_step"]),
            "min_val": float(va.min()),
            "min_val_step": int(hist.loc[va.idxmin(), "_step"]),
            "runtime_min": float(run.summary.get("Time/Wall clock minutes") or 0.0),
            "curve": {"steps": steps, "train": train, "val": val},
        }
        _WANDB_CACHE = info
        return info
    except Exception as e:  # offline / no creds — UI still works without it
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Model load — streams the checkpoint from Modal into memory (verbatim
# inference.load_model), nothing touches disk.
# ---------------------------------------------------------------------------
def load_into_inference(file_name):
    """Stream + load the checkpoint, wiring up the module globals that
    inference.generate() closes over. Returns the model-card dict."""
    global inference, Tokenizer
    if inference is None:
        import cs336_basics.inference as _inf
        from cs336_basics.tokenizer import Tokenizer as _Tok

        inference = _inf
        Tokenizer = _Tok

    _set_state(status="streaming", detail=f"Streaming {file_name} from Modal → {DEVICE} …")
    with open(inference.CONFIG_PATH) as f:
        config = json.load(f)

    # Tokenizer + model are the globals generate() reads.
    inference.tokenizer = Tokenizer.from_files(
        inference.TOKENIZER_DIR / "vocab.json",
        inference.TOKENIZER_DIR / "merges.json",
        special_tokens=["<|endoftext|>"],
    )
    # Passing the bare filename (not a local path) makes inference stream it
    # from the Modal volume straight into memory.
    net, iteration, context_length = inference.load_model(file_name, config, DEVICE, MODAL_VOLUME)
    inference.model = net

    meta = next((c for c in CHECKPOINTS if c["file"] == file_name), None)
    model_info = {
        "checkpoint": file_name,
        "checkpoint_label": meta["label"] if meta else file_name,
        "device": DEVICE,
        "iteration": iteration,
        "num_params": int(net.num_params),
        "arch": {
            "d_model": config["d_model"],
            "num_heads": config["num_heads"],
            "d_ff": config["d_ff"],
            "num_layers": config["num_layers"],
            "vocab_size": config["vocab_size"],
            "context_length": context_length,
        },
        "config": {
            "learning_rate": config.get("learning_rate"),
            "batch_size": config.get("batch_size"),
            "weight_decay": config.get("weight_decay"),
            "warmup_steps": config.get("warmup_steps"),
            "betas": config.get("betas"),
            "attention_backend": config.get("attention_backend"),
        },
        "checkpoints": CHECKPOINTS,
        "wandb": fetch_wandb_info(),
    }
    return model_info


def loader_thread(file_name):
    try:
        info = load_into_inference(file_name)
        _set_state(status="ready", detail="", checkpoint=file_name, model_info=info, error=None)
    except Exception as e:
        traceback.print_exc()
        _set_state(status="error", detail="", error=f"{type(e).__name__}: {e}")


def start_load(file_name):
    _set_state(status="streaming", detail="Preparing …", checkpoint=file_name, model_info=None, error=None)
    threading.Thread(target=loader_thread, args=(file_name,), daemon=True).start()


# ---------------------------------------------------------------------------
# Inference: call the verbatim generate(), strip the prompt prefix.
# ---------------------------------------------------------------------------
def run_generation(prompt, max_tokens, temperature, top_p, max_seq_len):
    with _gen_lock:
        if _get_state()["status"] != "ready":
            raise RuntimeError("model not ready")
        full_text, entropies, p_maxes = inference.generate(
            prompt=prompt,
            max_tokens=int(max_tokens),
            debug=False,
            temperature=float(temperature),
            p_threshold=float(top_p),
            max_sequence_length=int(max_seq_len),
        )
    # generate() returns decode(prompt_tokens + new_tokens); for byte-level BPE
    # decode is concatenative, so the decoded prompt is an exact prefix.
    prompt_decoded = inference.tokenizer.decode(inference.tokenizer.encode(prompt))
    completion = full_text[len(prompt_decoded):]
    avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    avg_p_max = sum(p_maxes) / len(p_maxes) if p_maxes else 0.0
    return {
        "prompt": prompt,
        "completion": completion,
        "full_text": full_text,
        "entropies": entropies,
        "p_maxes": p_maxes,
        "avg_entropy": avg_entropy,
        "avg_p_max": avg_p_max,
        "num_tokens": len(entropies),
    }


# ---------------------------------------------------------------------------
# HTTP handler.
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quieter console
        pass

    def _send(self, code, body, content_type="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = (WEBUI_DIR / "index.html").read_text()
            return self._send(200, html, "text/html; charset=utf-8")
        if self.path == "/api/status":
            return self._send(200, _get_state())
        if self.path == "/api/model-info":
            st = _get_state()
            if st["status"] != "ready":
                return self._send(409, {"error": "not ready", "status": st["status"]})
            return self._send(200, st["model_info"])
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            if self.path == "/api/generate":
                data = self._read_json()
                prompt = (data.get("prompt") or "").strip()
                if not prompt:
                    return self._send(400, {"error": "empty prompt"})
                result = run_generation(
                    prompt=prompt,
                    max_tokens=data.get("max_tokens", 150),
                    temperature=data.get("temperature", 0.7),
                    top_p=data.get("top_p", 0.9),
                    max_seq_len=data.get("max_seq_len", 512),
                )
                return self._send(200, result)
            if self.path == "/api/load":
                data = self._read_json()
                file_name = data.get("checkpoint")
                if file_name not in [c["file"] for c in CHECKPOINTS]:
                    return self._send(400, {"error": "unknown checkpoint"})
                if _get_state()["checkpoint"] == file_name and _get_state()["status"] == "ready":
                    return self._send(200, {"ok": True, "already": True})
                start_load(file_name)
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint file name to load first.")
    args = parser.parse_args()

    start_load(args.checkpoint)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  CS336 chat UI  →  {url}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  device: {DEVICE}   (loading in background — watch the status pill)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
