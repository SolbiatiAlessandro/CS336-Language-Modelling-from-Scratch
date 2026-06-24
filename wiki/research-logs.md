# Research Logs

Running notes on experiments, findings, and decisions for the CS336 language
model. Newest entries on top.

---

## 2026-06-24 — SFT on Alpaca: base rambler → instruction-follower

Fine-tuned the pretrained 67M base checkpoint on `tatsu-lab/alpaca` (tokenized with
our 32k BPE, response-only loss mask). **It works — the model now follows
instructions and stops.**

### Setup
- Config: `training_configs/sft_alpaca.json` (`sft: true`).
- Init from base `...step16612_final.pt`; fresh optimizer.
- LR 2e-5 → 2e-6 cosine, warmup 30, batch 64, ctx 512, betas [0.9,0.95], wd 0.0,
  grad clip 1.0.
- `training_steps` 2400 ≈ **~3 epochs** (153,600 example-draws / 51,002 train
  examples, with replacement). Held out last 1000 examples for masked validation.
- W&B run: `lessandro/cs336/sft_alpaca_deep6_1782286413.7466567`.
- Wall-clock ~5 min on B200, ~$0.50.

### How we decided "how long"
SFT duration is **not** a compute/Chinchilla decision like pretraining. Convention
is **1–3 epochs** (small curated data; >3–5 overfits / forgets). Alpaca itself used
3. So 2400 was set as a **ceiling**, with held-out val + checkpoints every 600 steps
so the real stopping point is found by **early stopping on val loss**, not fixed up
front. Pretraining = scaling law; SFT = epoch convention + early stop.

### Result
- Train (masked) 3.39 → ~2.2–2.4; **val (masked, held-out) 3.50 → ~2.42** (PPL ≈ 11
  over the response distribution; NOT comparable to the 3.42 pretraining loss —
  different, response-only token set).
- **No overfitting**: train ≈ val (gap ~0.1–0.2 nats), val **plateaued flat** from
  ~step 1500, never turned up. 3 epochs was a safe ceiling; final checkpoint is good.
- Final SFT checkpoint on `cs336-model-checkpoints`:
  `sft_alpaca_deep6_1782286413.7466567_step*_final.pt`.

### Qualitative (via inference.py `sft="alpaca"` template mode)
- "What's the boiling temperature of water?" → "The boiling temperature of water is
  8.8 degrees Fahrenheit.\<|endoftext|\>"  — **format perfect, fact wrong.**
- "How many fingers on a hand?" → "There are 5." — correct.

### Takeaway: format vs. facts
SFT taught **behavior, not knowledge.** It learned to answer concisely, on-topic, and
to emit `<|endoftext|>` and stop (no more news-article rambling). But factual errors
(8.8°F vs 212°F) are the **67M param ceiling** — world knowledge lives in pretraining
/ scale, and SFT can't inject it. The model is also **confidently wrong** (entropy
~0.75) — SFT made it decisive, which is great for format and exposes the missing
facts. Clean separation: SFT fixed behavior; knowledge is param-bound.

### Code wiring added this session
- `data.py`: `data_loading_with_masking` (response-only mask, truncation, train/val
  index range), returns torch tensors (long x/y, bool mask) on device.
- `train_model.py`: SFT branch (load tokens from mounted `cs336-sft-data` volume),
  load base checkpoint, masked loss via `z[mask]`/`y[mask]`, held-out masked
  validation, **unconditional final save** (this repo previously lacked one).
- `inference.py`: `sft="alpaca"` mode wraps the prompt in the training template so
  chat triggers the learned behavior.

### Next ideas
- Bigger pretrain (more params) is the lever for facts, not more SFT.
- RLVR / GSM8K (assignment 5 style) for verifiable-reward improvement.
- Multi-turn / better SFT mix (SmolTalk) if we want broader instruction-following.

---

## 2026-06-23 — Scaling check: are we trained optimally? (+ cost / storage)

Question: the final checkpoint trained for 45 min — (1) did we use all the
training data, and (2) if we keep training, what loss do we get?

### (1) Did we use all the data? No — ~a third of it.
From the OWT manifest (`cs336-owt-artifacts` volume):
- **train.uint16 = 5,526,394,772 bytes / 2 (uint16) = 2.76B tokens** available
- valid = 67.3M tokens

The run processed **1.088B tokens** (16,612 steps × 128 batch × 512 ctx):
- 1.088B / 2.76B = **39% of one epoch**
- `data_loading` samples random windows *with replacement* → expected unique
  coverage `1 - e^(-0.39) ≈ 33%`. The model never saw ~two-thirds of the corpus.
- We are **data-rich, not data-bound**. A full epoch ≈ 104 min on B200
  (2.76B / ~440k tok/s) ≈ 1.7 B200-hr ≈ ~$11.

### (2) Scaling law fit (from W&B run `lessandro/cs336/v5quc8v4`)
Pulled the 333 validation points and fit the Chinchilla data term
`L(D) = E + B·D^(-beta)` over D > 1e7:

**L(D) = 3.123 + 12020·D^(-0.516)**, RMSE 0.024 (tracks the curve almost exactly).
Plot: `wiki/assets/loss_curve.png`.

| Tokens D | Predicted val loss | PPL |
| --- | --- | --- |
| 1.09B (run end) | 3.39 (measured 3.42) | ~29 |
| 2.76B (1 full epoch) | 3.29 | 26.7 |
| 5.5B (2 epochs) | 3.24 | 25.4 |
| 11B (4 epochs) | 3.20 | 24.6 |
| asymptote (D→∞) | 3.12 | 22.7 |

So finishing the epoch buys ~3.42 → 3.29 (**~0.13 nats**) — real but modest. The
curve has clearly bent into its flat regime past ~3e8 tokens.

### Big caveat: the fit is confounded by the LR schedule
β = 0.52 is ~2× Chinchilla's 0.28. The cosine LR decay (to min at ~0.66B tokens,
`lr_decay_steps=10000`) does much of the loss-dropping, so the single-run fit
absorbs the *schedule*, not just data scaling. Therefore:
- The asymptote E = 3.12 is **this schedule's floor**, not the true irreducible
  loss (note it sits barely below the final 3.42 — the tail flattened because LR
  hit min, not because data ran out).
- The extrapolation is **pessimistic for training longer *properly***: a fresh run
  with a schedule annealed to 2.76B tokens would likely beat 3.29. A single
  annealed trajectory underestimates a longer dedicated run.
- A trustworthy scaling law needs the real Chinchilla method: several short runs
  each annealed to their own token budget, fit the law to their **endpoints**.

### Cost reference
- **B200 = $6.25/hr** ($0.001736/sec). A 45-min run ≈ **$4.69**. Full epoch ≈ $11.
- **Modal volume storage = $0.09/GiB/mo, first 1 TiB/mo free.** Currently using
  108.2 GiB total across 6 volumes → **$0** (11% of free tier). cs336 volumes:
  owt-artifacts 27.8 GiB, model-checkpoints 19.0 GiB.

### Decision
The model is **trained pretty optimally for its size** (67M params, D/N ≈ 16, just
under Chinchilla-optimal 20). It is **param-bound, not data-bound** — the lever
for lower loss is a bigger model (scale N and D together), not more tokens into
this one.

Not worth training more right now: to justify it we'd first need to know exactly
**what loss target we want and why**, **how much it costs**, and **how to do it**
(proper longer schedule, not resume-from-min-LR). Parking that.

**Next priority: SFT** (on `tatsu-lab/alpaca`) so we can actually *talk to it* and
get a qualitative feel for the model — more valuable right now than squeezing
~0.1 nats out of pretraining.

---

## 2026-06-23 — From base LM to instruction following: why it rambles, and the SFT path

### The observation
Prompted the trained model (final checkpoint, val loss 3.42) with a GSM8K-style
question via `inference.py`:

> Natalia sold clips to 48 of her friends in April, and then she sold half as
> many clips in May. How many clips did Natalia sell altogether in April and May?

It did **not** answer (correct answer: April 48, May 24, total 48 + 24 = 72).
Instead it free-associated into a news-article-style continuation about Natalia,
advertising campaigns, Oprah, etc.

Fun coincidence: **that Natalia problem is literally GSM8K** — the canonical
first example in the test set. We were prompting with instruction-tuning data
without realizing it.

### Why this happens (expected, not a bug)
What we trained is a **base (pretraining) LM**: it only learned
`P(next token | preceding text)` over raw OpenWebText (a web crawl, mostly news
and blogs). Given a math word problem it pattern-matches the genre and continues
in it. Fluency + world knowledge is exactly what pretraining gives you;
instruction following is a *separate* capability layered on top.

Evidence it's doing its job: entropy across checkpoints dropped 10.33 → 1.04 →
1.71 nats (early → 10% → final), going from uniform gibberish to fluent,
grammatical English with real-world entities.

### What we're missing
Two coupled things — and the tokens are only half of it:

1. **Chat/role structure.** Our tokenizer has exactly one special token,
   `<|endoftext|>` (id 256). No `<|system|>`/`<|user|>`/`<|assistant|>`, no turn
   delimiters. The model has no structural notion of "instruction → reply."
2. **The training stage that teaches it.** Adding role tokens does nothing on its
   own. Instruction following comes from **post-training**: SFT on
   (prompt, response) pairs, then optionally preference optimization (RLHF/DPO).
   Pipeline: pretraining (done) → SFT → alignment.

### Do we actually need new special tokens? No.
A chat template can be **plain text** that tokenizes with the existing BPE:

```
Question: Natalia sold clips to 48 of her friends...
Answer: She sold 48/2 = 24 in May. 48 + 24 = 72.<|endoftext|>
```

The model learns the `Question:/Answer:` structure from the text pattern; reuse
`<|endoftext|>` as the stop signal. **Zero vocab changes, zero embedding
surgery.** This is endorsed by assignment 5 itself: its `r1_zero` prompt is a
plain-text template ("put your answer in \boxed{}"), not special role tokens.

Adding *real* special tokens is harder in **our specific model**:
- **Tied embeddings** — `head` shares `embedding.embeddings`, so we'd resize a
  `(32000, 768)` matrix; new rows start random and need training.
- **Tokenizer numbering** — `first_merge_number = 255 + len(special_tokens) + 1`.
  Trained vocab is bytes 0–255, EOT at 256, merges from 257. Inserting new
  specials in that slot shifts every trained merge id. We'd have to *append* new
  tokens at the end (ids 32000+) and patch the tokenizer. Doable but fiddly →
  start with plain text.

### Is the SFT loop itself easy?
Mostly. It's the existing pretraining loop with **one real change: loss masking**
— compute next-token loss only on the **response** span, not the prompt span (the
prompt is conditioning, not a target). Build a mask that's 0 over `Question: …`
and 1 over `Answer: …` and zero those positions in the cross-entropy. Everything
else (data loading, AdamW, forward pass) we already have.

Honest difficulty breakdown:
- Tokens/template: easy (plain text, reuse EOT).
- SFT loop: easy-ish (existing loop + loss masking).
- Hard/uncertain: (a) data quality, and (b) a **67M-param model is tiny** — even
  after clean SFT, expect "answers in Q/A shape" more than "solves GSM8K."

### What assignment 5 actually does (checked the PDF, 39 pages)
Assignment 5 is **not** the classic base → SFT → RLHF recipe. It does **pure RL
with verifiable rewards (RLVR)** on **OLMo-2-0425-1B** (an existing AI2 base
model, not ours), targeting GSM8K:
1. Prompting baseline (`r1_zero` template)
2. On-policy **GRPO** (Group Relative Policy Optimization)
3. RL variants: **RFT, Dr. GRPO, MaxRL**
4. Off-policy GRPO + clipping

- **No standalone SFT, no DPO/RLHF, no chat-token training.**
- The one nuance: **RFT (Rejection Fine-Tuning) / Expert Iteration** (§5.2) *is*
  supervised fine-tuning — but on the model's **own reward-filtered correct
  rollouts**, not human demos. Framed as a GRPO special case
  (`baseline="none", advantage_normalizer="none"`).
- They skip human SFT because (1) they start from an already-capable base, and
  (2) the task has a **verifiable reward** (is the math answer right?), so they
  bootstrap with RL/RFT directly.

### SFT dataset options
HF's famous *pretraining* sets (FineWeb, Cosmopedia) are not SFT. For SFT:

| Dataset (HF repo) | ~Size | Notes |
| --- | --- | --- |
| `HuggingFaceTB/smoltalk` | ~1M | HF's own SFT mix, built for *small* models (SmolLM2) |
| `allenai/tulu-3-sft-mixture` | ~939k | **The OLMo/AI2 one**; broad, high quality |
| `teknium/OpenHermes-2.5` | ~1M | Popular general-purpose chat SFT (GPT-4 generated) |
| `HuggingFaceH4/no_robots` | ~10k | HF, fully human-written, high quality, small |
| `tatsu-lab/alpaca` | ~52k | Classic starter; simple instruction/response |
| `databricks/databricks-dolly-15k` | ~15k | Human-written, permissive license |
| `openai/gsm8k` | ~7.5k train | Math, **verifiable answers**, matches Natalia example |

"The OLMo SFT dataset" = **Tülu 3 SFT Mixture** (`allenai/tulu-3-sft-mixture`).
OLMo-2-Instruct = base → SFT on that → DPO → RLVR.

### A bit of history: how was the *first* SFT data made, before AI could generate it?
The premise "all SFT data is AI-generated" is true today but was the opposite at
the start. Two pre-AI human sources:

1. **Repurposing existing NLP datasets** (FLAN, Google 2021; T0, BigScience
   2021): take a decade of human-annotated supervised datasets (sentiment, QA,
   summarization, NLI…) and **reframe** them as natural-language instructions via
   ~10 **human-written templates** per task. No AI involved — just relabeling.
2. **Paying humans to write demonstrations** (InstructGPT, OpenAI 2022 — the
   ChatGPT ancestor): ~40 contractors hand-wrote ideal responses to prompts (many
   from real GPT-3 API queries). ~13k human-written demonstrations = the SFT set.
   Then RLHF: humans *ranked* outputs to train a reward model.

The flip to AI-generated came right after a capable model existed:
- **Self-Instruct** (late 2022): 175 human seed tasks → GPT-3 generates thousands
  more.
- **Alpaca** (Stanford, early 2023): self-instruct via `text-davinci-003`, 52k
  examples for ~$500 → kicked off the "distill from a bigger model" era.

Punchline: **humans were the first teacher model.** AI-generated SFT data is what
became possible *after* the first human-bootstrapped model crossed the capability
threshold to teach its successors. Also: GSM8K solutions were human-written
(pre-AI), and RLVR (GRPO) is arguably the next escape from human data — just
verify correctness, no teacher needed at all.

### Next steps / decisions
- **Plan: SFT on [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca)**
  (~52k, simple instruction/response). Will work on this later.
- Approach: plain-text template (reuse `<|endoftext|>`), loss-masked on the
  response span, fine-tune the `final` checkpoint. No new special tokens for v1.
- Consider GSM8K afterward for a verifiable-reward experiment (sets up RFT/GRPO
  later, à la assignment 5).
- Open question: is 67M big enough to show meaningful instruction following, or
  do we need a larger pretrain first?
