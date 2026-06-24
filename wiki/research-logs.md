# Research Logs

Running notes on experiments, findings, and decisions for the CS336 language
model. Newest entries on top.

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
