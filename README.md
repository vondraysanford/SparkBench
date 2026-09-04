# ⚡ SparkBench — Benchmarks and Fine-Tuning Studies on the NVIDIA DGX Spark

[![Hardware: DGX Spark](https://img.shields.io/badge/Hardware-NVIDIA%20DGX%20Spark-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![Tracking: MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Data: DVC](https://img.shields.io/badge/Data-DVC-13ADC7)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Building in public](https://img.shields.io/badge/Status-Building%20Sep%E2%80%93Nov%202026-orange)](#build-plan)

**Benchmarks and fine-tuning studies on the NVIDIA DGX Spark. Not that Spark.**

SparkBench is the umbrella repo for everything I measure on the DGX Spark (GB10, 128 GB unified memory), in two chapters: [`bench/`](bench/) — a reproducible inference benchmarking suite (throughput, latency, memory across models and quantization levels) — and [`experiments/01-finetune-vs-frontier/`](experiments/01-finetune-vs-frontier/) — the flagship study: **can a LoRA fine-tuned 7–14B open model, running locally, replace the frontier model inside [AgentReview](https://github.com/vondraysanford/agent-review)'s Quality Agent at comparable precision/recall and near-zero marginal cost per review?**

This is the third leg of a deliberate portfolio arc: [DocQuery](https://github.com/vondraysanford/docquery) proved RAG works in the .NET ecosystem, AgentReview proved multi-agent orchestration does too, and SparkBench asks whether the most expensive component in that stack — the frontier LLM call — can be replaced with a model I trained myself, on hardware I own.

> **🚧 Status: Phase 0 — scope lock.** Building in public, ~7 working weekends, September → mid-November 2026. Nothing below is claimed as done unless its box is checked, and every number in the results table must trace to an MLflow run ID.

**The checkboxes in this README are an honesty contract. No box gets checked until its phase's exit test passes — and every exit test is binary. It passed or it didn't.**

---

## Why This Project

Fine-tuning tutorials end at a loss curve. The hard, employable skill is the whole loop: building a contamination-proof eval set, manufacturing training data in the exact inference format, iterating inside a fixed budget, and reporting results with confidence intervals against a real production baseline — then shipping the model behind the same interface the frontier model occupied.

SparkBench runs that loop against a system I already built and measured. AgentReview's Quality Agent has published numbers (17/18 planted-bug recall, 100% human-judged precision, $0.068/review with a frontier model). That makes it a rare thing: a personal project with a genuine production baseline to beat.

**The experiment cannot fail — it can only conclude differently.** All three outcomes are pre-registered and all three are publishable:

- **Headline A:** "The fine-tuned 8B matched the frontier model — my code reviews now cost $0 marginal."
- **Headline B:** "Fine-tuning closed X points of the gap — here's exactly what it bought and what it didn't."
- **Headline C:** "The untuned small model was already good enough — you don't need fine-tuning, you need a harness."

The table decides which headline ships. Not me.

---

## The Experiment

One component varies: the LLM call inside AgentReview's Quality Agent. Everything else — the Roslyn static-analysis pass, the merge/synthesis code, the findings JSON schema, the seeded PRs, the decoding settings — is frozen at one git-tagged harness commit for the duration.

| Held constant (all conditions) | Varies (the conditions) |
|---|---|
| Roslyn static analysis pass | **A:** Frontier model, prompted (current AgentReview) |
| Deterministic merge/synthesis code | **B:** Small model, prompted, untuned |
| Findings JSON schema | **C:** Small model, LoRA fine-tuned |
| Seeded PRs and planted bugs | **C-q (optional):** C, quantized for Ollama serving |
| Temperature/decoding settings | |
| Harness version (git-tagged) | |

The local model drops in behind the same provider interface the agent already calls — the identical swap pattern DocQuery was built around.

```
                 AgentReview harness (frozen @ tagged commit)
                                  │
              Roslyn pass ──► Quality Agent ──► findings JSON ──► scoring
                                  │
                        ┌─────────┴─────────┐
                        │  swappable LLM    │   ◄── the ONLY thing that varies
                        └─────────┬─────────┘
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
   A. Frontier API        B. Small model,          C. Small model,
      (baseline)             untuned, local           LoRA fine-tuned
                                  │                       │
                                  └───── DGX Spark ───────┘
                                    (train + serve, vLLM)
```

### Master Results Table (fills in as phases complete — every cell traceable to an MLflow run ID)

| Condition | Eval-18 recall | Eval-18 precision | v2-TEST recall (95% CI) | v2-TEST precision (95% CI) | Schema-valid % | Latency/review | $/review |
|---|---|---|---|---|---|---|---|
| A. Frontier, prompted | — | — | — | — | — | — | — |
| B. Small, untuned | — | — | — | — | — | — | — |
| C. Small, fine-tuned | — | — | — | — | — | — | — |
| C-q. Fine-tuned, quantized | — | — | — | — | — | — | — |

**Statistical honesty rules, set before any data exists:** Eval-18 (n=18, AgentReview's legacy benchmark) reports raw counts only — "16/18," never "88.9%," never a significance claim. v2-TEST (n≈40+, built in Phase 3) reports bootstrap 95% confidence intervals; if intervals overlap heavily, the writeup says so. Cost is reported three ways — frontier API price, local *marginal* (measured watts × my $/kWh), and local *amortized* (hardware ÷ 3-year straight line) — because "free" without a footnote is a lie.

### The Rules (frozen at Phase 0, never renegotiated)

- **One-touch test sets:** Eval-18 and v2-TEST are run against the fine-tune exactly once, in Phase 6. All iteration happens on v2-DEV.
- **Iteration budget:** maximum 5 full training runs. Budgets force decisions; unlimited retries force overfitting.
- **Schema-failure policy:** an unparseable response scores as zero findings *and* increments a separately-tracked failure counter. No manual rescues, ever.
- **Contamination firewall:** eval repos excluded from training data at the repo level, then exact-hash and near-duplicate scans, with the check outputs published in the dataset card.
- **One template:** the training-example format and the inference format are the same file, imported by both pipelines. Format drift between train and serve is the #1 silent killer of fine-tunes.

---

## Tech Stack

**Training**
- Hugging Face `transformers` + `peft` + `trl` (SFTTrainer) — first candidate stack; `torchtune`/Unsloth evaluated in the Phase 1 smoke test, winner logged with reasons
- LoRA (r=16, α=32, bf16, sequence packing) — QLoRA only if memory or wall-time forces it
- Base model: a 7–14B open-weights code-capable model (Qwen coder-class / Llama 3.1 8B are the reference points; final picks logged in the experiment card)

**Serving**
- vLLM — merged-model serving on the Spark for eval runs
- Ollama / GGUF — optional quantized deployment target (condition C-q); you evaluate what you serve

**Rigor & Tracking**
- MLflow — one experiment for the whole project, one run per training/eval execution (the same discipline as [DriftWatch](https://github.com/vondraysanford/drift-watch))
- DVC — dataset versioning; every training corpus revision recorded in the dataset card
- OpenTelemetry — latency and token counts, via AgentReview's existing instrumentation

**Harness**
- [AgentReview](https://github.com/vondraysanford/agent-review) (C#/.NET) at a git-tagged commit — the production system under test

**Hardware**
- NVIDIA DGX Spark — GB10, aarch64, 128 GB unified memory. Training and serving both happen here; the ARM + Blackwell quirks are part of the story.

---

## Build Plan

~7 working weekends, September → mid-November 2026. Hard gate between phases: Phase N+1 does not start until Phase N's binary exit test passes. Evidence (screenshots, numbers, GIFs) is captured the session it first appears, into `evidence/phase-N/`; session notes accumulate in [`BUILDLOG.md`](BUILDLOG.md) and become the blog posts.

### Phase 0 — Scope Lock & the Experiment Card

- [ ] Repo scaffolded: `bench/`, `experiments/01-finetune-vs-frontier/`, DVC + MLflow initialized
- [ ] AgentReview harness commit git-tagged — the frozen baseline for every condition
- [ ] Two candidate base models picked and justified in the experiment card
- [ ] `experiment-card.md`: hypothesis, conditions, metrics contract, three pre-registered headlines, 5-run budget, and the pre-registered success margin for the fine-tune

**Exit test:** a stranger reading the experiment card can answer — without asking me — what is being compared, on what data, by what metrics, and what each of the three outcomes looks like.

### Phase 1 — Training Environment Smoke Test on the Spark

- [ ] Training stack stood up on GB10/aarch64; winner and war stories logged
- [ ] ~100 toy examples hand-written in the exact inference chat format; 50–100 step LoRA smoke train
- [ ] Full loop proven: train → save adapter → reload → merge → serve via vLLM → one schema-valid findings JSON back
- [ ] Measured tok/s projects a full training run (~3k examples × 2–3 epochs) at under 12 hours — overnight-able

**Exit test:** loss curve decreases in MLflow, the served fine-tune returns schema-valid JSON, and a full run fits overnight. If not: drop to the 7B or switch to QLoRA and re-smoke.

### Phase 2 — Baseline the Untuned Small Model

- [ ] Local model wired in behind AgentReview's provider interface; harness untouched
- [ ] Condition A (frontier) re-run fresh at the tagged commit — published numbers re-confirmed
- [ ] Condition B (small, untuned) run on Eval-18 after one honest day of prompt effort, then frozen
- [ ] Master table rows A and B complete for Eval-18, with latency and cost from OTel

**Exit test:** two complete, traceable table rows before any fine-tuning exists. Without row B, the fine-tune can't prove it did anything. *Kill-criteria checkpoint: if B already matches A, Headline C becomes the story — noted in the BUILDLOG that day, not retroactively.*

### Phase 3 — Eval Set v2: Plant, Verify, Freeze

- [ ] Planting protocol doc written *before* any bugs are planted
- [ ] ~60 bugs planted across ~25 PRs in fresh, permissively-licensed C# repos (plus genuinely clean PRs as false-positive bait) — every bug human-verified
- [ ] Split at the PR level: v2-DEV (~20 bugs, iteration) / v2-TEST (~40 bugs, one-touch)
- [ ] Eval set git-tagged, fixtures hashed, lineage documented; conditions A and B run once on both splits with bootstrap CIs

**Exit test:** v2 is frozen and replicable from the protocol doc alone; rows A and B are complete for all v2 columns.

### Phase 4 — The Training Data Factory

- [ ] ~2,000–3,000 examples from two sources: synthetic bug injection (ground truth for free) + frontier distillation on organic diffs — all in the exact inference format
- [ ] ~25–35% clean diffs whose correct output is an empty findings list — a model trained only on bugs learns every diff must contain one
- [ ] Decontamination: repo-level exclusion, exact-hash and MinHash scans against both eval sets, results saved
- [ ] 25-example manual audit ≥90% clean; dataset DVC-versioned with a full `dataset-card.md`

**Exit test:** dataset card complete, zero eval overlap, audit passed, revision recorded.

### Phase 5 — Train and Iterate (DEV only)

- [ ] Run 1: boring-on-purpose config (LoRA r=16, α=32, lr 1–2e-4 cosine, 2–3 epochs, bf16)
- [ ] Iterate within the 5-run budget — one knob per run, in leverage order: data mix, epochs, learning rate, rank
- [ ] Every run: merge → serve → full v2-DEV eval + schema validity, logged to MLflow with config as params
- [ ] If quantized serving is a target: quantize the current best once mid-phase and eval the quantized artifact

**Exit test (passes on either branch):** best checkpoint beats row B on v2-DEV by the pre-registered margin at ≥95% schema validity → Headline A/B track; *or* the budget is exhausted, the best checkpoint is frozen anyway, and the gap gets one honest paragraph → Headline B. The gate enforces discipline, not a result — both branches ship.

### Phase 6 — Final Evaluation: One Pass, Then Freeze

- [ ] Winning checkpoint tagged; each condition run exactly once on Eval-18 and v2-TEST
- [ ] Master table 100% complete — bootstrap CIs on v2-TEST, raw counts on Eval-18, every cell footnoted with its run ID
- [ ] **Per-miss autopsy:** one honest paragraph for every false negative and false positive from the fine-tune
- [ ] The headline the table actually supports selected; side-by-side GIF recorded (same PR, frontier vs. my fine-tune)

**Exit test:** the money table, produced once, defensible forever.

### Phase 7 — Ship Everything

- [ ] Repo public; experiment README opens with the master table and a "reproduce the eval in 30 minutes" section
- [ ] Adapter + model card on Hugging Face (license permitting), with the results table and an honest limitations section
- [ ] `bench/` chapter filled with months of real training/serving throughput, memory, and latency numbers as reproducible scripts
- [ ] Three posts assembled from the BUILDLOG: data + decontamination, training-on-Spark war stories, and the results post
- [ ] *Stretch:* a frontier-vs-fine-tune model toggle on the AgentReview live demo

**Exit test:** the stranger test — a clean machine reproduces the headline eval from the README in under 30 minutes without contacting me.

---

## Repository Layout (target)

```
sparkbench/
├── bench/                                # Chapter 1: inference benchmarks on the Spark
│   ├── scripts/                          #   reproducible throughput/latency/memory runs
│   └── results/                          #   measured numbers, per model × quantization
├── experiments/
│   └── 01-finetune-vs-frontier/          # Chapter 2: the flagship study
│       ├── experiment-card.md            #   hypothesis, conditions, metrics, headlines
│       ├── dataset-card.md               #   sources, licenses, contamination checks, audit
│       ├── data/                         #   DVC-versioned training corpus
│       ├── training/                     #   LoRA configs + run scripts
│       ├── eval/                         #   Eval-18 + v2 fixtures, scoring, autopsies
│       └── evidence/phase-N/             #   dated screenshots, numbers, GIFs
├── BUILDLOG.md                           # session-by-session: what worked, broke, surprised
└── README.md
```

## Related

- **[AgentReview](https://github.com/vondraysanford/agent-review)** — the multi-agent code-review system whose Quality Agent this experiment targets; source of the frozen harness and the baseline numbers
- **[DocQuery](https://github.com/vondraysanford/docquery)** — origin of the swappable-provider pattern that makes the model swap a config change
- **[DriftWatch](https://github.com/vondraysanford/drift-watch)** — origin of the MLflow run-tracking discipline every number here inherits
- Build posts land at [vondraysanford.com](https://vondraysanford.com) as each phase ships

## License

[MIT](LICENSE)

---

**Built by [Vondray Sanford](https://www.linkedin.com/in/vondray-sanford/)** — .NET engineer building at the intersection of enterprise systems and modern AI.
