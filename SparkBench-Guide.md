# SparkBench — A-to-Z Build Guide

**Working hypothesis (one sentence):** A LoRA fine-tuned 7–14B open model, running locally on a DGX Spark under vLLM, can replace the frontier model inside AgentReview's Quality Agent at comparable precision/recall and near-zero marginal cost per review.

**The name & shape:** `SparkBench` is the umbrella repo for everything measured on two DGX Sparks (GB10, 128 GB unified memory each, cabled together over ConnectX-7 200 GbE), in two chapters: `bench/` — the inference benchmarking suite your site already promises (throughput, latency, memory across models and quantization levels) — and `experiments/01-finetune-vs-frontier/` — this guide's study, the flagship. One repo, one portfolio card, and it retires the site's "coming soon" placeholder the day it goes public. Because the wider world associates "SparkBench" with Apache Spark benchmarking tooling, the disambiguation tagline travels with the name everywhere it appears: **"SparkBench — benchmarks and fine-tuning studies on the NVIDIA DGX Spark. Not that Spark."**

**Timebox:** ~7 working weekends, September → mid-November 2026. Done before OMSCS starts.

---

## 0. Operating Rules (read before every session)

1. **Hard gate:** Do not start Phase N+1 until Phase N's exit test passes. Every exit test below is written to be binary — it passed or it didn't. No "mostly done."
2. **Evidence at the moment of creation:** The screenshot/number/GIF gets captured in the same terminal session it first appears, into `evidence/phase-N/`, with the naming convention in §11. If you tell yourself "I'll grab it later," you won't, and the writeup will be worse for it. Minimum: **one shareable artifact per session** — §11.2 lists the moments worth catching per phase, so start recording *before* the interesting command, not after.
3. **One-touch test set:** The final test sets (Eval-18 and v2-TEST) are run against the fine-tuned model **exactly once**, in Phase 6. All iteration happens on v2-DEV. The moment you re-run the test set "just to check," your headline numbers are contaminated.
4. **Iteration budget:** Maximum 5 full training runs in Phase 5. Budgets force decisions; unlimited retries force overfitting.
5. **Every run is logged:** One MLflow experiment for the whole project, one run per training/eval execution. Any number that appears in the final table must trace to a run ID. (Same MLflow discipline as DriftWatch — one consistent story across your portfolio.)
6. **Log lines:** End every working session by adding 3 bullets to `BUILDLOG.md` — what worked, what broke, what surprised you — plus one line naming the day's capture that is post-worthy. These become the blog posts. Write them while the pain is fresh.
7. **Pre-registered outcomes:** All three possible results are publishable. The project cannot fail; it can only conclude differently:
   - **Headline A:** "The fine-tuned 8B matched the frontier model — my code reviews now cost $0 marginal."
   - **Headline B:** "Fine-tuning closed X points of the gap — here's exactly what it bought and what it didn't."
   - **Headline C:** "The untuned small model was already good enough — you don't need fine-tuning, you need a harness."

---

## 1. What's Held Constant vs. What Varies

The entire experiment swaps **one component**: the LLM call inside the Quality Agent. Everything else is frozen at one harness commit for the duration.

| Held constant (all conditions) | Varies (the conditions) |
|---|---|
| Roslyn static analysis pass | **A:** Frontier model, prompted (current AgentReview) |
| Deterministic merge/synthesis code | **B:** Small model, prompted, untuned, served by vLLM |
| Findings JSON schema | **C:** Small model, LoRA fine-tuned, merged, served by vLLM |
| Seeded PRs and planted bugs | **C-q (optional):** C, quantized to FP8 or NVFP4 with NVIDIA Model Optimizer, served by vLLM |
| Temperature/decoding settings | **D (optional, pre-registered):** ~70B-class open model, untuned, served by vLLM across both Sparks (tensor parallel 2) |
| Harness version (git tag it in Phase 0) | |
| Serving engine: vLLM, one pinned NGC container tag | |

**Scope decision:** Quality Agent only. It carries the headline metric and the hardest job. Do not fine-tune all three agents — that triples data work for zero additional story. (If Phase 2 reveals the quality lane is hopeless for a small model, the documented fallback is the Docs Agent — but decide that at the Phase 2 gate, not mid-phase.)

**Why D exists, and why it's optional:** the second Spark makes a new question askable — on this exact hardware, does fine-tuning an 8B beat simply running a much larger untuned model with both nodes? Pre-registering D now means it can't be bolted on later to rescue a story. It runs once, in Phase 6, only if Phases 0–5 land on schedule. It is the first thing cut, and cutting it is logged in `BUILDLOG.md`, not silently. D does not get its own headline; it adds one row that sharpens whichever of A/B/C the table supports. The exact model and precision for D are chosen in Phase 0 from NVIDIA's vLLM-on-Spark support matrix and written into the experiment card.

### 1.1 Two Sparks, two roles

| Node | Role | Container | What runs there |
|---|---|---|---|
| **Spark 1** | Train | `nvcr.io/nvidia/pytorch:<pinned>` | LoRA smoke tests, all 5 budgeted training runs, adapter merge, quantization jobs |
| **Spark 2** | Serve + eval | `nvcr.io/nvidia/vllm:<pinned>` | vLLM endpoint for B, C, C-q; every DEV/TEST eval; AgentReview points here |
| **Both** | Cluster (only for D and `bench/dual-spark/`) | vLLM Ray head on Spark 1, worker on Spark 2 | Tensor-parallel-2 serving of the large model; NCCL and RDMA link benchmarks |

The split buys one concrete thing: **Phase 5 iteration overlaps.** Run N+1 trains on Spark 1 while run N's DEV eval runs on Spark 2. With one machine those serialize. It also keeps the serving node's memory clean — a training job's page cache never competes with vLLM's KV cache (NVIDIA's playbooks warn that unified memory on Spark can OOM below nominal capacity for exactly this reason).

What the link is and isn't: the QSFP connection is 200 Gb/s Ethernet between nodes; local memory is 273 GB/s per node. NVIDIA's own framing is that two Sparks give **256 GB of combined memory for one model**, not a faster single model. Distributed FSDP fine-tuning across both nodes is documented by NVIDIA for 8B and 70B, but for a 7–14B LoRA that already fits in one node's memory, it is a `bench/` experiment to measure, not a critical-path assumption. Nothing in Phases 0–5 requires the cluster link.

The same username must exist on both nodes for NVIDIA's cluster tooling to work (both the `discover-sparks` script and NVIDIA Sync's Cluster Assistant check this). **Done:** `admin` (UID/GID 1000) on both, verified by the Cluster Assistant on 2026-09-06.

**Status, 2026-09-06:** the cluster is bridged. NVIDIA Sync's Cluster Assistant created the CX-7 network between `spark-927a.local` (`10.100.240.1`, `10.100.241.1`) and `spark-11ed.local` (`10.100.240.2`, `10.100.241.2`), measured **185.96 Gbps** against its > 180 Gbps threshold, and set up passwordless SSH. Cluster name `Lucia`. Everything recorded, including what is still to validate, lives in `infra/cluster.md`; the eight screenshots are in `evidence/phase-1/`.

### 1.2 One serving engine: vLLM everywhere

Every local condition is served by vLLM from the NGC container, one tag, pinned in `infra/containers.md` for the life of the experiment. Reasons, in order:

1. **No engine confound.** If B, C, and C-q ran on different engines, an engine difference (sampling implementation, chat-template handling, quantization kernels) could masquerade as a model difference. One engine removes that variable entirely.
2. **One provider in AgentReview.** vLLM's OpenAI-compatible API means every local condition is the same C# provider with a different base URL and model name. The harness stays untouched across conditions.
3. **NVIDIA validates it on this GPU.** The vLLM playbook publishes a model support matrix for DGX Spark with FP8 and NVFP4 variants of Qwen3-8B/14B and Llama 3.1 8B, and the NVFP4 quantization playbook runs Model Optimizer *inside the same vLLM container*. The quantized path is a documented workflow, not an adventure.
4. **Built-in benchmarking.** `vllm bench throughput` and `vllm bench serve` are what NVIDIA's own DGX Spark performance guide uses, so `bench/` numbers are directly comparable to NVIDIA's.

**The artifact evaluated is the artifact served.** Condition C is the merged bf16 checkpoint. C-q is the exported quantized checkpoint. vLLM can also serve an unmerged LoRA adapter on top of the base (`--enable-lora`), which is handy for fast Phase 5 spot checks — but any number that enters the master table comes from serving the frozen artifact in its final form.

---

## 2. The Metrics Contract

Primary metrics, defined once, never renegotiated after Phase 0:

- **Recall:** planted bugs found / planted bugs present (report raw counts alongside %, always).
- **Precision:** true findings / all findings, human-judged with the same scoring worksheet you built for AgentReview Phase 5.
- **Schema validity:** % of responses that parse against the findings schema on first try. A parse failure counts as a miss for recall *and* is tracked separately.
- **Latency:** wall-clock seconds per review, from OpenTelemetry.
- **Cost per review:** frontier = tokens × current API pricing (from OTel token counts). Local = two honest lines: *marginal* (measured watts × hours × your $/kWh) and *amortized* (hardware cost ÷ 3-year straight line ÷ reviews). Report both; never claim "free" without the footnote. Amortize only the Spark(s) the condition actually ran on: one unit for B, C, C-q; both units for D. Measure watts at the wall per node — the Spark ships with a 240 W supply and a 140 W GPU budget, so the marginal number will be small and should be shown, not rounded to zero.

### Master results table (fill incrementally; this exact table tops the final README)

| Condition | Eval-18 recall | Eval-18 precision | v2-TEST recall (95% CI) | v2-TEST precision (95% CI) | Schema-valid % | Latency/review | $/review |
|---|---|---|---|---|---|---|---|
| A. Frontier, prompted | | | | | | | |
| B. Small, untuned | | | | | | | |
| C. Small, fine-tuned | | | | | | | |
| C-q. Fine-tuned, quantized (vLLM FP8/NVFP4) | | | | | | | |
| D. Large, untuned (vLLM TP=2, optional) | | | | | | | |

Statistical honesty rules: on Eval-18 (n=18) report counts only — "16/18," never "88.9%," and never claim significance. On v2-TEST (n≈40+) report bootstrap 95% confidence intervals. If intervals overlap heavily, the writeup says so.

---

## 3. Phase 0 — Scope Lock & the Experiment Card

*Half a weekend (Saturday morning).*

**Goal:** Freeze every decision that could be quietly renegotiated later.

**Tasks**
1. Create the `sparkbench` repo; scaffold: root `README.md` (umbrella pitch + disambiguation tagline), `BUILDLOG.md`, `bench/`, `infra/`, and `experiments/01-finetune-vs-frontier/` containing `experiment-card.md`, `evidence/`, `data/`, `training/`, `quantize/`, `eval/`. Init DVC for data versioning, MLflow for run tracking. Decide Hugging Face naming now: `sparkbench-quality-agent-<basemodel>` for the adapter.
2. Git-tag the AgentReview harness commit that all conditions will run against.
3. Pick two candidate base models. Criteria: 7–14B parameters, strong on code, license compatible with publishing weights (Apache-2.0 preferred — matters if you push the adapter to Hugging Face), **and present in NVIDIA's [vLLM-on-Spark model support matrix](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md#model-support-matrix) with published FP8 and NVFP4 variants** — that is the evidence the quantized path (C-q) already works on this GPU. As of writing, Qwen3-8B, Qwen3-14B, and Llama 3.1 8B all qualify — **but verify the current small-model landscape the week you start; it moves monthly.** Log the two picks and the reason in the experiment card.
4. Pin the containers. Pick the current `nvcr.io/nvidia/vllm` and `nvcr.io/nvidia/pytorch` tags from NGC and write them into `infra/containers.md` with the date and the NGC release-notes link. These tags do not change until Phase 7 ships, even if NGC publishes a newer one mid-project. (NVIDIA's NVFP4 playbook requires vLLM container 26.05 or newer; the vLLM playbook's two-Spark recipe is pinned against 26.05-py3. Start from the newest tag NGC lists and note any DGX Spark known issues in its release notes — the 26.08 notes, for example, call out unified-memory OOMs and recommend lowering `--gpu-memory-utilization`.)
5. Decide C-q and D scope now, in writing. For D, name the exact model and precision (from NVIDIA's support matrix — Llama 3.3 70B is the model NVIDIA's own two-Spark vLLM recipe demonstrates) and the cut rule: D is dropped if Phase 5 has not exited by its calendar weekend.
6. Write `experiment-card.md`: the hypothesis, the conditions table (§1), the two-Spark role assignment (§1.1), the metrics contract (§2), the three pre-registered headlines, the iteration budget (5 runs), and the **pre-registered success margin** for Phase 5 (recommended: fine-tuned must beat untuned by ≥10 points of v2-DEV recall at no precision loss, with schema validity ≥95%).
7. Draw the empty master results table into the README.

**Exit test (binary):** A stranger reading `experiment-card.md` can answer, without asking you: what is being compared, on what data, by what metrics, and what each of the three outcomes would look like. The empty results table exists with every row and column named. Harness commit is tagged. Container tags are pinned.

**Evidence to capture:** the experiment card itself; screenshot of the repo tree; the tagged harness commit hash and the pinned container tags in `BUILDLOG.md`.

**If it stalls:** you're over-deliberating the model choice. Pick the two most boring defensible candidates and move — Phase 1's smoke test is allowed to eliminate one.

---

## 4. Phase 1 — Training Environment Smoke Test on the Sparks

*One weekend.*

**Goal:** Prove the full train → save → reload → merge → serve loop works across the two nodes before any real data exists. Environment quirks on ARM + Blackwell are exactly the kind of thing you verify, not assume — and exactly the war stories your blog runs on. NVIDIA's playbooks are the starting point for every step; deviations from them are the first BUILDLOG entries.

**Tasks**
1. Stand up the training stack on **Spark 1** inside the NGC PyTorch container, following NVIDIA's [Fine-Tune with PyTorch](https://build.nvidia.com/spark/pytorch-fine-tune) playbook (Hugging Face `transformers` + `peft` + `trl` SFTTrainer; NVIDIA's reference 8B LoRA script targets all seven projection modules). Copy NVIDIA's version pins from the performance guide rather than resolving your own — it pins `trl` and `bitsandbytes` and sets `BNB_CUDA_VERSION=130` for the container's CUDA. Fallbacks, in order: NVIDIA's [Unsloth](https://build.nvidia.com/spark/unsloth) playbook, then [NeMo AutoModel](https://build.nvidia.com/spark/nemo-fine-tune). Log which stack won and why.
2. Stand up vLLM on **Spark 2** from the pinned NGC container, following the [vLLM for Inference](https://build.nvidia.com/spark/vllm) playbook. Serve the untuned base model, wait for `/health`, and round-trip one chat completion. Record the `vllm serve` flags that worked (`--max-model-len` sized to a real review prompt, `--gpu-memory-utilization`) in `infra/serve/` — these are frozen decoding/serving settings from here on.
3. Pull both candidate base models. Confirm bf16 LoRA fits comfortably in unified memory for the 14B on Spark 1 (it should, with ~128 GB — record actual peak usage). If a load fails below nominal capacity, flush the page cache first (NVIDIA's documented fix: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`) and log it — unified memory surprises are a war story, not a blocker.
4. Hand-write ~100 toy training examples in the **exact** chat format the Quality Agent uses at inference time — same system prompt structure, diff + static-analysis context in, findings-schema JSON out. (Rule for the whole project: training format = inference format, byte-for-byte template match. Format drift between train and serve is the #1 silent killer of fine-tunes.)
5. Run a 50–100 step LoRA smoke train. Log to MLflow.
6. Save the adapter, reload it, merge it, copy the merged checkpoint to Spark 2, serve it through vLLM, and send it one review request.
7. Measure training throughput (tokens/sec) and project the full-run wall time for ~3k examples × 2–3 epochs.
8. **Non-gating — done 2026-09-06, ahead of order:** the cluster link is already bridged. NVIDIA Sync's [Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) verified SSH, GB10, OS, and sudo on both nodes, confirmed `admin` (UID/GID 1000) on both, created the CX-7 network (`10.100.240.0/24` and `10.100.241.0/24`, 200 Gbps per link), measured **185.96 Gbps** against its > 180 Gbps threshold, and distributed SSH keys. Cluster name: `Lucia`. Facts and remaining to-dos are in `infra/cluster.md`; the eight screenshots are in `evidence/phase-1/`. What is left, still non-gating: run `ibdev2netdev` on each node and record which logical interface carries which cluster IP (the vLLM two-Spark recipe needs it for `NCCL_SOCKET_IFNAME` and friends), then the [NCCL playbook](https://build.nvidia.com/spark/nccl)'s `all_gather_perf` and the performance guide's `ib_write_bw` — NVIDIA's guide shows ~190 Gbps aggregate across the two RDMA links; record what yours does. Those numbers are the first entries in `bench/results/`. Record the `ib_write_bw` counter ticking while you're there (§11.2). Do not re-apply the Connect Two Sparks playbook's netplan example: it uses different subnets than the ones the assistant chose.

**Exit test (binary):** (a) smoke-run loss curve decreases in MLflow; (b) the saved adapter reloads and the served model on Spark 2 returns at least one schema-valid findings JSON; (c) measured tok/s projects a full training run at **under 12 hours** — overnight-able. If (c) fails: drop to the 7B, or switch to QLoRA, and re-run the smoke test until it passes. The cluster link (task 8) is deliberately outside the gate: nothing on the critical path needs it before Phase 6.

**Evidence to capture:** MLflow loss-curve screenshot; peak memory reading on Spark 1; vLLM startup log on Spark 2 showing the model and memory allocation; terminal log of the first schema-valid generation from your own fine-tuned (toy) model — that moment is the money screenshot of the whole project; the tok/s number and projected full-run hours in `BUILDLOG.md`; the `ibdev2netdev` output and NCCL / RDMA numbers once task 8's remainder is run. Already captured: the Cluster Assistant sequence and the two-node cluster view (Sep 6–7).

---

## 5. Phase 2 — Baseline the Untuned Small Model

*One weekend.*

**Goal:** Two complete rows of the master table before any fine-tuning exists. Without row B, you cannot prove the fine-tune did anything.

**Tasks**
1. Add the local model as a provider behind the same interface the Quality Agent already calls — the identical pattern to DocQuery's provider swap. Concretely: an OpenAI-compatible chat-completions provider whose base URL is vLLM on Spark 2 and whose `model` field is the served model name. That one provider serves every local condition for the rest of the project. Roslyn, merge logic, schema, and decoding settings untouched.
2. Define the schema-failure policy in code: a response that fails to parse = zero findings for recall purposes + increments the schema-failure counter. No manual rescues.
3. Run **condition A** (frontier) fresh on the original 8 PRs / 18 bugs at the tagged harness commit — re-confirm your published numbers still hold on today's harness. If they don't, stop and reconcile before anything else; your baseline must be real.
4. Run **condition B** (small, untuned, prompted) on Eval-18. Give the prompt one honest day of effort — few-shot examples, tightened instructions — so the baseline is fair, then freeze it.
5. Pull latency and cost from OTel for both rows.

**Exit test (binary):** Master table rows A and B are complete for the Eval-18 columns, both produced by the tagged harness commit, each cell traceable to an MLflow run ID, schema-validity recorded.

**Evidence to capture:** eval output screenshots for both conditions; one side-by-side of the same PR reviewed by A vs B (this becomes a GIF later); the cost/latency numbers.

**Decision gate (kill-criteria checkpoint):** If row B already matches row A, Headline C just became your story. You still proceed — fine-tuning now answers "how much headroom is left?" — but the framing flips, and you note the flip in `BUILDLOG.md` today, not retroactively.

---

## 6. Phase 3 — Eval Set v2: Plant, Verify, Freeze

*One weekend.*

**Goal:** An evaluation set large enough to mean something. With n=18, one bug is 5.5 points of recall — you cannot distinguish two decent models on it. Eval-18 stays as the legacy benchmark (continuity with AgentReview's published numbers); v2 is where statistical claims live.

**Tasks**
1. Select fresh source repos: permissively licensed C# projects that appear **nowhere** in AgentReview's original evals and will be excluded from all training data. Record the repo list — this is your contamination firewall, at the repo level.
2. Write the planting protocol doc *first* (bug taxonomy, injection method, verification step) — the same "protocol before data" discipline as your Phase 5 post.
3. Plant ~60 bugs across ~25 PRs. LLM-assisted injection is fine for speed; **every single bug is human-verified** by you against the taxonomy — the 45-manual-judgments standard. Include genuinely clean PRs too (false-positive bait).
4. Balance the taxonomy across the Quality Agent's lanes (null-handling, resource leaks, async misuse, off-by-ones, etc. — whatever your existing taxonomy says).
5. Split: **v2-DEV** ≈ 20 bugs (iteration set), **v2-TEST** ≈ 40 bugs (one-touch set). Split at the PR level, not the bug level.
6. Freeze: git-tag the eval set, hash the fixtures, write the lineage note (which repos, which commit, who verified).
7. Run conditions A and B once each on v2-DEV and v2-TEST. (Baselines are frozen conditions — they get their one touch now. The one-touch rule exists to stop *iteration* against TEST, and nothing iterates on A or B.)

**Exit test (binary):** v2 is tagged and hashed; the planting protocol doc is complete enough that a stranger could replicate the set; taxonomy distribution table exists; master table rows A and B are complete for all v2 columns, with bootstrap CIs computed.

**Evidence to capture:** taxonomy distribution table; one before/after diff of a planted bug; the updated master table; the tag + hash in `BUILDLOG.md`.

---

## 7. Phase 4 — The Training Data Factory

*One weekend.*

**Goal:** ~2,000–3,000 training examples in the exact inference format, decontaminated, versioned, and audited.

**Tasks**
1. **Source 1 — synthetic injection (ground truth for free):** run your Phase-3 injection tooling over a *different* set of permissively licensed C# repos (disjoint from both eval sets) to produce diff → known-findings pairs.
2. **Source 2 — frontier distillation:** collect real diffs from those same training-side repos and label them with the current frontier Quality Agent. This teaches the small model the frontier's judgment on organic code, not just planted patterns.
3. **Teach silence:** include a healthy fraction (~25–35%) of clean diffs whose correct output is an empty findings list. This is what protects precision — a model trained only on buggy diffs learns that every diff must contain a bug.
4. **Decontaminate:** repo-level exclusion of every eval repo; then exact-match and near-duplicate checks (hashing + MinHash or similar) of all eval diffs against the training corpus. The frontier labeler must never process an eval PR. Save the check outputs — they go in the dataset card.
5. **License hygiene:** training-source repos must be permissively licensed if the dataset or weights go public. Record licenses per repo.
6. **Audit:** sample 25 random examples; manually judge label quality against the taxonomy. Pass bar: ≥90% clean. Below that, fix the pipeline and re-audit — do not hand-fix individual examples.
7. Version the dataset with DVC; write `dataset-card.md` (counts by source and category, format spec, licenses, contamination-check results, audit score).

**Exit test (binary):** dataset card committed with all sections filled; contamination checks show zero eval overlap; audit ≥90%; dataset DVC-versioned at a recorded revision.

**Evidence to capture:** the dataset card; the audit spreadsheet; a category-distribution chart; the DVC revision in `BUILDLOG.md`.

---

## 8. Phase 5 — Train and Iterate (DEV only)

*One weekend, plus overnight weeknight runs as needed.*

**Goal:** The best fine-tune you can get inside a 5-run budget, measured only on v2-DEV.

**Tasks**
1. **Run 1 starting config:** LoRA r=16, α=32, lr 1–2e-4 (cosine), 2–3 epochs, bf16, sequence packing on, effective batch sized to fit measured memory from Phase 1. Boring on purpose.
2. After each run: merge on Spark 1 → copy to Spark 2 → serve via vLLM → full v2-DEV eval + schema-validity + Eval-18-style spot check. Log everything to MLflow with the config as params. **Use both nodes:** kick off run N+1 on Spark 1 the moment run N's merge is copied over; Spark 2 evaluates run N while Spark 1 trains. Overnight weeknight runs land on Spark 1, and Spark 2 is free to eval them the next morning.
3. Iterate within budget. Highest-leverage knobs, in order: data mix (buggy:clean ratio), epochs (watch for the precision cliff from overfitting), learning rate, rank. Change one thing per run.
4. If C-q is in scope, quantize the current-best merged checkpoint once mid-phase and eval **the quantized artifact** on DEV. FP8 first (smallest accuracy risk, roughly halves weight memory). NVFP4 second if time allows — it is Blackwell-native and NVIDIA quotes ~3.5× memory reduction versus FP16 with typically under 1% accuracy loss, but "typically" is exactly the kind of claim this project exists to measure. Produce the checkpoint with NVIDIA Model Optimizer's `hf_ptq.py` recipes inside the pinned vLLM container, exactly as the [NVFP4 Quantization](https://build.nvidia.com/spark/nvfp4-quantization) playbook does, and serve the exported checkpoint with the same container. Recipes and the calibration dataset go in `quantize/`. You evaluate what you serve — quantization moves numbers.
5. Log every run's DEV results in a small runs table in `BUILDLOG.md` as they finish, not at the end of the day.

**Exit test (binary — passes on either branch, provided the budget was respected and the decision is logged):**
- **Branch 1:** best checkpoint beats row B on v2-DEV by the pre-registered margin from Phase 0, with schema validity ≥95% → proceed toward Headline A or B.
- **Branch 2:** budget of 5 runs exhausted without hitting the margin → freeze the best checkpoint anyway, write one paragraph on the gap and your best hypothesis why, proceed under Headline B.
The gate enforces discipline, not a result — both branches ship.

**Evidence to capture:** MLflow run-comparison screenshot across all runs; best run's loss curve; the DEV runs table; if quantized, the bf16-vs-quantized DEV delta.

---

## 9. Phase 6 — Final Evaluation: One Pass, Then Freeze

*Half a weekend.*

**Goal:** The money table, produced once, defensible forever.

**Tasks**
1. Freeze the winning checkpoint (tag it; record adapter + merge hashes).
2. Run each condition **once**: A, B (already done in Phases 2–3 — reuse those runs; do not re-roll), C, and C-q if serving quantized — on Eval-18 and v2-TEST.
   - If D is still in scope: join both Sparks as a vLLM Ray cluster per the "Run on two Sparks" section of NVIDIA's [vLLM playbook](https://build.nvidia.com/spark/vllm) (head on Spark 1, worker on Spark 2, all traffic bound to the QSFP interface via `NCCL_SOCKET_IFNAME` and friends), serve the pre-registered model with `--tensor-parallel-size 2 --distributed-executor-backend ray`, raise `--max-model-len` from the playbook's 2048 demo value to what a real review prompt needs, and run D once on both splits through the same AgentReview provider. Tear the cluster down afterwards so Spark 2 goes back to single-node serving for the GIF.
3. Compute bootstrap 95% CIs for the v2-TEST cells. Raw counts for Eval-18 cells.
4. Fill the master table completely. Every cell gets its MLflow run ID in a footnote file.
5. **Per-miss autopsy:** every false negative and every false positive from condition C gets one honest paragraph — what it was, why the model likely missed or invented it. This is your signature move; it's also where the best blog material lives.
6. Pick the headline (A/B/C from §0.7) that the table actually supports. Not the one you wanted.
7. Record the side-by-side GIF: the same PR reviewed by the frontier model and by your fine-tune, terminal or UI, 20–30 seconds.

**Exit test (binary):** master table 100% complete and traceable; autopsy written for every miss and false positive; headline selected and one supporting paragraph drafted; GIF exists in `evidence/phase-6/`.

**Evidence to capture:** the final table (screenshot + markdown); latency/cost comparison chart; the GIF; the autopsy doc.

---

## 10. Phase 7 — Ship Everything

*One weekend.*

**Goal:** Public, reproducible, and linked from every surface.

**Tasks**
1. **Repo public:** root README = the umbrella pitch, the disambiguation tagline, and both chapters linked. The experiment's README opens with the master table, then the 3-sentence story, then a "reproduce the eval in 30 minutes" section (pinned deps, one script, fixture download). Experiment card, dataset card, and autopsies linked.
2. **Model card:** publish the adapter (and merged weights if the base license allows) to Hugging Face with the results table and the honest limitations section.
3. **Fill the `bench/` chapter:** you now have months of real throughput/memory/latency numbers from this hardware across training and serving. Write them up as reproducible benchmark scripts plus a results page — the inference-suite promise from your site, fulfilled inside the same repo. Use the methodology from NVIDIA's [DGX Spark performance guide](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md) so the numbers are comparable to NVIDIA's own: `vllm bench throughput` for offline, `vllm bench serve` with `--percentile-metrics ttft,tpot,itl,e2el` for online, fixed input/output lengths. Two sections:
   - `bench/scripts/single-spark/`: the base model and the fine-tune at bf16, FP8, and NVFP4 — throughput, TTFT, TPOT, peak memory.
   - `bench/scripts/dual-spark/`: measured link bandwidth (NCCL `all_gather_perf`, `ib_write_bw`, `ib_write_lat`); the 14B at TP=1 on one node versus TP=2 across both; the ~70B model that only fits with both nodes. The honest question this section answers: *what does the second Spark actually buy at inference time on a 200 GbE link?*
4. **Site + GitHub:** the site's existing SparkBench card finally gets its link and absorbs the fine-tuning story into its description; retire the "Up next — DGX Spark benchmarks" line in the profile README; update llms.txt; consider pinning `sparkbench` in place of the Twitter bot.
5. **Writeups from BUILDLOG.md** (already written in pieces — assemble, don't compose): at minimum (i) the data + decontamination post, (ii) the training-on-Spark war-stories post, (iii) the results post led by the table. Submit each to the .NET aggregators/newsletters; the results post is the Show HN / r/LocalLLaMA candidate.
6. **Stretch, only if time remains:** a model toggle on the AgentReview live demo (frontier vs. your fine-tune) — a public artifact almost nobody else has.

**Exit test (binary):** the stranger test — a clean machine (or a friend) reproduces the headline eval from the README in under 30 minutes without contacting you; results post published; zero "coming soon" promises remain on your site or profile that this project was supposed to fulfill.

**Evidence to capture:** public links; README screenshot; first external engagement (comment, star, question) whenever it lands.

---

## 11. Evidence & Content Capture

Two audiences read this project: the stranger who reproduces it, and the scroller who decides in three seconds whether to click. §11.1 serves the first; §11.2–11.4 serve the second. Same artifacts, filed once.

### 11.1 Conventions

```
experiments/01-finetune-vs-frontier/evidence/
  phase-0/  2026-09-05-experiment-card.png
  phase-1/  2026-09-06-cluster-assistant-01-select-devices.png        ← captured
            …  (02 access, 03 user details, 04 network plan, 05 configured)
            2026-09-06-cluster-assistant-06-bandwidth-185-96-gbps.png  ← captured
            2026-09-06-cluster-assistant-07-ssh-setup-complete.png     ← captured
            2026-09-07-nvidia-sync-cluster-lucia-2-nodes-online.png    ← captured
            2026-09-13-first-schema-valid-generation.png
            2026-09-13-smoke-train-progress.gif
            raw/                                          .mov / .cast / .mp4 — gitignored
  phase-6/  2026-10-18-side-by-side-review.gif
            2026-10-18-ray-status-two-nodes.png           (only if D ran)
```

- Name: `YYYY-MM-DD-what-it-shows` — no `img_4021.png`, no `final_v2_REAL`. Sequences get `-01-`, `-02-`.
- Screenshots: full terminal context visible (command + output), not cropped fragments.
- Numbers: also land in `BUILDLOG.md` as text the moment they exist — screenshots rot, text greps.
- GIFs: capture at the moment of first success; 20–30 s; terminal-native tools are fine.
- Raw recordings (`.mov`, `.cast`) and the social MP4s go in `evidence/phase-N/raw/`, which is gitignored. The GIF is the committed derivative; the raw file is what you re-cut from later.

### 11.2 The capture calendar — what to shoot, when, and why it travels

Legend: 📷 photo (a phone is fine) · 🖼 screenshot · 🎞 recording → GIF/MP4. **Bold** = the one that matters most that phase.

| Phase | Moment | Medium | Why it travels |
|---|---|---|---|
| 0 | **Both Sparks stacked, QSFP cable visible, dark backdrop** — shoot it once, well; it becomes the README banner and every post's cover | 📷 | The hardware is the hook. Nobody else's portfolio has this photo. |
| 0 | Cable close-up; the experiment card open in the editor; the empty master table | 📷 🖼 | "Here is the contract before any data exists" is a rare and credible post. |
| 1 | **NVIDIA Sync cluster view with both nodes busy** — Spark 1's GPU pegged by the smoke train, Spark 2 holding vLLM's memory | 🖼 | The visual thesis of the whole project: two boxes, two jobs. |
| 1 | Smoke train progress bar with tok/s ticking (asciinema, 20 s, 2× speed) | 🎞 | Training on ARM + Blackwell at home is still novel; the number in the bar is the story. |
| 1 | vLLM booting on Spark 2, scrolling to "Application startup complete" | 🎞 | Proves the serving side is real, not a slide. |
| 1 | **First schema-valid findings JSON from your own toy fine-tune** — the request, the pause, the response | 🎞 + 🖼 | The money moment. Record it, don't just screenshot it — the pause before the JSON appears is the drama. |
| 1 | MLflow loss curve; peak memory reading; `nvidia-smi` on both nodes in a tmux split | 🖼 | Proof-of-work images for the war-stories post. |
| 1 | NCCL `all_gather_perf` output; the `ib_write_bw` counter running for 10 s | 🖼 🎞 | The first `bench/` numbers; ~90 Gbps per link ticking by is oddly satisfying. |
| 2 | The one-line config diff that points AgentReview at Spark 2 | 🖼 | "The swap is a config change" is the DocQuery thesis, proven again. |
| 2 | The same PR reviewed by A and by B, recorded separately at the **same terminal size and font** | 🎞 | These become panes in the Phase 6 side-by-side. Record them in the final layout now. |
| 2 | Rows A and B filled in; the OTel latency histogram; the three cost lines | 🖼 | The first table post: "here is the gap fine-tuning has to close." |
| 3 | Taxonomy distribution chart; one planted bug as a before/after diff | 🖼 | Shows the eval set is built, not scraped. |
| 3 | 30 s of the human-verification pass over one planted bug | 🎞 | Rigor on camera. Optional, high trust per second. |
| 4 | Decontamination output showing zero overlap; category distribution; the audit sheet | 🖼 | The data post's evidence. |
| 4 | The data factory running across repos, 4× speed | 🎞 | Pipelines look like work. Twenty seconds is enough. |
| 5 | **MLflow run comparison across all five budgeted runs** | 🖼 | The discipline is the story: five runs, one knob each, done. |
| 5 | **Both nodes busy in NVIDIA Sync** — Spark 1 on run N+1, Spark 2 evaluating run N — a screenshot and a 20 s clip of the trend lines | 🖼 🎞 | The two-Spark payoff, visible. |
| 5 | Next-morning NVIDIA Sync "All time" trend after an overnight run | 🖼 | A time-lapse without a time-lapse. |
| 5 | Model Optimizer quantization finishing; `hf_quant_config.json` appearing; the bf16 vs quantized DEV delta | 🖼 | The C-q post. |
| 6 | **Side-by-side: frontier vs. your fine-tune on the same PR, 20–30 s** (`tools/side-by-side.sh`) | 🎞 | The GIF the results post opens with. |
| 6 | The completed master table rendered in a clean dark-theme viewer | 🖼 | The single most-shared image of the project. Make it legible at phone width. |
| 6 | If D ran: `ray status` showing two nodes; the 70B loading with both memory bars filling in NVIDIA Sync | 🖼 🎞 | "256 GB across two desktop boxes" gets attention on its own. |
| 6 | **Lab shot**: both Sparks, the dashboard on one screen, the results table on the other | 📷 | Results-post cover. |
| 7 | Hugging Face model card; the README top; the first external star or comment | 🖼 | Closing the loop publicly. |
| 7 | **The 30-minute reproduction as a VHS tape** (`tools/tapes/reproduce-eval.tape`) | 🎞 | Scripted, so it is honest: anyone can re-render it. Becomes the README's demo GIF. |

Standing rule from §0: one shareable artifact per session, minimum. If a session ends with nothing new in `evidence/`, the session isn't over.

### 11.3 Tooling

**Install once on the Mac:**
```bash
brew install ffmpeg gifsicle asciinema agg   # convert, optimise, record terminals, render .cast → GIF
brew install ttyd vhs                          # scripted, reproducible terminal demos
```
On the Sparks (DGX OS is Ubuntu-based): `sudo apt install asciinema`, or `pipx install asciinema`. Record there, render on the Mac.

**Screen recordings (anything with a GUI: NVIDIA Sync, DGX Dashboard, MLflow, a browser):** macOS `⌘⇧5` → choose a region → record → the `.mov` lands on the Desktop. Then:
```bash
tools/gif.sh ~/Desktop/rec.mov \
  experiments/01-finetune-vs-frontier/evidence/phase-1/2026-09-13-vllm-boot \
  --start 3 --duration 25 --speed 2
```
That writes a two-pass-palette GIF for the README and an H.264 MP4 for LinkedIn/X in `raw/`. Options: `--fps` (8 for slow terminal output, 15 for UI motion), `--width` (1200 for README, 800 for social), `--speed`.

**Terminal sessions (training, vLLM logs, the first JSON):** asciinema records text, not pixels — tiny files, crisp output, re-renderable at any size later.
```bash
asciinema rec -c "python train.py --config configs/run1.yaml" \
  experiments/01-finetune-vs-frontier/evidence/phase-1/raw/2026-09-13-smoke-train.cast
agg --speed 2 --font-size 16 --cols 100 --rows 30 \
  experiments/01-finetune-vs-frontier/evidence/phase-1/raw/2026-09-13-smoke-train.cast \
  experiments/01-finetune-vs-frontier/evidence/phase-1/2026-09-13-smoke-train.gif
```

**Side-by-side (Phase 6):** record each condition separately at the same terminal size, font, and theme, then:
```bash
tools/side-by-side.sh raw/frontier.mov raw/finetune.mov \
  experiments/01-finetune-vs-frontier/evidence/phase-6/2026-10-18-side-by-side-review --speed 1.5
```

**Reproducible demos (Phase 7):** a VHS tape is a script that types the commands and records the result. Re-render after every change so the README GIF never goes stale. Starter tape in `tools/tapes/reproduce-eval.tape`; render with `vhs tools/tapes/reproduce-eval.tape`.

**Official NVIDIA views worth screenshotting:** NVIDIA Sync's cluster view (what you captured on Sep 6–7) and the [DGX Dashboard](https://build.nvidia.com/spark/dgx-dashboard) for per-node GPU, memory, and temperature graphs. Both are dark-themed and read well at phone width.

### 11.4 Publishing rules

- **Secrets:** never record with `HF_TOKEN`, API keys, or your management LAN in frame. Export tokens from a file (`export HF_TOKEN=$(cat ~/.hf-token)`) and clear scrollback before recording. The cluster's `10.100.x` addresses live only on the QSFP cable and are fine to show.
- **Terminal look:** one font, 14–16 pt, dark theme, roughly 100×30, tmux status bar hidden. Consistency is what lets Phase 2 clips become Phase 6 panes.
- **Sizes:** README GIFs under ~10 MB (GitHub renders larger ones slowly), 8–12 fps, at most 1200 px wide. LinkedIn and X get the MP4, not the GIF.
- **Cadence:** one micro-post per phase exit — that phase's bold artifact plus the session's three BUILDLOG bullets — and the three long posts from §10. The first micro-post is available today: the Cluster Assistant sequence and the 185.96 Gbps number.
- **Attribution:** when a capture shows an NVIDIA tool, name it. It is accurate, and the [DGX Spark developer forum](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10) is a natural place to share it.

---

## 12. Suggested Calendar (adjust to life; keep the order)

| Weekend | Dates (2026) | Phase |
|---|---|---|
| 1 | Sep 5–6 | Phase 0 + start Phase 1 |
| 2 | Sep 12–13 | Phase 1 exit |
| 3 | Sep 19–20 | Phase 2 |
| 4 | Sep 26–27 | Phase 3 |
| 5 | Oct 3–4 | Phase 4 |
| 6 | Oct 10–11 | Phase 5 (overnight runs into the week) |
| 7 | Oct 17–18 | Phase 5 spillover + Phase 6 |
| 8 | Oct 24–25 | Phase 7 |
| — | Oct 31–Nov 15 | Buffer (slippage, stretch demo) |

Two weekends of built-in slack before OMSCS. If a phase slips, the buffer absorbs it — the order never changes, and no gate gets waived to "catch up."

---

## 13. Definition of Done (the whole project)

- [ ] Master results table complete, every cell traceable to a run ID
- [ ] Eval v2 frozen, documented, and contamination-proof by construction
- [ ] Dataset card + experiment card + per-miss autopsies public
- [ ] Adapter/model card on Hugging Face (license permitting)
- [ ] `bench/` chapter published with real inference numbers
- [ ] Three posts live; results post submitted to aggregators
- [ ] Site + GitHub profile updated; zero dangling "coming soon"
- [ ] Stranger test passed

---

## Appendix A — Quick-Reference Starting Points

**Training config (Run 1):** LoRA r=16, α=32, dropout 0.05, lr 1e-4–2e-4 cosine w/ warmup ~3%, 2–3 epochs, bf16, packing on. QLoRA only if memory or wall-time forces it.

**Contamination checklist (run before any training):** eval repos excluded at repo level → exact-hash overlap scan (train vs. both eval sets) → near-dup scan (MinHash/similarity) → frontier labeler log confirms zero eval PRs processed → results pasted into dataset card.

**Schema-failure policy:** unparseable response = 0 findings (scored as misses) + schema-failure counter +1. Never hand-repair an output during eval.

**Prompt-format rule:** the training example template and the inference template are the same file. One template, imported by both pipelines.

**Cost accounting:** frontier $/review = OTel tokens × current API price. Local marginal $/review = measured watts × hours × your $/kWh ÷ reviews in window. Local amortized adds (hardware ÷ 36 months) ÷ monthly reviews, counting only the Spark(s) the condition ran on. Publish all three; let readers pick their favorite fight.

**Serving rule:** one engine (vLLM), one pinned NGC container tag, one OpenAI-compatible provider in AgentReview. The artifact evaluated is the artifact served: merged bf16 for C, exported quantized checkpoint for C-q. Unmerged `--enable-lora` serving is allowed for Phase 5 spot checks only; nothing from it enters the master table.

**Two-Spark quick reference (from NVIDIA's Connect Two Sparks and vLLM playbooks — confirm against the current playbook before typing):**
- Same physical QSFP port on both nodes; one cable carries full bandwidth. `ibdev2netdev` shows which logical pair is `(Up)` — each physical port exposes two interfaces (e.g. `enp1s0f1np1` and `enP2p1s0f1np1`).
- Static IPs on the cluster interfaces. **Ours are set** (by NVIDIA Sync's Cluster Assistant): `10.100.240.0/24` and `10.100.241.0/24`, `.1` on `spark-927a`, `.2` on `spark-11ed` — see `infra/cluster.md`. NVIDIA's manual playbook uses `192.168.100.x` / `192.168.101.x` in its examples; substitute, never apply its netplan file on top.
- Same username on both nodes; passwordless SSH via NVIDIA's `discover-sparks` script.
- Multi-node vLLM: `run_cluster.sh` from the vLLM repo, head on Spark 1, worker on Spark 2, with `VLLM_HOST_IP`, `NCCL_SOCKET_IFNAME`, `UCX_NET_DEVICES`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME` all set to the QSFP interface; then `vllm serve <model> --tensor-parallel-size 2 --distributed-executor-backend ray`. Run head and worker in `tmux` — an SSH drop tears down the cluster.
- Two-node FSDP fine-tuning (bench only): NVIDIA's PyTorch playbook uses Docker Swarm plus `accelerate launch` with its `config_fsdp_lora.yaml` (FULL_SHARD, bf16, `num_machines: 2`).
- Unified-memory OOM below nominal capacity: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`, then lower `--gpu-memory-utilization`.

---

## Appendix B — Official NVIDIA Documentation Index

Order of authority when sources disagree: NVIDIA's DGX Spark User Guide, then NVIDIA's playbooks, then vLLM's own docs, then the DGX Spark developer forum, then everything else. The playbooks repo is updated frequently (it carries "Last Updated" dates per playbook), so re-read the relevant one the week its phase starts and note the date in `BUILDLOG.md`.

| Topic | Official source |
|---|---|
| DGX Spark User Guide — setup, DGX OS, software, updates, recovery | [docs.nvidia.com/dgx/dgx-spark](https://docs.nvidia.com/dgx/dgx-spark/) |
| Hardware specification (GB10, 128 GB LPDDR5x at 273 GB/s, ConnectX-7 2× QSFP) | [Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) |
| Connecting Sparks — topologies, approved QSFP cables, interface naming | [Spark Stacking](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html) |
| Automated multi-node setup with link-speed validation (what bridged `Lucia`) | [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) · [NVIDIA Sync User Guide](https://docs.nvidia.com/sync/latest/) |
| Per-node GPU / memory / thermal graphs for screenshots | [DGX Dashboard](https://docs.nvidia.com/dgx/dgx-spark/dgx-dashboard.html) · [playbook](https://build.nvidia.com/spark/dgx-dashboard) |
| All DGX Spark playbooks (published form) | [build.nvidia.com/spark](https://build.nvidia.com/spark) |
| Playbook source, assets, and scripts (Apache-2.0) | [github.com/NVIDIA/dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) |
| vLLM for Inference — single node, two nodes via Ray, model support matrix | [playbook](https://build.nvidia.com/spark/vllm) · [source](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md) |
| vLLM NGC container | [catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm) · [release notes](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) |
| PyTorch NGC container | [catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) |
| Connect Two Sparks — cabling, netplan, passwordless SSH | [playbook](https://build.nvidia.com/spark/connect-two-sparks) · [source](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/README.md) |
| NCCL for multiple Sparks — build and `all_gather_perf` validation | [playbook](https://build.nvidia.com/spark/nccl) |
| Fine-Tune with PyTorch — LoRA/QLoRA scripts, two-node FSDP via accelerate | [playbook](https://build.nvidia.com/spark/pytorch-fine-tune) · [assets](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/pytorch-fine-tune/assets) |
| Fine-Tune Faster with Unsloth | [playbook](https://build.nvidia.com/spark/unsloth) |
| Fine-Tune with NVIDIA NeMo AutoModel | [playbook](https://build.nvidia.com/spark/nemo-fine-tune) |
| Fine-Tune LLMs with LLaMA Factory | [playbook](https://build.nvidia.com/spark/llama-factory) |
| NVFP4 Quantization with NVIDIA Model Optimizer, served by vLLM | [playbook](https://build.nvidia.com/spark/nvfp4-quantization) |
| Speed Up Inference with Speculative Decoding | [playbook](https://build.nvidia.com/spark/speculative-decoding) |
| DGX Spark performance guide — offline/online serving, fine-tuning, dual-Spark RDMA bandwidth and latency | [performance_benchmarking_guide.md](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md) |
| NVIDIA technical blog — software updates, NVFP4, dual-Spark results | [New Software and Model Optimizations Supercharge NVIDIA DGX Spark](https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark/) |
| Owner community, known issues, container reports | [DGX Spark developer forum](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10) |

**vLLM project docs used alongside the NVIDIA material** (not NVIDIA, but authoritative for the engine): [LoRA adapter serving](https://docs.vllm.ai/en/latest/features/lora.html), [quantization methods](https://docs.vllm.ai/en/latest/features/quantization/), [benchmarking CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/).
