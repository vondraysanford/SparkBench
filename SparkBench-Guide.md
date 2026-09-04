# SparkBench — A-to-Z Build Guide

**Working hypothesis (one sentence):** A LoRA fine-tuned 7–14B open model, running locally on the DGX Spark, can replace the frontier model inside AgentReview's Quality Agent at comparable precision/recall and near-zero marginal cost per review.

**The name & shape:** `SparkBench` is the umbrella repo for everything measured on the DGX Spark, in two chapters: `bench/` — the inference benchmarking suite your site already promises (throughput, latency, memory across models and quantization levels) — and `experiments/01-finetune-vs-frontier/` — this guide's study, the flagship. One repo, one portfolio card, and it retires the site's "coming soon" placeholder the day it goes public. Because the wider world associates "SparkBench" with Apache Spark benchmarking tooling, the disambiguation tagline travels with the name everywhere it appears: **"SparkBench — benchmarks and fine-tuning studies on the NVIDIA DGX Spark. Not that Spark."**

**Timebox:** ~7 working weekends, September → mid-November 2026. Done before OMSCS starts.

---

## 0. Operating Rules (read before every session)

1. **Hard gate:** Do not start Phase N+1 until Phase N's exit test passes. Every exit test below is written to be binary — it passed or it didn't. No "mostly done."
2. **Evidence at the moment of creation:** The screenshot/number/GIF gets captured in the same terminal session it first appears, into `evidence/phase-N/`, with the naming convention in §11. If you tell yourself "I'll grab it later," you won't, and the writeup will be worse for it.
3. **One-touch test set:** The final test sets (Eval-18 and v2-TEST) are run against the fine-tuned model **exactly once**, in Phase 6. All iteration happens on v2-DEV. The moment you re-run the test set "just to check," your headline numbers are contaminated.
4. **Iteration budget:** Maximum 5 full training runs in Phase 5. Budgets force decisions; unlimited retries force overfitting.
5. **Every run is logged:** One MLflow experiment for the whole project, one run per training/eval execution. Any number that appears in the final table must trace to a run ID. (Same MLflow discipline as DriftWatch — one consistent story across your portfolio.)
6. **Log lines:** End every working session by adding 3 bullets to `BUILDLOG.md` — what worked, what broke, what surprised you. These become the blog posts. Write them while the pain is fresh.
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
| Deterministic merge/synthesis code | **B:** Small model, prompted, untuned |
| Findings JSON schema | **C:** Small model, LoRA fine-tuned |
| Seeded PRs and planted bugs | **C-q (optional):** C, quantized for Ollama serving |
| Temperature/decoding settings | |
| Harness version (git tag it in Phase 0) | |

**Scope decision:** Quality Agent only. It carries the headline metric and the hardest job. Do not fine-tune all three agents — that triples data work for zero additional story. (If Phase 2 reveals the quality lane is hopeless for a small model, the documented fallback is the Docs Agent — but decide that at the Phase 2 gate, not mid-phase.)

---

## 2. The Metrics Contract

Primary metrics, defined once, never renegotiated after Phase 0:

- **Recall:** planted bugs found / planted bugs present (report raw counts alongside %, always).
- **Precision:** true findings / all findings, human-judged with the same scoring worksheet you built for AgentReview Phase 5.
- **Schema validity:** % of responses that parse against the findings schema on first try. A parse failure counts as a miss for recall *and* is tracked separately.
- **Latency:** wall-clock seconds per review, from OpenTelemetry.
- **Cost per review:** frontier = tokens × current API pricing (from OTel token counts). Local = two honest lines: *marginal* (measured watts × hours × your $/kWh) and *amortized* (hardware cost ÷ 3-year straight line ÷ reviews). Report both; never claim "free" without the footnote.

### Master results table (fill incrementally; this exact table tops the final README)

| Condition | Eval-18 recall | Eval-18 precision | v2-TEST recall (95% CI) | v2-TEST precision (95% CI) | Schema-valid % | Latency/review | $/review |
|---|---|---|---|---|---|---|---|
| A. Frontier, prompted | | | | | | | |
| B. Small, untuned | | | | | | | |
| C. Small, fine-tuned | | | | | | | |
| C-q. Fine-tuned, quantized | | | | | | | |

Statistical honesty rules: on Eval-18 (n=18) report counts only — "16/18," never "88.9%," and never claim significance. On v2-TEST (n≈40+) report bootstrap 95% confidence intervals. If intervals overlap heavily, the writeup says so.

---

## 3. Phase 0 — Scope Lock & the Experiment Card

*Half a weekend (Saturday morning).*

**Goal:** Freeze every decision that could be quietly renegotiated later.

**Tasks**
1. Create the `sparkbench` repo; scaffold: root `README.md` (umbrella pitch + disambiguation tagline), `BUILDLOG.md`, `bench/`, and `experiments/01-finetune-vs-frontier/` containing `experiment-card.md`, `evidence/`, `data/`, `training/`, `eval/`. Init DVC for data versioning, MLflow for run tracking. Decide Hugging Face naming now: `sparkbench-quality-agent-<basemodel>` for the adapter.
2. Git-tag the AgentReview harness commit that all conditions will run against.
3. Pick two candidate base models. Criteria: 7–14B parameters, strong on code, license compatible with publishing weights (Apache-2.0 preferred — matters if you push the adapter to Hugging Face). As of writing, the Qwen coder-class models (7B/14B, Apache-2.0) and Llama 3.1 8B are the reference points — **but verify the current small-model landscape the week you start; it moves monthly.** Log the two picks and the reason in the experiment card.
4. Write `experiment-card.md`: the hypothesis, the conditions table (§1), the metrics contract (§2), the three pre-registered headlines, the iteration budget (5 runs), and the **pre-registered success margin** for Phase 5 (recommended: fine-tuned must beat untuned by ≥10 points of v2-DEV recall at no precision loss, with schema validity ≥95%).
5. Draw the empty master results table into the README.

**Exit test (binary):** A stranger reading `experiment-card.md` can answer, without asking you: what is being compared, on what data, by what metrics, and what each of the three outcomes would look like. The empty results table exists with every row and column named. Harness commit is tagged.

**Evidence to capture:** the experiment card itself; screenshot of the repo tree; the tagged harness commit hash in `BUILDLOG.md`.

**If it stalls:** you're over-deliberating the model choice. Pick the two most boring defensible candidates and move — Phase 1's smoke test is allowed to eliminate one.

---

## 4. Phase 1 — Training Environment Smoke Test on the Spark

*One weekend.*

**Goal:** Prove the full train → save → reload → serve loop works on GB10/aarch64 before any real data exists. Environment quirks on ARM + Blackwell are exactly the kind of thing you verify, not assume — and exactly the war stories your blog runs on.

**Tasks**
1. Stand up the training stack. Candidates, in order of least-surprise: Hugging Face `transformers` + `peft` + `trl` (SFTTrainer); `torchtune`; Unsloth **if** its current build supports your platform (verify, don't assume). Use NVIDIA's DGX Spark playbooks as the starting point; log which stack won and why.
2. Pull both candidate base models. Confirm bf16 LoRA fits comfortably in unified memory for the 14B (it should, with ~128 GB — record actual peak usage).
3. Hand-write ~100 toy training examples in the **exact** chat format the Quality Agent uses at inference time — same system prompt structure, diff + static-analysis context in, findings-schema JSON out. (Rule for the whole project: training format = inference format, byte-for-byte template match. Format drift between train and serve is the #1 silent killer of fine-tunes.)
4. Run a 50–100 step LoRA smoke train. Log to MLflow.
5. Save the adapter, reload it, merge it, serve the merged model through vLLM, and send it one review request.
6. Measure training throughput (tokens/sec) and project the full-run wall time for ~3k examples × 2–3 epochs.

**Exit test (binary):** (a) smoke-run loss curve decreases in MLflow; (b) the saved adapter reloads and the served model returns at least one schema-valid findings JSON; (c) measured tok/s projects a full training run at **under 12 hours** — overnight-able. If (c) fails: drop to the 7B, or switch to QLoRA, and re-run the smoke test until it passes.

**Evidence to capture:** MLflow loss-curve screenshot; peak memory reading; terminal log of the first schema-valid generation from your own fine-tuned (toy) model — that moment is the money screenshot of the whole project; the tok/s number and projected full-run hours in `BUILDLOG.md`.

---

## 5. Phase 2 — Baseline the Untuned Small Model

*One weekend.*

**Goal:** Two complete rows of the master table before any fine-tuning exists. Without row B, you cannot prove the fine-tune did anything.

**Tasks**
1. Add the local model as a provider behind the same interface the Quality Agent already calls — the identical pattern to DocQuery's provider swap. Roslyn, merge logic, schema, and decoding settings untouched.
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
2. After each run: merge → serve via vLLM → full v2-DEV eval + schema-validity + Eval-18-style spot check. Log everything to MLflow with the config as params.
3. Iterate within budget. Highest-leverage knobs, in order: data mix (buggy:clean ratio), epochs (watch for the precision cliff from overfitting), learning rate, rank. Change one thing per run.
4. If Ollama/GGUF is a deployment target, quantize the current-best checkpoint once mid-phase and eval **the quantized artifact** on DEV. You evaluate what you serve — quantization moves numbers.
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
3. **Fill the `bench/` chapter:** you now have months of real throughput/memory/latency numbers from this hardware across training and serving. Write them up as reproducible benchmark scripts plus a results page — the inference-suite promise from your site, fulfilled inside the same repo.
4. **Site + GitHub:** the site's existing SparkBench card finally gets its link and absorbs the fine-tuning story into its description; retire the "Up next — DGX Spark benchmarks" line in the profile README; update llms.txt; consider pinning `sparkbench` in place of the Twitter bot.
5. **Writeups from BUILDLOG.md** (already written in pieces — assemble, don't compose): at minimum (i) the data + decontamination post, (ii) the training-on-Spark war-stories post, (iii) the results post led by the table. Submit each to the .NET aggregators/newsletters; the results post is the Show HN / r/LocalLLaMA candidate.
6. **Stretch, only if time remains:** a model toggle on the AgentReview live demo (frontier vs. your fine-tune) — a public artifact almost nobody else has.

**Exit test (binary):** the stranger test — a clean machine (or a friend) reproduces the headline eval from the README in under 30 minutes without contacting you; results post published; zero "coming soon" promises remain on your site or profile that this project was supposed to fulfill.

**Evidence to capture:** public links; README screenshot; first external engagement (comment, star, question) whenever it lands.

---

## 11. Evidence Conventions

```
experiments/01-finetune-vs-frontier/evidence/
  phase-0/  2026-09-05-experiment-card.png
  phase-1/  2026-09-12-first-schema-valid-generation.png
            2026-09-12-smoke-loss-curve.png
  phase-6/  2026-10-18-side-by-side-review.gif
```

- Name: `YYYY-MM-DD-what-it-shows` — no `img_4021.png`, no `final_v2_REAL`.
- Screenshots: full terminal context visible (command + output), not cropped fragments.
- Numbers: also land in `BUILDLOG.md` as text the moment they exist — screenshots rot, text greps.
- GIFs: capture at the moment of first success; 20–30s; terminal-native tools are fine.

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

**Cost accounting:** frontier $/review = OTel tokens × current API price. Local marginal $/review = measured watts × hours × your $/kWh ÷ reviews in window. Local amortized adds (hardware ÷ 36 months) ÷ monthly reviews. Publish all three; let readers pick their favorite fight.
