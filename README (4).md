# CSE 151B Competition Submission

This repository contains the inference pipeline used for the CSE 151B mathematical reasoning competition submission. The final inference path uses the required base model, `Qwen/Qwen3-4B-Thinking-2507`, with prompt engineering, self-consistency, model-intrinsic answer-format repair, and optional LoRA/QLoRA adapters trained from the same required base model. A training roadmap for pushing beyond the prompt-only plateau is documented in [TRAINING.md](TRAINING.md).

## Official Reproducible Entry Point

For submission verification, use `run_inference.run_inference()` as the single entry point. The default call regenerates `results/private_4b/submission.csv` directly from `private.jsonl` using `Qwen/Qwen3-4B-Thinking-2507` and the fixed default hyperparameters documented below:

```python
from run_inference import run_inference

run_inference(data_path="private.jsonl", output_dir="results/private_4b")
```

The precomputed selector and leaderboard CSVs under `results/` are retained as experiment artifacts. Do not treat those files as the single-entry submission path unless their full source-generation and selection workflow is explicitly run through `run_inference()`.

## Hardware And Runtime

- GPU type used: 2x NVIDIA B200
- Approximate total generation/inference time: about 10-15 minutes for one full 943-example private-set pass on 2x B200 with a warm vLLM/HuggingFace cache, including model loading, vLLM warmup, batched generation, sampled candidates, final answer cleanup, structured repair, merging, and CSV export. Cold-cache runs and reproducing every ensemble source from scratch take longer.
- Memory behavior: each GPU is used by one independent vLLM process. During the dry run, both GPUs were active at roughly 95-100% utilization and released memory after completion.

Runtime will vary with dataset size, cache state, and whether HuggingFace model weights are already downloaded.

## Dependencies

Install the reproduction environment with:

```bash
python3 -m pip install -r requirements-training.txt
```

This includes vLLM for generation, `safetensors` for adapter/model loading, and `antlr4-python3-runtime==4.11` for the local public-set LaTeX judger used by `evaluate_results.py`. Private inference itself does not use the public-set judger, but keeping the evaluator dependency installed makes migrated-machine validation reproducible.

## Model Weights

No fine-tuned checkpoint is required for the baseline. The pipeline loads the designated competition model directly:

```text
Qwen/Qwen3-4B-Thinking-2507
```

The model is downloaded automatically by HuggingFace/vLLM into the normal cache directories, such as `~/.cache/huggingface` and `~/.cache/vllm`. If the verification environment is offline, pre-download the model into the HuggingFace cache or set `HF_HOME` / `TRANSFORMERS_CACHE` to a directory containing the cached model files.

If trained adapters are used, place them locally, for example under `adapters/qwen3-4b-math-sft-trace-only-r32` and `adapters/qwen3-4b-math-sft-trace-union-r32`, or upload them to HuggingFace Hub. Pass one adapter with `--adapter`, or use the public-tested hybrid with `--mcq-adapter` and `--free-adapter`. Environment alternatives are `QWEN_MATH_ADAPTER`, `QWEN_MATH_MCQ_ADAPTER`, and `QWEN_MATH_FREE_ADAPTER`.

For a strict migrated-machine verification run with adapters, prefer HuggingFace Hub adapter IDs in those arguments/environment variables, or make sure the referenced local `adapters/...` directories are included with the submission.

Optional, for faster and higher-rate downloads:

```bash
huggingface-cli login
# or set HF_TOKEN in the environment
```

## Single Entry Point

The required entry point is [run_inference.py](run_inference.py). It exposes:

```python
from run_inference import run_inference

run_inference(data_path="private.jsonl", output_dir="results/private_4b")
```

Optional adapter/hybrid mode is also contained in the same function:

```python
run_inference(
    data_path="private.jsonl",
    output_dir="results/private_hybrid",
    mcq_adapter="adapters/qwen3-4b-math-sft-trace-union-r32",
    free_adapter="adapters/qwen3-4b-math-sft-trace-only-r32",
)
```

This function performs the full end-to-end pipeline:

1. Loads `Qwen/Qwen3-4B-Thinking-2507` with vLLM.
2. Splits the dataset across the configured GPUs.
3. Runs batched free-form generation.
4. Runs self-consistency with `MCQ_N = 5` and `FREE_N = 3`.
5. Optionally applies model-intrinsic candidate adjudication for tied or ambiguous candidates.
6. Applies bad-format retry, final-answer cleanup, and direct fallback through the same Qwen3-4B model.
7. Runs structured answer-format repair for any remaining no-box/no-letter records.
8. Merges GPU shards.
9. Writes the final `id,response` submission CSV.

The default output is:

```text
results/private_4b/submission.csv
```

## CLI Usage

Place `private.jsonl` at the repository root, then run:

```bash
python3 run_inference.py --data private.jsonl --output-dir results/private_4b --gpus 0,1
```

With a trained adapter:

```bash
python3 run_inference.py \
  --data private.jsonl \
  --output-dir results/private_hybrid \
  --gpus 0,1 \
  --mcq-adapter adapters/qwen3-4b-math-sft-trace-union-r32 \
  --free-adapter adapters/qwen3-4b-math-sft-trace-only-r32
```

For a single GPU:

```bash
python3 run_inference.py --data private.jsonl --output-dir results/private_4b_single --gpus 0
```

For the included public-shaped dry-run file, if present:

```bash
python3 run_inference.py --data private_like.jsonl --output-dir results/private_like_entrypoint --gpus 0,1
```

## Candidate Artifacts

The following files record private leaderboard experiments and ablations. They are useful for analysis, but the official reproducible entry point above is the submission-ready path unless one of these selection workflows is deliberately promoted into `run_inference()`.

Private leaderboard feedback now shows that the learned selector transferred better than the earlier precision-biased rule/vote variants:

```text
private_precision_mcq_vote_precision_free:      0.646
private_candidate_vote_full_available_precision: 0.650
previous private_majority5_mcq:                 0.660
private_learned_selector_precision_best:        0.674
private_strict_mcq_exact_free_v1:               0.664
private_learned_selector_precision_exact_v1:    0.678
private_selector_precision_exact_ensemble_lr0p12_v1: 0.674
private_conservative_delta_free_v1:             0.678
```

So the active best confirmed private candidate is `private_learned_selector_precision_exact_v1`. The raw exact-free routed source is rejected as a standalone submission because it scored below the learned selector and below `private_learned_selector_precision_best`.

The ensemble candidate at `results/private_selector_precision_exact_ensemble_lr0p12_v1/submission.csv` is now rejected as a primary submission despite better local pseudo-private metrics. It changed 299 private rows versus the 0.678 exact selector, mostly through `shuffle_free_sc`, and scored 0.674 on the private leaderboard.

The low-overfit exploratory candidate at `results/private_conservative_delta_free_v1/submission.csv` matched the confirmed best private leaderboard score at 0.678. It starts from the confirmed exact selector and applies only high-consensus free-form replacements. On public validation it scored 728/1126 = 64.65%, changing 12 public free-form rows with an estimated +2/-0 delta. On private it changes only 13 free-form rows, so it is confirmed neutral rather than an upgrade. An even stricter fallback is `results/private_conservative_delta_free_strict_v1/submission.csv`; it scored 727/1126 public and changes only 3 private rows.

The private feedback supports the strategy: generate diverse Qwen3-4B candidate sources, but only submit them after learned selection rather than as raw routed sources.

For the majority5-targeted SFT continuation, the headline public score is not the only decision signal because the adapter was trained from public-derived repair examples. On held-out targeted validation problem IDs, the strict-MCQ/new-free hybrid improved strict repair from 70/104 to 72/104. On public IDs not used in the targeted SFT train split, it improved strict repair from 362/536 to 371/536. This makes the hybrid worth testing, but it is still more aggressive than strict repair because it changes 244 private free-form answer keys.

Public validation snapshot:

```text
public_precision_v1:                         717 / 1126 = 63.68%  (MCQ 249, free 468)
public_candidate_vote_full_available:        739 / 1126 = 65.63%  (MCQ 273, free 466)
public_candidate_vote_full_available_precision: 742 / 1126 = 65.90%  (MCQ 277, free 465)
public_candidate_vote_precision_mcq_free_dup_ultra: 743 / 1126 = 65.99%  (MCQ 277, free 466)
public_precision_mcq_vote_precision_free:    745 / 1126 = 66.16%  (MCQ 277, free 468)
public_dpo_candidate_step80_eval:            721 / 1126 = 64.03%  (MCQ 262, free 459)
public_mixed_sft_v1_eval:                    713 / 1126 = 63.32%  (MCQ 256, free 457)
public_strict_mcq_mixed_sft_free:            721 / 1126 = 64.03%  (MCQ 264, free 457)
public_narrow_sft_v1_eval:                   719 / 1126 = 63.85%  (MCQ 259, free 460)
public_strict_mcq_narrow_sft_free:           724 / 1126 = 64.30%  (MCQ 264, free 460)
public_lora_majority5_targeted_eval:         711 / 1126 = 63.14%  (MCQ 250, free 461)
public_strict_mcq_new_lora_free_badonly:     715 / 1126 = 63.50%  (MCQ 264, free 451)
public_strict_mcq_new_lora_free:             725 / 1126 = 64.39%  (MCQ 264, free 461)
public_learned_selector_precision_best:      725 / 1126 = 64.39%  (MCQ 269, free 456)
public_exact_free_v1:                        712 / 1126 = 63.23%  (MCQ 250, free 462)
public_strict_mcq_exact_free_v1:             726 / 1126 = 64.48%  (MCQ 264, free 462)
public_learned_selector_precision_exact_v1:  726 / 1126 = 64.48%
public_mcq_shuffle_source_v1:                745 / 1126 = 66.16%  (MCQ 283, free 462)
public_selector_precision_exact_ensemble_v1: 727 / 1126 = 64.56%
public_conservative_delta_free_v1:           728 / 1126 = 64.65%  (MCQ 270, free 458)
public_conservative_delta_free_strict_v1:    727 / 1126 = 64.56%  (MCQ 270, free 457)
public_mcq_shuffle_overlay_all_v1:           740 / 1126 = 65.72%  (MCQ 284, free 456)
public_mcq_shuffle_all_conservative_free_v1: 742 / 1126 = 65.90%  (MCQ 284, free 458)
```

The learned selector was trained only on public correctness labels over already-generated Qwen3-4B candidate outputs, then checked with repeated pseudo-private 30/70 public splits. Its best precision-pool run has pseudo-final mean 64.13% versus 63.24% for strict repair, with p10 delta +0.38 percentage points. It is still riskier than strict repair because it learns from public labels, but it is less hand-mined than the ultra rule routers.

The exact-free prompt family adds `--prompt-style exact_free --finalize-all-free`, forcing a second-stage final-answer formatter over every free-form row. Standalone exact-free hurts MCQ, so the usable source is strict MCQ plus exact-free free-form. Adding that source to the learned selector improved pseudo-private delta to mean +0.95 percentage points and p10 +0.51 percentage points versus strict repair.

The ensemble source builder in `run_ensemble_sources.py` adds two more source families: five MCQ option-order shuffles mapped back to the original labels, and `free_n=5` exact-free self-consistency for free-form rows. The broad learned selector over these sources over-routed and scored 0.674 private, but a direct confidence overlay for MCQ-only shuffle disagreements is now the next cleaner probe.

Recommended generated private CSVs:

```text
confirmed best private-LB candidate:
  results/private_learned_selector_precision_exact_v1/submission.csv
  (private leaderboard: 0.678; same selector family plus exact-free source)

low-delta exploratory candidate:
  results/private_conservative_delta_free_v1/submission.csv
  (private leaderboard: 0.678; 13 private free-form changes versus the 0.678 exact selector)

next high-upside inference-consistency candidate:
  results/private_mcq_shuffle_all_conservative_free_v1/submission.csv
  (unsubmitted; option-shuffle MCQ overlay plus confirmed-neutral conservative free overlay)

lower-risk MCQ-shuffle fallback:
  results/private_mcq_shuffle_t2m1_conservative_free_v1/submission.csv
  (unsubmitted; requires at least 2 shuffle votes and margin 1 for MCQ answer replacement)

strict low-delta fallback:
  results/private_conservative_delta_free_strict_v1/submission.csv
  (unsubmitted; 3 private free-form changes versus the 0.678 exact selector)

rejected ensemble ablation:
  results/private_selector_precision_exact_ensemble_lr0p12_v1/submission.csv
  (private leaderboard: 0.674; over-routed to shuffle/free self-consistency source)

previous best private-LB candidate:
  results/private_learned_selector_precision_best/submission.csv
  (private leaderboard: 0.674)

fallback / sanity check:
  results/private_majority5_mcq_strict_repair/submission.csv

lower priority exact-free ablation:
  results/private_strict_mcq_exact_free_v1/submission.csv
  (private leaderboard: 0.664; strict MCQ plus exact-preserving free-form prompt family)

conservative SFT-assisted ablation:
  results/private_strict_mcq_new_lora_free_badonly/submission.csv
  (changes 5 private free-form rows versus strict repair)

known private-LB backup:
  results/private_majority5_mcq/submission.csv

do not prioritize based on latest private feedback:
  results/private_precision_mcq_vote_precision_free/submission.csv
  results/private_candidate_vote_full_available_precision/submission.csv
  adapters/qwen3-4b-math-sft-narrow-v1-r32
```

These CSVs were validated against `private.jsonl`: 943 rows, matching row order, no missing IDs, no extra IDs, no duplicate IDs, and no empty responses.

The latest pseudo-private split report is `reports/overfit_validation_precision_final.json`. In that 60-seed simulation, `public_precision_mcq_vote_precision_free` has a 65.98% mean pseudo-final score, while greedy public-rule selection overfits despite higher pseudo-leaderboard scores.

To regenerate the current precision source:

```bash
python3 run_inference.py \
  --data private.jsonl \
  --output-dir results/private_precision_v1 \
  --gpus 0,1
```

To rebuild the current label-free precision ensemble from available Qwen3-4B candidate files:

```bash
python3 select_results.py \
  --data private.jsonl \
  --output results/private_candidate_vote_full_available_precision/selected.jsonl \
  --result ultra:results/private_ultra_rules/selected.jsonl \
  --result ultra_dup:results/private_ultra_rules/selected.jsonl \
  --result hybrid:results/private_hybrid_union_mcq_trace_free/repaired.jsonl \
  --result trace:results/private_lora_trace_final/repaired.jsonl \
  --result union:results/private_lora_union_final/repaired.jsonl \
  --result bf16_repaired:results/private_4b/repaired.jsonl \
  --result opt3:results/private_4b_opt3/repaired.jsonl \
  --result precision:results/private_precision_v1/repaired.jsonl

python3 combine_results_by_format.py \
  --mcq-results results/private_candidate_vote_full_available_precision/selected.jsonl \
  --free-results results/private_precision_v1/repaired.jsonl \
  --output results/private_precision_mcq_vote_precision_free/selected.jsonl

python3 make_submission.py \
  --data private.jsonl \
  --results results/private_precision_mcq_vote_precision_free/selected.jsonl \
  --output results/private_precision_mcq_vote_precision_free/submission.csv
```

## Final Hyperparameters

These are fixed in `run_inference.py` and `infer_vllm.py` for reproducibility:

```text
model: Qwen/Qwen3-4B-Thinking-2507
weights: BF16 / no bitsandbytes for main inference
max_model_len: 16384
gpu_memory_utilization: 0.80
max_num_seqs: 256
max_num_batched_tokens: 32768
free-form max_tokens: 4096
MCQ max_tokens: 3072
free-form self-consistency samples: 3
MCQ self-consistency samples: 5
free-form sampling temperature: 0.6 when n > 1
MCQ temperature: 0.6 when n > 1
finalizer max_tokens: 512
prefix caching: enabled
attention backend: FLASH_ATTN
FlashInfer sampler: disabled
vLLM V1 multiprocessing: disabled
optional exact-free candidate flags: --prompt-style exact_free --finalize-all-free
optional ensemble source builder: python3 run_ensemble_sources.py --data private.jsonl --output-dir results/private_shuffle_free_sc_v1 --gpus 0,1 --shuffle-count 5 --free-n 5 --free-temperature 0.75 --prompt-style exact_free --baseline-result results/private_strict_mcq_exact_free_v1/repaired.jsonl
```

The scripts guard against accidentally using non-competition models. A different model requires explicitly setting `ALLOW_NONCOMPETITION_MODEL=1`, which is only for local experiments and should not be used for submission.

## Training Roadmap

The prompt-only/self-consistency pipeline validated at 61.19% on the public set. A conservative trace-only QLoRA adapter improved that to 62.08%. A second union-trace adapter improved MCQ but regressed free-form; the format-gated hybrid scored 700/1126 = 62.17% public accuracy. MCQ majority voting plus fixed free-form routing improved the public gate to 722/1126 = 64.12%. A label-free precision ensemble reached 745/1126 = 66.16% public accuracy but private feedback was weaker. The latest majority5-targeted SFT continuation scored 711/1126 standalone; keeping strict-repair MCQ and using that adapter for free-form scored 725/1126, with held-out split improvements, while the bad-format-only variant scored 715/1126 with only five private free-form replacements. A more aggressive public-gated selector reached 754/1126 = 66.96%, and `route_ultra2_rules.py` reaches 791/1126 = 70.25% public, but those rule routers are high-risk because the second-stage rules were mined against the full public set. Keep conservative and label-free candidates as backups. See [TRAINING.md](TRAINING.md) for details.

Short version:

1. Train a QLoRA adapter on curated math traces and competition-format answers.
2. Validate with the same `run_inference()` command on a held-out public split.
3. Only after SFT is stable, run outcome-reward RL with the local judger as the reward signal.
4. Upload the final adapter/checkpoint to HuggingFace Hub and pass the adapter path through `run_inference(..., adapter="...")`, `--adapter`, or `QWEN_MATH_ADAPTER`.

The final private inference must still use `Qwen/Qwen3-4B-Thinking-2507` as the base model and cannot call external tools/APIs at test time.

Implemented training commands:

```bash
python3 build_sft_dataset.py \
  --public public.jsonl \
  --scored results/two_gpu_opt3/scored.jsonl \
  --output-dir training_data/sft_trace_only \
  --no-gold-final

accelerate launch --num_processes 2 train_qlora_sft.py \
  --train-file training_data/sft_trace_only/sft_train.jsonl \
  --eval-file training_data/sft_trace_only/sft_valid.jsonl \
  --output-dir adapters/qwen3-4b-math-sft-trace-only-r32 \
  --num-train-epochs 1 \
  --learning-rate 5e-5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --gradient-accumulation-steps 8 \
  --save-steps 50 \
  --eval-steps 50 \
  --save-total-limit 2

python3 run_inference.py \
  --data public.jsonl \
  --output-dir results/public_lora_trace_eval \
  --gpus 0,1 \
  --adapter adapters/qwen3-4b-math-sft-trace-only-r32
```

Experimental GRPO is scaffolded in `train_grpo_rl.py`. `trl` is installed in this environment and listed in `requirements-training.txt` for reproducibility.

Full DPO preference optimization is implemented in `build_preference_dataset.py` and `train_dpo_adapter.py`. The best DPO checkpoint so far is `adapters/qwen3-4b-math-dpo-candidate-r32/checkpoint-80`; it scored 64.03% public standalone and did not improve the current precision selectors, so it is kept as a diagnostic artifact rather than a private-submission default.

Mixed external-data SFT is implemented in `build_mixed_math_sft_dataset.py`. The first conservative mixed run used OpenR1-Math-220k, NuminaMath-1.5, public-correct Qwen traces, license-filtered MathInstruct, and OpenMathInstruct-2 fallback rows. The resulting adapter `adapters/qwen3-4b-math-sft-mixed-v1-r32` scored 63.32% public standalone and 64.03% with strict MCQ routing, so it is also a diagnostic artifact rather than a private-submission default.

Hybrid public validation and private output:

```text
public_hybrid_union_mcq_trace_free: 700 / 1126 = 62.17%
public_majority5_mcq: 710 / 1126 = 63.06%
public_majority5_mcq_free_rules: 722 / 1126 = 64.12%
public_ultra_rules: 754 / 1126 = 66.96%
public_ultra2_rules: 791 / 1126 = 70.25%
public_ultra2_rules_no_cross: 790 / 1126 = 70.16%
public_precision_mcq_vote_precision_free: 745 / 1126 = 66.16%
public_strict_mcq_new_lora_free: 725 / 1126 = 64.39%
public_mixed_sft_v1_eval: 713 / 1126 = 63.32%
current recommended low-risk private candidate: results/private_majority5_mcq_strict_repair/submission.csv
new conservative SFT-assisted private candidate: results/private_strict_mcq_new_lora_free_badonly/submission.csv
new higher-upside SFT-assisted private candidate: results/private_strict_mcq_new_lora_free/submission.csv
highest-public-score private candidate, high overfit risk: results/private_ultra2_rules/submission.csv
conservative private candidate: results/private_majority5_mcq/submission.csv
higher-stability private candidate: results/private_majority5_mcq_free_rules/submission.csv
```


MCQ-majority and free-rule selection used for the current highest-public-score candidate:

```bash
python3 select_results.py \
  --data public.jsonl \
  --output results/public_majority5_mcq/selected.jsonl \
  --mcq-only \
  --result hybrid:results/public_hybrid_union_mcq_trace_free/scored.jsonl \
  --result targeted:results/public_lora_targeted_eval/scored.jsonl \
  --result base:results/two_gpu_opt3/scored.jsonl \
  --result grpo:results/public_grpo_trace_step8_eval/scored.jsonl \
  --result numina:results/public_lora_numina_eval/scored.jsonl

python3 route_free_rules.py \
  --data public.jsonl \
  --baseline results/public_majority5_mcq/selected.jsonl \
  --opt3 results/two_gpu_opt3/scored.jsonl \
  --bf16 results/two_gpu_bf16/scored.jsonl \
  --output results/public_majority5_mcq_free_rules/selected.jsonl
```

Highest-public-score ultra-v2 selection:

```bash
python3 route_ultra2_rules.py \
  --data public.jsonl \
  --baseline results/public_ultra_rules/scored.jsonl \
  --hybrid results/public_hybrid_union_mcq_trace_free/scored.jsonl \
  --trace results/public_lora_trace_eval/scored.jsonl \
  --targeted results/public_lora_targeted_eval/scored.jsonl \
  --grpo results/public_grpo_trace_step8_eval/scored.jsonl \
  --numina results/public_lora_numina_eval/scored.jsonl \
  --opt3 results/two_gpu_opt3/scored.jsonl \
  --opt2 results/two_gpu_opt2/repaired_scored.jsonl \
  --lora results/public_lora_eval/scored.jsonl \
  --union results/public_lora_union_eval/scored.jsonl \
  --bf16 results/two_gpu_bf16/repaired_scored.jsonl \
  --cross results/public_cross_adjudicate_5/scored.jsonl \
  --output results/public_ultra2_rules/selected.jsonl
```

Private ultra-v2 candidate:

```text
results/private_ultra2_rules/submission.csv
```

This CSV was validated against `private.jsonl`: 943 rows, no missing ids, no extras, no duplicate ids, no empty responses, and matching row order. It is not the currently recommended first submission because prior private-LB feedback showed public-label-mined rules transferring worse than their public score.

Rejected public-tested experiments:

```text
trace+Numina continuation SFT: 643 / 1126 = 57.10%
trace-only short GRPO step8: 691 / 1126 = 61.37%
targeted repair continuation SFT: 688 / 1126 = 61.10%
```



## Files

| File | Purpose |
|---|---|
| `run_inference.py` | Single official entry point for end-to-end private inference, with optional adapter loading |
| `infer_vllm.py` | One-GPU vLLM shard runner with prompts, MCQ voting, and cleanup passes |
| `repair_bad_format.py` | Structured model-intrinsic formatting repair pass |
| `merge_jsonl.py` | Merges GPU shard JSONL outputs |
| `make_submission.py` | Validates coverage and writes `id,response` CSV |
| `evaluate_results.py` | Public-set diagnostics only; not used for private scoring |
| `TRAINING.md` | LoRA/QLoRA SFT and RL/GRPO roadmap for the next score push |
| `build_sft_dataset.py` | Builds chat-format SFT data from public answers and clean correct traces |
| `build_external_sft_dataset.py` | Builds chat-format SFT data from an external public math dataset |
| `merge_sft_datasets.py` | Merges local and external SFT datasets into one train/validation split |
| `build_trace_union_dataset.py` | Builds trace-only SFT data from multiple scored public runs |
| `build_mixed_math_sft_dataset.py` | Builds filtered mixed external/public math SFT data |
| `analyze_public_errors.py` | Buckets public-set failures for targeted data curation |
| `build_targeted_sft_dataset.py` | Builds targeted repair SFT data from fixable public failures |
| `combine_results_by_format.py` | Combines MCQ rows from one result file with free-form rows from another for hybrid inference |
| `select_results.py` | Applies MCQ majority voting across multiple candidate model outputs |
| `mcq_shuffle_confidence_overlay.py` | Applies confidence-gated MCQ option-shuffle replacements on top of a baseline |
| `conservative_delta_selector.py` | Applies small high-consensus answer overlays against a baseline |
| `bagged_candidate_selector.py` | Experimental out-of-fold bagged selector over generated candidate sources |
| `route_free_rules.py` | Applies public-gated free-form routing rules on top of a selected baseline |
| `route_ultra2_rules.py` | Applies the highest-public-score, high-risk ultra-v2 routing rules |
| `adjudicate_results.py` | Experimental same-model cross-run adjudication, rejected after public gate |
| `train_qlora_sft.py` | Trains the QLoRA math adapter |
| `merge_lora.py` | Optionally merges a LoRA adapter into a BF16 checkpoint |
| `rl_rewards.py` | Local correctness/format reward helper for RL experiments |
| `train_grpo_rl.py` | Experimental TRL GRPO training entry point |
| `requirements-training.txt` | Reproduction dependencies for inference, public evaluation, and training experiments |
| `judger.py`, `utils.py` | Public-set local scoring helpers |
| `starter_code_cse151b_comp.ipynb` | Original starter notebook and exploratory workflow |

## Submission CSV Format

The generated CSV has exactly two columns:

```csv
id,response
```

The `response` field contains the full model response trace. CSV quoting is handled by Python's standard `csv` module, so commas, quotes, and newlines are escaped correctly.
