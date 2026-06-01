#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from infer_vllm import REQUIRED_MODEL, enforce_competition_model


REPO_DIR = Path(__file__).resolve().parent
MODEL_ID = REQUIRED_MODEL
MCQ_N = int(os.environ.get("QWEN_MATH_MCQ_N", "5"))
GPU_MEMORY_UTILIZATION = 0.80
FREE_N = int(os.environ.get("QWEN_MATH_FREE_N", "3"))
FREE_TEMPERATURE = float(os.environ.get("QWEN_MATH_FREE_TEMPERATURE", "0.6"))
MCQ_TEMPERATURE = float(os.environ.get("QWEN_MATH_MCQ_TEMPERATURE", "0.6"))
MAX_MODEL_LEN = 16384
MAX_NUM_SEQS = 256
MAX_NUM_BATCHED_TOKENS = 32768
MAX_TOKENS = 4096
MCQ_MAX_TOKENS = 3072
FINALIZER_MAX_TOKENS = 512
ADJUDICATOR_MAX_TOKENS = int(os.environ.get("QWEN_MATH_ADJUDICATOR_MAX_TOKENS", "384"))
ADJUDICATE_CANDIDATES = os.environ.get("QWEN_MATH_ADJUDICATE_CANDIDATES", "0") == "1"
PROMPT_STYLE = os.environ.get("QWEN_MATH_PROMPT_STYLE", "default").strip() or "default"
FINALIZE_ALL_FREE = os.environ.get("QWEN_MATH_FINALIZE_ALL_FREE", "0") == "1"
USE_BF16_WEIGHTS = True
ADAPTER_ID = os.environ.get("QWEN_MATH_ADAPTER", "").strip()
MCQ_ADAPTER_ID = os.environ.get("QWEN_MATH_MCQ_ADAPTER", "").strip()
FREE_ADAPTER_ID = os.environ.get("QWEN_MATH_FREE_ADAPTER", "").strip()
MAX_LORA_RANK = int(os.environ.get("QWEN_MATH_MAX_LORA_RANK", "64"))


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _has_gold(data_path: Path) -> bool:
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return "answer" in json.loads(line)
    return False


def _tail(path: Path, chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text[-chars:]


def _run_checked(cmd: list[str], log_path: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, check=True, env=env)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}\n"
            f"Last log lines from {log_path}:\n{_tail(log_path)}"
        )


def _shard_ranges(total: int, nshards: int) -> list[tuple[int, int]]:
    return [(i * total // nshards, (i + 1) * total // nshards) for i in range(nshards)]


def _parse_named_path(spec: str) -> tuple[str, str]:
    if ":" in spec:
        name, path = spec.split(":", 1)
        return name.strip(), path.strip()
    path = spec.strip()
    name = Path(path).name or "adapter"
    return name, path


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "candidate"


def _write_mcq_subset(data_path: Path, output_path: Path) -> int:
    rows = _load_jsonl(data_path)
    mcq_rows = [row for row in rows if row.get("options")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in mcq_rows:
            f.write(json.dumps(row) + "\n")
    return len(mcq_rows)


def _append_mcq_majority_selection(
    *,
    data_path: Path,
    output_dir: Path,
    baseline_results: Path,
    gpus: str | Sequence[str],
    model: str,
    repair: bool,
    adjudicate_candidates: bool,
    adjudicator_max_tokens: int,
    prompt_style: str,
    mcq_n: int,
    free_n: int,
    free_temperature: float,
    mcq_temperature: float,
    mcq_majority_adapters: Sequence[str] | None,
    mcq_majority_include_base: bool,
) -> Path:
    adapters = list(mcq_majority_adapters or [])
    if not adapters and not mcq_majority_include_base:
        return baseline_results

    mcq_data = output_dir / "mcq_majority_data.jsonl"
    mcq_count = _write_mcq_subset(data_path, mcq_data)
    if mcq_count == 0:
        return baseline_results

    result_name = "repaired.jsonl" if repair else "merged.jsonl"
    selector_specs = [f"baseline:{baseline_results}"]

    if mcq_majority_include_base:
        candidate_dir = output_dir / "mcq_vote_base"
        _run_single_inference(
            data_path=str(mcq_data),
            output_dir=str(candidate_dir),
            gpus=gpus,
            model=model,
            adapter=None,
            repair=repair,
            adjudicate_candidates=adjudicate_candidates,
            adjudicator_max_tokens=adjudicator_max_tokens,
            prompt_style=prompt_style,
            mcq_n=mcq_n,
            free_n=free_n,
            free_temperature=free_temperature,
            mcq_temperature=mcq_temperature,
        )
        selector_specs.append(f"base:{candidate_dir / result_name}")

    seen_names = {"baseline", "base"}
    for spec in adapters:
        name, adapter_path = _parse_named_path(spec)
        name = _safe_name(name)
        if name in seen_names:
            suffix = 2
            original = name
            while f"{original}_{suffix}" in seen_names:
                suffix += 1
            name = f"{original}_{suffix}"
        seen_names.add(name)

        candidate_dir = output_dir / f"mcq_vote_{name}"
        _run_single_inference(
            data_path=str(mcq_data),
            output_dir=str(candidate_dir),
            gpus=gpus,
            model=model,
            adapter=adapter_path,
            repair=repair,
            adjudicate_candidates=adjudicate_candidates,
            adjudicator_max_tokens=adjudicator_max_tokens,
            prompt_style=prompt_style,
            mcq_n=mcq_n,
            free_n=free_n,
            free_temperature=free_temperature,
            mcq_temperature=mcq_temperature,
        )
        selector_specs.append(f"{name}:{candidate_dir / result_name}")

    selected = output_dir / "majority_selected.jsonl"
    cmd = [
        sys.executable,
        str(REPO_DIR / "select_results.py"),
        "--data",
        str(data_path),
        "--output",
        str(selected),
        "--mcq-only",
    ]
    for spec in selector_specs:
        cmd.extend(["--result", spec])
    _run_checked(cmd)
    return selected



SUBMISSION_VARIANTS = (
    "learned_selector_precision_exact_v1",
    "mcq_shuffle_all_conservative_free_v1",
)

VARIANT_DEFAULT_OUTPUT_DIRS = {
    "learned_selector_precision_exact_v1": "results/private_learned_selector_precision_exact_v1",
    "mcq_shuffle_all_conservative_free_v1": "results/private_mcq_shuffle_all_conservative_free_v1",
}

LEARNED_SELECTOR_PRECISION_EXACT_PUBLIC_RESULTS = [
    "majority5_strict:results/public_majority5_mcq_strict_repair/scored.jsonl",
    "majority5:results/public_majority5_mcq/scored.jsonl",
    "free_rules:results/public_majority5_mcq_free_rules/scored.jsonl",
    "hybrid:results/public_hybrid_union_mcq_trace_free/scored.jsonl",
    "trace:results/public_lora_trace_eval/scored.jsonl",
    "union:results/public_lora_union_eval/scored.jsonl",
    "opt3:results/two_gpu_opt3/scored.jsonl",
    "bf16:results/two_gpu_bf16/scored.jsonl",
    "m5_targeted:results/public_lora_majority5_targeted_eval/scored.jsonl",
    "strict_new_lora:results/public_strict_mcq_new_lora_free/scored.jsonl",
    "strict_new_lora_badonly:results/public_strict_mcq_new_lora_free_badonly/scored.jsonl",
    "precision_v1:results/public_precision_v1/scored.jsonl",
    "precision:results/public_precision_mcq_vote_precision_free/scored.jsonl",
    "full_precision:results/public_candidate_vote_full_available_precision/scored.jsonl",
    "precision_dup:results/public_candidate_vote_precision_mcq_free_dup_ultra/scored.jsonl",
    "exact_free:results/public_strict_mcq_exact_free_v1/scored.jsonl",
]

LEARNED_SELECTOR_PRECISION_EXACT_PRIVATE_RESULTS = [
    "majority5_strict:results/private_majority5_mcq_strict_repair/repaired.jsonl",
    "majority5:results/private_majority5_mcq/selected.jsonl",
    "free_rules:results/private_majority5_mcq_free_rules/selected.jsonl",
    "hybrid:results/private_hybrid_union_mcq_trace_free/repaired.jsonl",
    "trace:results/private_lora_trace_final/repaired.jsonl",
    "union:results/private_lora_union_final/repaired.jsonl",
    "opt3:results/private_4b_opt3/repaired.jsonl",
    "bf16:results/private_4b/repaired.jsonl",
    "m5_targeted:results/private_lora_majority5_targeted_eval/repaired.jsonl",
    "strict_new_lora:results/private_strict_mcq_new_lora_free/selected.jsonl",
    "strict_new_lora_badonly:results/private_strict_mcq_new_lora_free_badonly/selected.jsonl",
    "precision_v1:results/private_precision_v1/repaired.jsonl",
    "precision:results/private_precision_mcq_vote_precision_free/selected.jsonl",
    "full_precision:results/private_candidate_vote_full_available_precision/selected.jsonl",
    "precision_dup:results/private_candidate_vote_precision_mcq_free_dup_ultra/selected.jsonl",
    "exact_free:results/private_strict_mcq_exact_free_v1/repaired.jsonl",
]


def _resolve_data_path(data_path: str) -> Path:
    data = Path(data_path)
    if not data.is_absolute():
        cwd_data = Path.cwd() / data
        repo_data = REPO_DIR / data
        data = cwd_data if cwd_data.exists() else repo_data
    if not data.exists():
        raise FileNotFoundError(f"Input dataset not found: {data}")
    return data.resolve()


def _resolve_output_dir(output_dir: str, variant: str | None = None) -> Path:
    if variant and output_dir == "results/private_4b":
        output_dir = VARIANT_DEFAULT_OUTPUT_DIRS[variant]
    out = Path(output_dir)
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _spec_path(spec: str) -> Path:
    path = Path(spec.rsplit(":", 1)[1])
    return path if path.is_absolute() else REPO_DIR / path


def _abs_result_spec(spec: str) -> str:
    name, path = spec.split(":", 1)
    result_path = Path(path)
    if not result_path.is_absolute():
        result_path = REPO_DIR / result_path
    return f"{name}:{result_path}"


def _require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _validate_variant_inputs(specs: Sequence[str]) -> None:
    missing = [_spec_path(spec) for spec in specs if not _spec_path(spec).exists()]
    if missing:
        shown = "\n".join(str(path) for path in missing[:12])
        extra = "" if len(missing) <= 12 else f"\n... and {len(missing) - 12} more"
        raise FileNotFoundError(f"Missing candidate source files for submission variant:\n{shown}{extra}")


def _write_submission_from_results(data: Path, results: Path, submission: Path) -> None:
    _run_checked(
        [
            sys.executable,
            str(REPO_DIR / "make_submission.py"),
            "--data",
            str(data),
            "--results",
            str(results),
            "--output",
            str(submission),
        ]
    )


def _run_learned_selector_precision_exact_v1(data: Path, out: Path, submission_path: str | None) -> str:
    _validate_variant_inputs(
        LEARNED_SELECTOR_PRECISION_EXACT_PUBLIC_RESULTS
        + LEARNED_SELECTOR_PRECISION_EXACT_PRIVATE_RESULTS
    )
    public_data = _require_path(REPO_DIR / "public.jsonl", "public selector training data")
    cmd = [
        sys.executable,
        str(REPO_DIR / "learned_candidate_selector.py"),
        "--public-data",
        str(public_data),
        "--private-data",
        str(data),
        "--output-dir",
        str(out),
        "--pool",
        "precision",
        "--seeds",
        "100",
        "--train-steps",
        "600",
        "--learning-rate",
        "0.08",
        "--l2",
        "0.02",
    ]
    for spec in LEARNED_SELECTOR_PRECISION_EXACT_PUBLIC_RESULTS:
        cmd.extend(["--public-result", _abs_result_spec(spec)])
    for spec in LEARNED_SELECTOR_PRECISION_EXACT_PRIVATE_RESULTS:
        cmd.extend(["--private-result", _abs_result_spec(spec)])
    _run_checked(cmd)

    selected = _require_path(out / "selected.jsonl", "learned-selector output")
    submission = Path(submission_path) if submission_path else out / "submission.csv"
    if not submission.is_absolute():
        submission = (Path.cwd() / submission).resolve()
    _write_submission_from_results(data, selected, submission)
    print(f"Submission CSV: {submission}", flush=True)
    return str(submission)


def _run_mcq_shuffle_all_conservative_free_v1(data: Path, out: Path, submission_path: str | None) -> str:
    learned_dir = out / "learned_selector_precision_exact_v1"
    _run_learned_selector_precision_exact_v1(data, learned_dir, None)
    learned_selected = _require_path(learned_dir / "selected.jsonl", "learned-selector baseline")

    shuffle_source = _require_path(
        REPO_DIR / "results/private_shuffle_free_sc_v1/mcq_shuffle_source.jsonl",
        "MCQ option-shuffle source",
    )
    conservative_free = _require_path(
        REPO_DIR / "results/private_conservative_delta_free_v1/selected.jsonl",
        "conservative free-form overlay source",
    )

    overlay_dir = out / "mcq_shuffle_overlay_all_v1"
    overlay_selected = overlay_dir / "selected.jsonl"
    _run_checked(
        [
            sys.executable,
            str(REPO_DIR / "mcq_shuffle_confidence_overlay.py"),
            "--data",
            str(data),
            "--baseline",
            str(learned_selected),
            "--shuffle",
            str(shuffle_source),
            "--output",
            str(overlay_selected),
            "--submission",
            str(overlay_dir / "submission.csv"),
            "--report",
            str(overlay_dir / "report.json"),
            "--min-top",
            "1",
            "--min-margin",
            "0",
        ]
    )

    selected = out / "selected.jsonl"
    _run_checked(
        [
            sys.executable,
            str(REPO_DIR / "combine_results_by_format.py"),
            "--mcq-results",
            str(overlay_selected),
            "--free-results",
            str(conservative_free),
            "--output",
            str(selected),
        ]
    )

    submission = Path(submission_path) if submission_path else out / "submission.csv"
    if not submission.is_absolute():
        submission = (Path.cwd() / submission).resolve()
    _write_submission_from_results(data, selected, submission)
    print(f"Submission CSV: {submission}", flush=True)
    return str(submission)


def _run_submission_variant(
    *,
    variant: str,
    data_path: str,
    output_dir: str,
    submission_path: str | None,
    model: str,
) -> str:
    enforce_competition_model(model)
    if variant not in SUBMISSION_VARIANTS:
        raise ValueError(f"Unknown submission variant {variant!r}; expected one of {SUBMISSION_VARIANTS}")
    data = _resolve_data_path(data_path)
    out = _resolve_output_dir(output_dir, variant)
    if variant == "learned_selector_precision_exact_v1":
        return _run_learned_selector_precision_exact_v1(data, out, submission_path)
    if variant == "mcq_shuffle_all_conservative_free_v1":
        return _run_mcq_shuffle_all_conservative_free_v1(data, out, submission_path)
    raise AssertionError(f"Unhandled submission variant: {variant}")

def run_inference(
    data_path: str = "private.jsonl",
    output_dir: str = "results/private_4b",
    submission_path: str | None = None,
    gpus: str | Sequence[str] = ("0", "1"),
    model: str = MODEL_ID,
    adapter: str | None = ADAPTER_ID or None,
    repair: bool = True,
    adjudicate_candidates: bool = ADJUDICATE_CANDIDATES,
    adjudicator_max_tokens: int = ADJUDICATOR_MAX_TOKENS,
    prompt_style: str = PROMPT_STYLE,
    finalize_all_free: bool = FINALIZE_ALL_FREE,
    mcq_n: int = MCQ_N,
    free_n: int = FREE_N,
    free_temperature: float = FREE_TEMPERATURE,
    mcq_temperature: float = MCQ_TEMPERATURE,
) -> str:
    """Run the full competition inference pipeline and return the submission CSV path."""
    enforce_competition_model(model)
    data = Path(data_path)
    if not data.is_absolute():
        cwd_data = Path.cwd() / data
        repo_data = REPO_DIR / data
        data = cwd_data if cwd_data.exists() else repo_data
    if not data.exists():
        raise FileNotFoundError(f"Input dataset not found: {data}")
    data = data.resolve()

    if isinstance(gpus, str):
        gpu_list = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    else:
        gpu_list = [str(gpu).strip() for gpu in gpus if str(gpu).strip()]
    if not gpu_list:
        raise ValueError("At least one GPU id must be provided.")

    records = _load_jsonl(data)
    if not records:
        raise ValueError(f"Input dataset is empty: {data}")

    out = Path(output_dir)
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    include_gold = _has_gold(data)
    active_gpu_list = gpu_list[: min(len(gpu_list), len(records))]
    ranges = _shard_ranges(len(records), len(active_gpu_list))
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    procs: list[tuple[subprocess.Popen, Path]] = []
    shard_paths: list[Path] = []
    for shard_idx, (gpu, (start, end)) in enumerate(zip(active_gpu_list, ranges)):
        shard_path = out / f"gpu{shard_idx}.jsonl"
        shard_paths.append(shard_path)
        log_path = out / f"gpu{shard_idx}.log"
        cmd = [
            sys.executable,
            str(REPO_DIR / "infer_vllm.py"),
            "--data",
            str(data),
            "--model",
            model,
            "--gpu",
            gpu,
            "--start",
            str(start),
            "--end",
            str(end),
            "--output",
            str(shard_path),
            "--mcq-n",
            str(mcq_n),
            "--free-n",
            str(free_n),
            "--free-temperature",
            str(free_temperature),
            "--mcq-temperature",
            str(mcq_temperature),
            "--gpu-memory-utilization",
            str(GPU_MEMORY_UTILIZATION),
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--max-num-seqs",
            str(MAX_NUM_SEQS),
            "--max-num-batched-tokens",
            str(MAX_NUM_BATCHED_TOKENS),
            "--max-tokens",
            str(MAX_TOKENS),
            "--mcq-max-tokens",
            str(MCQ_MAX_TOKENS),
            "--finalizer-max-tokens",
            str(FINALIZER_MAX_TOKENS),
            "--adjudicator-max-tokens",
            str(adjudicator_max_tokens),
            "--prompt-style",
            prompt_style,
            "--retry-bad-format",
            "--finalize-answers",
            "--direct-fallback",
        ]
        if finalize_all_free:
            cmd.append("--finalize-all-free")
        if adjudicate_candidates:
            cmd.append("--adjudicate-candidates")
        if adapter:
            cmd.extend(["--lora-adapter", adapter, "--max-lora-rank", str(MAX_LORA_RANK)])
        if USE_BF16_WEIGHTS:
            cmd.append("--no-bnb")
        if include_gold:
            cmd.append("--include-gold")
        print("+ " + " ".join(cmd) + f" > {log_path} 2>&1", flush=True)
        log = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.close()
        procs.append((proc, log_path))

    failures = []
    for proc, log_path in procs:
        code = proc.wait()
        if code != 0:
            failures.append((code, log_path))
    if failures:
        details = "\n\n".join(
            f"{log_path} failed with exit code {code}\n{_tail(log_path)}"
            for code, log_path in failures
        )
        raise RuntimeError(f"Inference shard failure:\n{details}")

    merged = out / "merged.jsonl"
    _run_checked([sys.executable, str(REPO_DIR / "merge_jsonl.py"), "--output", str(merged), *map(str, shard_paths)])

    final_results = merged
    if repair:
        repaired = out / "repaired.jsonl"
        repair_cmd = [
            sys.executable,
            str(REPO_DIR / "repair_bad_format.py"),
            "--data",
            str(data),
            "--results",
            str(merged),
            "--output",
            str(repaired),
            "--gpu",
            gpu_list[0],
            "--model",
            model,
            "--gpu-memory-utilization",
            str(GPU_MEMORY_UTILIZATION),
            "--prompt-style",
            prompt_style,
        ]
        if adapter:
            repair_cmd.extend(["--lora-adapter", adapter, "--max-lora-rank", str(MAX_LORA_RANK)])
        if USE_BF16_WEIGHTS:
            repair_cmd.append("--no-bnb")
        _run_checked(repair_cmd, log_path=out / "repair.log", env=env)
        final_results = repaired

    submission = Path(submission_path) if submission_path else out / "submission.csv"
    _run_checked(
        [
            sys.executable,
            str(REPO_DIR / "make_submission.py"),
            "--data",
            str(data),
            "--results",
            str(final_results),
            "--output",
            str(submission),
        ]
    )

    if include_gold:
        _run_checked(
            [
                sys.executable,
                str(REPO_DIR / "evaluate_results.py"),
                str(final_results),
                "--output",
                str(out / "scored.jsonl"),
                "--show-wrong",
                "10",
            ],
            log_path=out / "eval.log",
            env=env,
        )

    print(f"Submission CSV: {submission}", flush=True)
    return str(submission)


_run_single_inference = run_inference


def run_inference(
    data_path: str = "private.jsonl",
    output_dir: str = "results/private_4b",
    submission_path: str | None = None,
    gpus: str | Sequence[str] = ("0", "1"),
    model: str = MODEL_ID,
    variant: str | None = None,
    adapter: str | None = ADAPTER_ID or None,
    mcq_adapter: str | None = MCQ_ADAPTER_ID or None,
    free_adapter: str | None = FREE_ADAPTER_ID or None,
    repair: bool = True,
    adjudicate_candidates: bool = ADJUDICATE_CANDIDATES,
    adjudicator_max_tokens: int = ADJUDICATOR_MAX_TOKENS,
    prompt_style: str = PROMPT_STYLE,
    finalize_all_free: bool = FINALIZE_ALL_FREE,
    mcq_majority_adapters: Sequence[str] | None = None,
    mcq_majority_include_base: bool = False,
    mcq_n: int = MCQ_N,
    free_n: int = FREE_N,
    free_temperature: float = FREE_TEMPERATURE,
    mcq_temperature: float = MCQ_TEMPERATURE,
) -> str:
    """Run the full pipeline, optionally using named final-submission variants."""
    if variant:
        return _run_submission_variant(
            variant=variant,
            data_path=data_path,
            output_dir=output_dir,
            submission_path=submission_path,
            model=model,
        )

    if not mcq_adapter and not free_adapter and not mcq_majority_adapters and not mcq_majority_include_base:
        return _run_single_inference(
            data_path=data_path,
            output_dir=output_dir,
            submission_path=submission_path,
            gpus=gpus,
            model=model,
            adapter=adapter,
            repair=repair,
            adjudicate_candidates=adjudicate_candidates,
            adjudicator_max_tokens=adjudicator_max_tokens,
            prompt_style=prompt_style,
            finalize_all_free=finalize_all_free,
            mcq_n=mcq_n,
            free_n=free_n,
            free_temperature=free_temperature,
            mcq_temperature=mcq_temperature,
        )

    enforce_competition_model(model)
    data = Path(data_path)
    if not data.is_absolute():
        cwd_data = Path.cwd() / data
        repo_data = REPO_DIR / data
        data = cwd_data if cwd_data.exists() else repo_data
    if not data.exists():
        raise FileNotFoundError(f"Input dataset not found: {data}")
    data = data.resolve()

    out = Path(output_dir)
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    mcq_run_dir = out / "mcq_run"
    free_run_dir = out / "free_run"
    mcq_adapter = mcq_adapter or adapter
    free_adapter = free_adapter or adapter

    print(f"Hybrid inference: MCQ adapter={mcq_adapter or 'base'}, free-form adapter={free_adapter or 'base'}", flush=True)
    _run_single_inference(
        data_path=str(data),
        output_dir=str(mcq_run_dir),
        gpus=gpus,
        model=model,
        adapter=mcq_adapter,
        repair=repair,
        adjudicate_candidates=adjudicate_candidates,
        adjudicator_max_tokens=adjudicator_max_tokens,
        prompt_style=prompt_style,
        finalize_all_free=finalize_all_free,
        mcq_n=mcq_n,
        free_n=free_n,
        free_temperature=free_temperature,
        mcq_temperature=mcq_temperature,
    )
    _run_single_inference(
        data_path=str(data),
        output_dir=str(free_run_dir),
        gpus=gpus,
        model=model,
        adapter=free_adapter,
        repair=repair,
        adjudicate_candidates=adjudicate_candidates,
        adjudicator_max_tokens=adjudicator_max_tokens,
        prompt_style=prompt_style,
        finalize_all_free=finalize_all_free,
        mcq_n=mcq_n,
        free_n=free_n,
        free_temperature=free_temperature,
        mcq_temperature=mcq_temperature,
    )

    result_name = "repaired.jsonl" if repair else "merged.jsonl"
    hybrid_results = out / result_name
    _run_checked(
        [
            sys.executable,
            str(REPO_DIR / "combine_results_by_format.py"),
            "--mcq-results",
            str(mcq_run_dir / result_name),
            "--free-results",
            str(free_run_dir / result_name),
            "--output",
            str(hybrid_results),
        ]
    )

    final_results = _append_mcq_majority_selection(
        data_path=data,
        output_dir=out,
        baseline_results=hybrid_results,
        gpus=gpus,
        model=model,
        repair=repair,
        adjudicate_candidates=adjudicate_candidates,
        adjudicator_max_tokens=adjudicator_max_tokens,
        prompt_style=prompt_style,
        mcq_n=mcq_n,
        free_n=free_n,
        free_temperature=free_temperature,
        mcq_temperature=mcq_temperature,
        mcq_majority_adapters=mcq_majority_adapters,
        mcq_majority_include_base=mcq_majority_include_base,
    )

    submission = Path(submission_path) if submission_path else out / "submission.csv"
    _run_checked(
        [
            sys.executable,
            str(REPO_DIR / "make_submission.py"),
            "--data",
            str(data),
            "--results",
            str(final_results),
            "--output",
            str(submission),
        ]
    )

    if _has_gold(data):
        _run_checked(
            [
                sys.executable,
                str(REPO_DIR / "evaluate_results.py"),
                str(final_results),
                "--output",
                str(out / "scored.jsonl"),
                "--show-wrong",
                "10",
            ],
            log_path=out / "eval.log",
            env=os.environ.copy(),
        )

    print(f"Submission CSV: {submission}", flush=True)
    return str(submission)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Qwen3-4B competition inference pipeline.")
    parser.add_argument("--data", default="private.jsonl")
    parser.add_argument("--output-dir", default="results/private_4b")
    parser.add_argument("--submission", default=None)
    parser.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU ids.")
    parser.add_argument(
        "--variant",
        choices=SUBMISSION_VARIANTS,
        default=None,
        help="Named final-submission workflow to rebuild a selected private CSV.",
    )
    parser.add_argument("--adapter", default=ADAPTER_ID or None, help="Optional LoRA/QLoRA adapter path or HF repo id.")
    parser.add_argument("--mcq-adapter", default=MCQ_ADAPTER_ID or None, help="Optional adapter used only for MCQ rows; enables hybrid inference.")
    parser.add_argument("--free-adapter", default=FREE_ADAPTER_ID or None, help="Optional adapter used only for free-form rows; enables hybrid inference.")
    parser.add_argument("--adjudicate-candidates", action="store_true", default=ADJUDICATE_CANDIDATES, help="Use same-model adjudication for ambiguous self-consistency candidates.")
    parser.add_argument("--adjudicator-max-tokens", type=int, default=ADJUDICATOR_MAX_TOKENS)
    parser.add_argument("--mcq-n", type=int, default=MCQ_N, help="Total MCQ candidates per row, including deterministic pass.")
    parser.add_argument("--free-n", type=int, default=FREE_N, help="Total free-form candidates per row, including deterministic pass.")
    parser.add_argument("--free-temperature", type=float, default=FREE_TEMPERATURE)
    parser.add_argument("--mcq-temperature", type=float, default=MCQ_TEMPERATURE)
    parser.add_argument(
        "--prompt-style",
        choices=["default", "exact_free"],
        default=PROMPT_STYLE,
        help="Prompt family to use for generation and repair.",
    )
    parser.add_argument(
        "--finalize-all-free",
        action="store_true",
        default=FINALIZE_ALL_FREE,
        help="Run final-answer formatting over every free-form response.",
    )
    parser.add_argument(
        "--mcq-majority-include-base",
        action="store_true",
        help="Add a base-model MCQ-only candidate run and majority-vote MCQ final answers.",
    )
    parser.add_argument(
        "--mcq-majority-adapter",
        action="append",
        default=[],
        help="Add an MCQ-only adapter candidate as name:path and majority-vote MCQ final answers. Repeatable.",
    )
    parser.add_argument("--no-repair", action="store_true", help="Skip structured bad-format repair.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_inference(
        data_path=args.data,
        output_dir=args.output_dir,
        submission_path=args.submission,
        gpus=tuple(gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()),
        variant=args.variant,
        adapter=args.adapter,
        mcq_adapter=args.mcq_adapter,
        free_adapter=args.free_adapter,
        repair=not args.no_repair,
        adjudicate_candidates=args.adjudicate_candidates,
        adjudicator_max_tokens=args.adjudicator_max_tokens,
        prompt_style=args.prompt_style,
        finalize_all_free=args.finalize_all_free,
        mcq_n=args.mcq_n,
        free_n=args.free_n,
        free_temperature=args.free_temperature,
        mcq_temperature=args.mcq_temperature,
        mcq_majority_adapters=args.mcq_majority_adapter,
        mcq_majority_include_base=args.mcq_majority_include_base,
    )


if __name__ == "__main__":
    main()
