import argparse
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd

from adapters.ollama_adapter import OllamaAdapter
from metrics.bleu_metric import BLEUMetric, RougeMetric, BertScoreMetric
from metrics.faithfulness_jaccard import FaithfulnessJaccard
from runner.experiment_runner import ExperimentRunner
from runner.bias_runner import run_bias_test_for_runner
from runner.occupation_bias_runner import run_occupation_gender_bias
from runner.composite_scorer import compute_composite_scores
from runner.medical_bias_runner import run_medical_bias
from utils.run_metadata import save_run_metadata


DATA_DIR = Path("data_pdfs")
OUTPUT_DIR = Path("results")
INDIVIDUAL_DIR = OUTPUT_DIR / "individual_runs"


def sanitize_model_name(name: str) -> str:
    return name.replace(":", "_").replace("/", "_").replace(" ", "_")


MODEL_REGISTRY = {
    "mistral:7b": lambda: OllamaAdapter("mistral:7b", temperature=0.0),
    "phi:2.7b": lambda: OllamaAdapter("phi:2.7b", temperature=0.0),
    "deepseek-r1:8b": lambda: OllamaAdapter("deepseek-r1:8b", temperature=0.0),
    "gpt-oss:20b": lambda: OllamaAdapter("gpt-oss:20b", temperature=0.0),
    "qwen3:4b": lambda: OllamaAdapter("qwen3:4b", temperature=0.0),
    "gemma3:4b": lambda: OllamaAdapter("gemma3:4b", temperature=0.0),

    
}


ARCHIVE_PATTERNS = [
    "benchmark_results.csv",
    "benchmark_summary.csv",
    "composite_scores.csv",
    "bias_results.csv",
    "bias_samples.csv",
    "occ_bias_summary.csv",
    "occ_bias_per_occ.csv",
    "occ_bias_samples.csv",
    "occ_bias.sqlite",
    "occ_bias_index_*.png",
    "occ_pronoun_heatmap_*.png",
    "medical_bias_summary.csv",
    "medical_bias_per_category.csv",
    "medical_bias_per_type.csv",
    "medical_bias_items.csv",
    "medical_bias.sqlite",
    "run_metadata.json",
    "latency_vs_*.png",
]


def archive_run_outputs(selected_models):
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    tag = "-".join(sanitize_model_name(m) for m in selected_models)
    run_dir = INDIVIDUAL_DIR / f"{timestamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for pattern in ARCHIVE_PATTERNS:
        for path in OUTPUT_DIR.glob(pattern):
            dest = run_dir / path.name
            shutil.copy2(path, dest)
            copied += 1

    if copied == 0:
        print("No output files found to archive.")
    else:
        print(f"Archived {copied} files to {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark and bias-evaluate selected LLMs.")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODEL_REGISTRY.keys()),
        help="Comma-separated model names to run (default: all known models)",
    )
    parser.add_argument(
        "--archive-run",
        action="store_true",
        help="Archive all output files for this run under results/individual_runs/",
    )
    args = parser.parse_args()

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    if not requested:
        print("No models specified. Exiting.")
        return

    missing = [m for m in requested if m not in MODEL_REGISTRY]
    if missing:
        print(f"Unknown models requested: {missing}\nKnown models: {list(MODEL_REGISTRY.keys())}")
        return

    runner = ExperimentRunner(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)
    for model_name in requested:
        runner.register_model(model_name, MODEL_REGISTRY[model_name]())

    if not runner.models:
        print("No models registered. Exiting.")
        return

    save_run_metadata(OUTPUT_DIR, runner)

    # Optional quick bias probe before main benchmark
    run_bias_test_for_runner(runner)

    runner.register_metric(BLEUMetric())
    runner.register_metric(RougeMetric())
    runner.register_metric(BertScoreMetric())
    runner.register_metric(FaithfulnessJaccard())

    runner.run_all()

    compute_composite_scores(OUTPUT_DIR / "benchmark_results.csv", OUTPUT_DIR)

    run_occupation_gender_bias(runner, repeats=1, save_csv=True, save_sqlite=True, save_plots=True)

    medical_csv = Path("data/Implicit and Explicit/Bias_dataset.csv")
    if medical_csv.exists():
        run_medical_bias(runner, medical_csv, repeats=1, save_csv=True, save_sqlite=True)
    else:
        print("Medical bias dataset not found at data/Implicit and Explicit/Bias_dataset.csv")

    if args.archive_run:
        archive_run_outputs(requested)


if __name__ == "__main__":
    main()
