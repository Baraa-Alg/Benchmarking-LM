import re
import warnings
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Some weights of RobertaModel")


class ExperimentRunner:
    """Runs benchmarking experiments with modular adapters and metrics."""

    def __init__(self, data_dir: Path, output_dir: Path = Path("results")):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.models = {}
        self.metrics = []
        self.results = []

        self.output_dir.mkdir(exist_ok=True)

    # ------------------------------
    # Registration
    # ------------------------------
    def register_model(self, name, adapter):
        self.models[name] = adapter

    def register_metric(self, metric):
        self.metrics.append(metric)

    # ------------------------------
    # Utilities
    # ------------------------------
    @staticmethod
    def extract_pdf_sections(pdf_path: Path):
        doc = fitz.open(pdf_path)
        text = " ".join([page.get_text("text") for page in doc])
        text = re.sub(r"\s+", " ", text)

        match = re.search(
            r"(?is)\babstract\b[:\s\n]*(.+?)(?=\b(?:introduction|background|keywords|methods|1\.|materials)\b)",
            text,
        )
        abstract = match.group(1).strip() if match else None
        body = text.replace(abstract, "") if abstract else text
        return abstract, body.strip()

    # ------------------------------
    # Main run loop
    # ------------------------------
    def run_all(self):
        pdf_files = list(self.data_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDFs found in {self.data_dir}")
            return

        for pdf in tqdm(pdf_files, desc="PDFs"):
            print(f"\nProcessing {pdf.name}")
            abstract, body = self.extract_pdf_sections(pdf)
            if not body:
                print("No text found, skipping.")
                continue

            for model_name, adapter in tqdm(list(self.models.items()), desc="Models", leave=False):
                print(f"Running model: {model_name}")
                prompt = (
                    "You are a professional researcher. "
                    "Read the following academic paper text and write a concise, formal abstract (150-250 words) "
                    "summarizing the main goal, methods, results, and conclusions. "
                    "Do not include extraneous details or repeat section titles.\n\n"
                    f"Paper text:\n{body[:4000]}\n\nAbstract:"
                )

                generated, latency = adapter.generate(prompt)

                metric_scores = {}
                for metric in self.metrics:
                    metric_scores[metric.name] = metric.compute(abstract, generated)

                row = {
                    "pdf_file": pdf.name,
                    "model": model_name,
                    "latency": latency,
                    "generated_abstract": generated,
                    **metric_scores,
                }
                self.results.append(row)

        self._save_results()

    # ------------------------------
    # Save and visualize
    # ------------------------------
    def _save_results(self):
        if not self.results:
            print("No results to save.")
            return

        df = pd.DataFrame(self.results)
        out_file = self.output_dir / "benchmark_results.csv"
        df.to_csv(out_file, index=False, encoding="utf-8")
        print(f"\nResults saved to {out_file}")

        metric_names = [m.name for m in self.metrics if m.name in df.columns]

        summary_cols = ["latency"] + metric_names
        summary = df.groupby("model")[summary_cols].mean().reset_index()
        print("\n=== Summary ===")
        print(summary.round(4))

        summary_out = self.output_dir / "benchmark_summary.csv"
        summary.to_csv(summary_out, index=False, encoding="utf-8")
        print(f"Summary saved to {summary_out}")

        for metric in metric_names:
            plt.figure(figsize=(6, 4))
            for model in df["model"].unique():
                subset = df[df["model"] == model]
                plt.scatter(subset["latency"], subset[metric], label=model, s=80)
            plt.xlabel("Latency (s)")
            plt.ylabel(metric)
            plt.title(f"Latency vs {metric}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"latency_vs_{metric}.png", dpi=300)
            plt.close()

        print("\nAll metrics processed and plots saved.")
