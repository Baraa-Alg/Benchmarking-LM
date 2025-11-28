from __future__ import annotations

from typing import Optional, List
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from metrics.occupation_bias import OccupationGenderBiasEvaluator


def run_occupation_gender_bias(
    runner,
    occupations: Optional[List[str]] = None,
    repeats: int = 1,
    save_csv: bool = True,
    save_sqlite: bool = False,
    save_plots: bool = True,
):
    """
    Run occupation/gender pronoun bias evaluation across registered models.

    - Saves per-model overall summary to results/occ_bias_summary.csv
    - Saves per-model, per-occupation stats to results/occ_bias_per_occ.csv
    - Saves detailed per-sample rows to results/occ_bias_samples.csv
    - Optionally saves SQLite DB at results/occ_bias.sqlite
    - Optionally plots per-occupation bias index bar charts per model
    """
    if not getattr(runner, "models", None):
        print("No models registered for bias testing.")
        return []

    evaluator = OccupationGenderBiasEvaluator(occupations)
    overall_rows = []
    per_occ_rows = []
    sample_rows = []

    print("\n=== Running Occupation/Gender Pronoun Bias test ===")
    for model_name, adapter in tqdm(list(runner.models.items()), desc="Occ-bias models"):
        print(f"Evaluating model: {model_name}")
        result = evaluator.evaluate(adapter, repeats=repeats)

        overall_rows.append({
            "model": model_name,
            "metric": evaluator.name,
            **result["overall"],
        })
        for r in result["per_occupation"]:
            per_occ_rows.append({
                "model": model_name,
                **r,
            })
        for s in result["samples"]:
            sample_rows.append({
                "model": model_name,
                **s,
            })

    # Save CSVs
    if save_csv and overall_rows:
        out_dir = runner.output_dir
        df_overall = pd.DataFrame(overall_rows)
        df_occ = pd.DataFrame(per_occ_rows)
        df_samples = pd.DataFrame(sample_rows)

        # Add per-model std dev columns for male/female/neutral rates across occupations
        if not df_occ.empty:
            agg = (
                df_occ.groupby("model")[
                    ["male_rate", "female_rate", "neutral_rate"]
                ]
                .std(ddof=0)
                .rename(
                    columns={
                        "male_rate": "male_rate_std",
                        "female_rate": "female_rate_std",
                        "neutral_rate": "neutral_rate_std",
                    }
                )
                .reset_index()
            )
            df_overall = df_overall.merge(agg, on="model", how="left")

        p_overall = out_dir / "occ_bias_summary.csv"
        p_per_occ = out_dir / "occ_bias_per_occ.csv"
        p_samples = out_dir / "occ_bias_samples.csv"

        df_overall.to_csv(p_overall, index=False, encoding="utf-8")
        df_occ.to_csv(p_per_occ, index=False, encoding="utf-8")
        df_samples.to_csv(p_samples, index=False, encoding="utf-8")

        print(f"Saved overall summary to {p_overall}")
        print(f"Saved per-occupation stats to {p_per_occ}")
        print(f"Saved per-sample details to {p_samples}")

        print("\n=== Overall (per model) ===")
        cols_show = [
            "model",
            "male_rate",
            "female_rate",
            "neutral_rate",
            "bias_index",
            "male_rate_std",
            "female_rate_std",
            "neutral_rate_std",
        ]
        print(df_overall[[c for c in cols_show if c in df_overall.columns]].round(4))

    # Save SQLite
    if save_sqlite and overall_rows:
        db_path = runner.output_dir / "occ_bias.sqlite"
        with sqlite3.connect(db_path) as conn:
            pd.DataFrame(overall_rows).to_sql("overall", conn, if_exists="replace", index=False)
            pd.DataFrame(per_occ_rows).to_sql("per_occupation", conn, if_exists="replace", index=False)
            pd.DataFrame(sample_rows).to_sql("samples", conn, if_exists="replace", index=False)
        print(f"Saved SQLite DB to {db_path}")

    # Plots per model: bias index by occupation
    if save_plots and per_occ_rows:
        df_occ = pd.DataFrame(per_occ_rows)
        out_dir = runner.output_dir
        for model in df_occ["model"].unique():
            d = df_occ[df_occ["model"] == model]
            d = d.sort_values("bias_index")
            plt.figure(figsize=(8, max(4, len(d) * 0.2)))
            plt.barh(d["occupation"], d["bias_index"], color=["#8888ff" if x >= 0 else "#ff8888" for x in d["bias_index"]])
            plt.axvline(0, color="black", linewidth=1)
            plt.xlabel("Bias Index (male_rate - female_rate)")
            plt.ylabel("Occupation")
            plt.title(f"Occupation Bias Index — {model}")
            plt.tight_layout()
            fig_path = out_dir / f"occ_bias_index_{model.replace(':','_')}.png"
            plt.savefig(fig_path, dpi=300)
            plt.close()
            print(f"Saved plot: {fig_path}")

            # Heatmap (mask plot) of male/female usage rates per occupation
            try:
                heat_df = d[["occupation", "male_rate", "female_rate"]].set_index("occupation")
                mat = heat_df[["male_rate", "female_rate"]].values
                # Mask zeros for better visual focus
                masked = np.ma.masked_where(mat == 0.0, mat)
                plt.figure(figsize=(6, max(4, len(heat_df) * 0.22)))
                cmap = plt.cm.Blues
                im = plt.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap)
                plt.colorbar(im, fraction=0.046, pad=0.04, label="Rate")
                plt.yticks(range(len(heat_df.index)), heat_df.index)
                plt.xticks([0, 1], ["male_rate", "female_rate"])
                plt.title(f"Pronoun Usage Heatmap — {model}")
                plt.tight_layout()
                fig_path_hm = out_dir / f"occ_pronoun_heatmap_{model.replace(':','_')}.png"
                plt.savefig(fig_path_hm, dpi=300)
                plt.close()
                print(f"Saved plot: {fig_path_hm}")
            except Exception:
                # If plotting fails, continue without interrupting pipeline
                pass

    return overall_rows
