from __future__ import annotations

import math
import subprocess
import json
from configparser import ConfigParser
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
from statsmodels.stats.multitest import fdrcorrection

from .config import PipelineConfig


TRV_COLUMNS = {
    0: "time",
    2: "rev_before",
    3: "no_rev",
    4: "stim_rev",
    7: "dist",
    8: "dist_std",
    9: "dist_stderr",
    11: "dist_0th",
    12: "dist_1st",
    13: "dist_2nd",
    14: "dist_3rd",
    15: "dist_100th",
    18: "dura",
    19: "dura_std",
    20: "dura_stderr",
    22: "dura_0th",
    23: "dura_1st",
    24: "dura_2nd",
    25: "dura_3rd",
    26: "dura_100th",
}

DAT_COLUMNS = {
    0: "Time",
    1: "n",
    2: "Number",
    3: "Speed",
    4: "Interval Speed",
    5: "Bias",
    6: "Tap",
    7: "Puff",
    8: "x",
    9: "y",
    10: "Morphwidth",
    11: "Midline",
    12: "Area",
    13: "Angular Speed",
    14: "Aspect Ratio",
    15: "Kink",
    16: "Curve",
    17: "Crab",
    18: "Pathlength",
}

BASELINE_TSTAT_METRICS = [
    "Morphwidth",
    "Midline",
    "Area",
    "Aspect Ratio",
    "Speed",
    "Angular Speed",
    "Bias",
    "Kink",
    "Curve",
    "Crab",
    "Pathlength",
]

TAP_TSTAT_METRICS = [
    "init_dura",
    "init_prob",
    "init_speed",
    "final_dura",
    "final_prob",
    "final_speed",
    "habit_dura",
    "habit_prob",
    "habit_speed",
    "recovery_dura",
    "recovery_prob",
    "recovery_speed",
    "memory_retention_dura",
    "memory_retention_prob",
    "memory_retention_speed",
    "init_psa_speed",
    "final_psa_speed",
    "recov_psa_speed",
    "max_psa_speed",
    "mean_psa_speed",
    "sensitization_psa_speed",
    "habit_psa_speed",
    "recovery_psa_speed",
    "memory_retention_psa_speed",
    "init_psa_bias",
    "final_psa_bias",
    "recov_psa_bias",
    "max_psa_bias",
    "mean_psa_bias",
    "sensitization_psa_bias",
    "habit_psa_bias",
    "recovery_psa_bias",
    "memory_retention_psa_bias",
    "init_psa_aspect_ratio",
    "final_psa_aspect_ratio",
    "recov_psa_aspect_ratio",
    "max_psa_aspect_ratio",
    "mean_psa_aspect_ratio",
    "sensitization_psa_aspect_ratio",
    "habit_psa_aspect_ratio",
    "recovery_psa_aspect_ratio",
    "memory_retention_psa_aspect_ratio",
    "init_psa_kink",
    "final_psa_kink",
    "recov_psa_kink",
    "max_psa_kink",
    "mean_psa_kink",
    "sensitization_psa_kink",
    "habit_psa_kink",
    "recovery_psa_kink",
    "memory_retention_psa_kink",
    "init_psa_curve",
    "final_psa_curve",
    "recov_psa_curve",
    "max_psa_curve",
    "mean_psa_curve",
    "sensitization_psa_curve",
    "habit_psa_curve",
    "recovery_psa_curve",
    "memory_retention_psa_curve",
    "init_psa_crab",
    "final_psa_crab",
    "recov_psa_crab",
    "max_psa_crab",
    "mean_psa_crab",
    "sensitization_psa_crab",
    "habit_psa_crab",
    "recovery_psa_crab",
    "memory_retention_psa_crab",
]

MSD_RENAMES = {
    "habit_dura": "Habituation of Response Duration",
    "habit_prob": "Habituation of Response Probability",
    "habit_speed": "Habituation of Response Speed",
    "habit_psa_speed": "Habituation of PSA Speed",
    "habit_psa_bias": "Habituation of PSA Bias",
    "habit_psa_aspect_ratio": "Habituation of PSA Aspect Ratio",
    "habit_psa_kink": "Habituation of PSA Kink",
    "habit_psa_curve": "Habituation of PSA Curve",
    "habit_psa_crab": "Habituation of PSA Crab",
    "init_dura": "Initial Response Duration",
    "init_prob": "Initial Response Probability",
    "init_speed": "Initial Response Speed",
    "init_psa_speed": "Initial PSA Speed",
    "init_psa_bias": "Initial PSA Bias",
    "init_psa_aspect_ratio": "Initial PSA Aspect Ratio",
    "init_psa_kink": "Initial PSA Kink",
    "init_psa_curve": "Initial PSA Curve",
    "init_psa_crab": "Initial PSA Crab",
    "final_dura": "Final Response Duration",
    "final_prob": "Final Response Probability",
    "final_speed": "Final Response Speed",
    "final_psa_speed": "Final PSA Speed",
    "final_psa_bias": "Final PSA Bias",
    "final_psa_aspect_ratio": "Final PSA Aspect Ratio",
    "final_psa_kink": "Final PSA Kink",
    "final_psa_curve": "Final PSA Curve",
    "final_psa_crab": "Final PSA Crab",
    "recovery_dura": "Spontaneous Recovery of Response Duration",
    "recovery_prob": "Spontaneous Recovery of Response Probability",
    "recovery_speed": "Spontaneous Recovery of Response Speed",
    "recovery_psa_speed": "Spontaneous Recovery of PSA Speed",
    "recovery_psa_bias": "Spontaneous Recovery of PSA Bias",
    "recovery_psa_aspect_ratio": "Spontaneous Recovery of PSA Aspect Ratio",
    "recovery_psa_kink": "Spontaneous Recovery of PSA Kink",
    "recovery_psa_curve": "Spontaneous Recovery of PSA Curve",
    "recovery_psa_crab": "Spontaneous Recovery of PSA Crab",
    "recov_psa_speed": "Spontaneous Recovery Stimulus of PSA Speed",
    "recov_psa_bias": "Spontaneous Recovery Stimulus of PSA Bias",
    "recov_psa_aspect_ratio": "Spontaneous Recovery Stimulus of PSA Aspect Ratio",
    "recov_psa_kink": "Spontaneous Recovery Stimulus of PSA Kink",
    "recov_psa_curve": "Spontaneous Recovery Stimulus of PSA Curve",
    "recov_psa_crab": "Spontaneous Recovery Stimulus of PSA Crab",
    "memory_retention_dura": "Memory Retention of Response Duration",
    "memory_retention_prob": "Memory Retention of Response Probability",
    "memory_retention_speed": "Memory Retention of Response Speed",
    "memory_retention_psa_speed": "Memory Retention of PSA Speed",
    "memory_retention_psa_bias": "Memory Retention of PSA Bias",
    "memory_retention_psa_aspect_ratio": "Memory Retention of PSA Aspect Ratio",
    "memory_retention_psa_kink": "Memory Retention of PSA Kink",
    "memory_retention_psa_curve": "Memory Retention of PSA Curve",
    "memory_retention_psa_crab": "Memory Retention of PSA Crab",
    "sensitization_psa_speed": "Sensitization of PSA Speed",
    "sensitization_psa_bias": "Sensitization of PSA Bias",
    "sensitization_psa_aspect_ratio": "Sensitization of PSA Aspect Ratio",
    "sensitization_psa_kink": "Sensitization of PSA Kink",
    "sensitization_psa_curve": "Sensitization of PSA Curve",
    "sensitization_psa_crab": "Sensitization of PSA Crab",
    "max_psa_speed": "Peak PSA Speed",
    "max_psa_bias": "Peak PSA Bias",
    "max_psa_aspect_ratio": "Peak PSA Aspect Ratio",
    "max_psa_kink": "Peak PSA Kink",
    "max_psa_curve": "Peak PSA Curve",
    "max_psa_crab": "Peak PSA Crab",
    "mean_psa_speed": "Average PSA Speed",
    "mean_psa_bias": "Average PSA Bias",
    "mean_psa_aspect_ratio": "Average PSA Aspect Ratio",
    "mean_psa_kink": "Average PSA Kink",
    "mean_psa_curve": "Average PSA Curve",
    "mean_psa_crab": "Average PSA Crab",
}

TSTAT_RENAMES = {
    "Habituation of Duration": "Habituation of Response Duration",
    "Habituation of Probability": "Habituation of Response Probability",
    "Habituation of Speed": "Habituation of Response Speed",
    "Initial Duration": "Initial Response Duration",
    "Initial Probability": "Initial Response Probability",
    "Initial Speed": "Initial Response Speed",
    "Final Duration": "Final Response Duration",
    "Final Probability": "Final Response Probability",
    "Final Speed": "Final Response Speed",
    "Recovery Duration": "Spontaneous Recovery of Response Duration",
    "Recovery Probability": "Spontaneous Recovery of Response Probability",
    "Recovery Speed": "Spontaneous Recovery of Response Speed",
    "Memory Retention Duration": "Memory Retention of Response Duration",
    "Memory Retention Probability": "Memory Retention of Response Probability",
    "Memory Retention Speed": "Memory Retention of Response Speed",
}

PSA_RENAMED_COLUMNS = [
    "Speed",
    "Bias",
    "Morphwidth",
    "Midline",
    "Area",
    "Angular Speed",
    "Aspect Ratio",
    "Kink",
    "Curve",
    "Crab",
    "Pathlength",
]

PSA_FEATURE_COLUMNS = [
    "PSA Speed",
    "PSA Bias",
    "PSA Aspect Ratio",
    "PSA Kink",
    "PSA Curve",
    "PSA Crab",
]

POSITIVE_RECOVERY_METRICS = {"PSA Aspect Ratio", "PSA Kink", "PSA Curve", "PSA Crab"}


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.run_chore:
        run_chore_extraction(config)

    tap_output = build_tap_output(config)
    baseline_output, post_stimulus_output = build_baseline_and_psa_outputs(config)
    tap_response_data, tap_feature_data, psa_summarised_data = build_analysis_outputs(
        tap_output=tap_output,
        post_stimulus_output=post_stimulus_output,
        config=config,
    )

    baseline_by_gene = get_output_byplate(baseline_output, dataset_type="baseline", allele=False)
    baseline_by_allele = get_output_byplate(baseline_output, dataset_type="baseline", allele=True)
    tap_by_gene = get_output_byplate(tap_feature_data, dataset_type="tap", allele=False)
    tap_by_allele = get_output_byplate(tap_feature_data, dataset_type="tap", allele=True)

    gene_msd = get_combined_msd(
        baseline_byplate=baseline_by_gene,
        tap_byplate=tap_by_gene,
        by="Gene",
        screen=config.screen,
    )
    allele_msd = get_combined_msd(
        baseline_byplate=baseline_by_allele,
        tap_byplate=tap_by_allele,
        by="dataset",
        screen=config.screen,
    )

    baseline_tstats_gene = pair_pvals(do_ttest(baseline_output, "Gene", "baseline", config.screen))
    baseline_tstats_allele = pair_pvals(do_ttest(baseline_output, "dataset", "baseline", config.screen))
    tap_tstats_gene = pair_pvals(do_ttest(tap_feature_data, "Gene", "tap", config.screen))
    tap_tstats_allele = pair_pvals(do_ttest(tap_feature_data, "dataset", "tap", config.screen))

    tstat_gene_data = merge_tstats(baseline_tstats_gene, tap_tstats_gene, "Gene", config.screen)
    tstat_allele_data = merge_tstats(baseline_tstats_allele, tap_tstats_allele, "dataset", config.screen)

    outputs = {
        f"{config.screen}_tap_output.csv": tap_output,
        f"{config.screen}_baseline_output.csv": baseline_output,
        f"{config.screen}_post_stimulus.csv": post_stimulus_output,
        "tap_response_data.csv": tap_response_data,
        "tap_baseline_data.csv": baseline_output,
        "tstat_gene_data.csv": tstat_gene_data.reset_index(),
        "tstat_allele_data.csv": tstat_allele_data.reset_index(),
        "gene_MSD.csv": reorder_first_columns(gene_msd, ["Screen", "Gene"]),
        "allele_MSD.csv": reorder_first_columns(allele_msd, ["Screen", "dataset"]),
        "psa_summarised_data.csv": psa_summarised_data,
    }

    for file_name, data in outputs.items():
        target = output_dir / file_name
        data.to_csv(target, index=False)

    manifest = {
        **{key: str(value) for key, value in asdict(config).items()},
        "generated_files": len(outputs),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if config.upload_to_db:
        upload_outputs(
            config=config,
            tap_response_data=tap_response_data,
            baseline_output=baseline_output,
            tstat_gene_data=tstat_gene_data.reset_index(),
            tstat_allele_data=tstat_allele_data.reset_index(),
            gene_msd=reorder_first_columns(gene_msd, ["Screen", "Gene"]),
            allele_msd=reorder_first_columns(allele_msd, ["Screen", "dataset"]),
            psa_summarised_data=psa_summarised_data,
        )

    return {
        "tap_output": tap_output,
        "baseline_output": baseline_output,
        "post_stimulus_output": post_stimulus_output,
        "tap_response_data": tap_response_data,
        "tap_feature_data": tap_feature_data,
        "gene_msd": gene_msd,
        "allele_msd": allele_msd,
        "tstat_gene_data": tstat_gene_data,
        "tstat_allele_data": tstat_allele_data,
        "psa_summarised_data": psa_summarised_data,
    }


def run_chore_extraction(config: PipelineConfig) -> None:
    zip_files = find_files(config.raw_data_dir, ".zip")
    if not zip_files:
        raise FileNotFoundError(f"No .zip files found under {config.raw_data_dir}")

    base_command = [
        "java",
        f"-Xms{config.java_heap}",
        "-jar",
        str(config.chore_jar),
        "-p",
        "0.027",
        "-s",
        "0.1",
        "-t",
        "20",
        "-M",
        "2",
        "--shadowless",
        "-S",
        "-o",
        "nNss*b12xyMmeSakcrP",
        "--plugin",
        "Reoutline::exp",
        "--plugin",
        "Respine",
    ]
    reversal_plugins = [
        "--plugin",
        "MeasureReversal::tap::dt=1::collect=0.5::postfix=trv",
        "--plugin",
        "MeasureReversal::puff::dt=3::collect=0.5::postfix=prv",
        "--plugin",
        "MeasureReversal::postfix=txt",
    ]

    for zip_file in zip_files:
        subprocess.run(base_command + reversal_plugins + [str(zip_file)], check=True, cwd=config.raw_data_dir)
    for zip_file in zip_files:
        subprocess.run(base_command + ["-N", "all", str(zip_file)], check=True, cwd=config.raw_data_dir)


def build_tap_output(config: PipelineConfig) -> pd.DataFrame:
    trv_files = find_files(config.raw_data_dir, ".trv")
    if not trv_files:
        raise FileNotFoundError(f"No .trv files found under {config.raw_data_dir}")

    tap_tolerances = build_tap_tolerances(config.number_of_taps, config.first_tap_seconds, config.isi_seconds)
    datasets = get_sorted_datasets(trv_files, config.screen)
    frames = [process_trv_dataset(dataset, trv_files, config.screen, tap_tolerances) for dataset in datasets]
    tap_output = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    tap_output["Screen"] = config.screen
    tap_output[["Gene", "Allele"]] = tap_output["dataset"].str.split("_", n=1, expand=True)
    tap_output["Allele"] = tap_output["Allele"].fillna("N2")
    return tap_output


def build_baseline_and_psa_outputs(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    dat_files = [
        path
        for path in find_files(config.raw_data_dir, ".dat")
        if "_" in path.stem and not path.name.startswith("._")
    ]
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found under {config.raw_data_dir}")

    datasets = get_sorted_datasets(dat_files, config.screen)
    psa_tolerances = build_psa_tolerances(config.number_of_taps, config.first_tap_seconds, config.isi_seconds)

    experiment_counter = 1
    frames: list[pd.DataFrame] = []
    for dataset in datasets:
        dataset_frames = [path for path in dat_files if dataset == extract_dataset(path)]
        for file_path in dataset_frames:
            frame = read_dat_file(file_path, config.screen, experiment_counter)
            assign_taps(frame, "Time", psa_tolerances)
            insert_plate_numbers(frame)
            frame["dataset"] = dataset
            frame[["Gene", "Allele"]] = frame["dataset"].str.split("_", n=1, expand=True)
            frame["Allele"] = frame["Allele"].fillna("N2")
            frames.append(frame)
            experiment_counter += 1

    base = pd.concat(frames, ignore_index=True)
    base["Screen"] = config.screen

    baseline_output = (
        base.drop(columns=["Tap", "Puff", "x", "y", "Experiment", "taps", "plate"])
        .dropna()
        .reset_index(drop=True)
    )
    baseline_output = baseline_output[(baseline_output["Time"] <= 590.0) & (baseline_output["Time"] >= 490.0)]

    post_stimulus = base[base["Time"] > 599.0].drop(columns=["Puff", "x", "y"]).dropna().reset_index(drop=True)
    post_stimulus = (
        post_stimulus.groupby(
            ["Experiment", "Screen", "Date", "Plate_id", "Gene", "Allele", "dataset", "taps"],
            as_index=False,
        )
        .agg(
            {
                "Time": "min",
                "n": "mean",
                "Number": "mean",
                "Speed": "mean",
                "Bias": "mean",
                "Tap": "mean",
                "Morphwidth": "mean",
                "Midline": "mean",
                "Area": "mean",
                "Angular Speed": "mean",
                "Aspect Ratio": "mean",
                "Kink": "mean",
                "Curve": "mean",
                "Crab": "mean",
                "Pathlength": "mean",
            }
        )
        .reset_index(drop=True)
    )

    return baseline_output, post_stimulus


def build_analysis_outputs(
    tap_output: pd.DataFrame,
    post_stimulus_output: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    psa_output = post_stimulus_output.copy()
    for column in PSA_RENAMED_COLUMNS:
        psa_output = psa_output.rename(columns={column: f"PSA {column}"})

    tap_response_data = pd.merge(
        tap_output,
        psa_output.drop(
            columns=[
                "Experiment",
                "Time",
                "Tap",
                "PSA Morphwidth",
                "PSA Midline",
                "PSA Area",
                "PSA Angular Speed",
            ],
            errors="ignore",
        ),
        how="outer",
        on=["dataset", "Gene", "Allele", "Date", "Plate_id", "Screen", "taps"],
    )
    tap_response_data = tap_response_data[
        [
            "dataset",
            "Gene",
            "Allele",
            "Date",
            "Plate_id",
            "plate",
            "Screen",
            "taps",
            "time",
            "dura",
            "dist",
            "prob",
            "speed",
            "PSA Speed",
            "PSA Bias",
            "PSA Aspect Ratio",
            "PSA Kink",
            "PSA Curve",
            "PSA Crab",
        ]
    ]

    tap_feature_data = build_tap_feature_data(tap_response_data, config)
    psa_summarised_data = summarize_psa_by_plate(tap_response_data, config)

    tap_response_data = reorder_first_columns(
        tap_response_data,
        ["Screen", "dataset", "Gene", "Allele", "Plate_id", "Date", "taps"],
    )
    psa_summarised_data = reorder_first_columns(
        psa_summarised_data,
        ["Screen", "dataset", "Gene", "Allele", "Plate_id", "Date"],
    )

    return tap_response_data, tap_feature_data, psa_summarised_data


def build_tap_feature_data(tap_response_data: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    first_tap = (
        tap_response_data[tap_response_data["taps"] == 1]
        .reset_index(drop=True)
        .rename(
            columns={
                "dura": "init_dura",
                "prob": "init_prob",
                "speed": "init_speed",
                "PSA Speed": "init_psa_speed",
                "PSA Bias": "init_psa_bias",
                "PSA Aspect Ratio": "init_psa_aspect_ratio",
                "PSA Kink": "init_psa_kink",
                "PSA Curve": "init_psa_curve",
                "PSA Crab": "init_psa_crab",
            }
        )
    )
    recovery_tap = (
        tap_response_data[tap_response_data["taps"] == config.recovery_tap]
        .reset_index(drop=True)
        .rename(
            columns={
                "dura": "recov_dura",
                "prob": "recov_prob",
                "speed": "recov_speed",
                "PSA Speed": "recov_psa_speed",
                "PSA Bias": "recov_psa_bias",
                "PSA Aspect Ratio": "recov_psa_aspect_ratio",
                "PSA Kink": "recov_psa_kink",
                "PSA Curve": "recov_psa_curve",
                "PSA Crab": "recov_psa_crab",
            }
        )
    )
    final_taps = (
        tap_response_data[
            tap_response_data["taps"].between(config.number_of_taps - 2, config.number_of_taps, inclusive="both")
        ]
        .groupby(["dataset", "Date", "Plate_id", "Screen", "Gene", "Allele", "plate"], as_index=False)
        .mean(numeric_only=True)
        .rename(
            columns={
                "dura": "final_dura",
                "prob": "final_prob",
                "speed": "final_speed",
                "PSA Speed": "final_psa_speed",
                "PSA Bias": "final_psa_bias",
                "PSA Aspect Ratio": "final_psa_aspect_ratio",
                "PSA Kink": "final_psa_kink",
                "PSA Curve": "final_psa_curve",
                "PSA Crab": "final_psa_crab",
            }
        )
    )

    habit = pd.merge(
        first_tap,
        final_taps,
        on=["dataset", "plate", "Plate_id", "Screen", "Gene", "Allele", "Date"],
        how="left",
    ).drop(columns=["time_x", "time_y", "dist_x", "dist_y", "taps_x", "taps_y"], errors="ignore")

    for metric in ["dura", "prob", "speed", "psa_speed", "psa_bias", "psa_aspect_ratio", "psa_kink", "psa_curve", "psa_crab"]:
        habit[f"habit_{metric}"] = habit[f"init_{metric}"] - habit[f"final_{metric}"]

    merge_how = "outer" if recovery_tap.empty else "left"
    tap_feature_data = pd.merge(
        habit,
        recovery_tap,
        on=["dataset", "plate", "Plate_id", "Screen", "Gene", "Allele", "Date"],
        how=merge_how,
    )
    if config.screen not in {"Neuron_Genes_Screen", "G-Proteins_Screen"}:
        tap_feature_data = tap_feature_data.dropna()

    for metric in ["dura", "prob", "speed", "psa_speed", "psa_bias", "psa_aspect_ratio", "psa_kink", "psa_curve", "psa_crab"]:
        tap_feature_data[f"recovery_{metric}"] = percentage_change(
            tap_feature_data[f"recov_{metric}"],
            tap_feature_data[f"init_{metric}"],
        )
        tap_feature_data[f"memory_retention_{metric}"] = (
            tap_feature_data[f"recov_{metric}"] - tap_feature_data[f"final_{metric}"]
        )

    max_psa = (
        tap_response_data.groupby(
            ["dataset", "Gene", "Allele", "Date", "Plate_id", "plate", "Screen"],
            as_index=False,
        )
        .max(numeric_only=True)
        .round(4)
        .rename(
            columns={
                "PSA Speed": "max_psa_speed",
                "PSA Bias": "max_psa_bias",
                "PSA Aspect Ratio": "max_psa_aspect_ratio",
                "PSA Kink": "max_psa_kink",
                "PSA Curve": "max_psa_curve",
                "PSA Crab": "max_psa_crab",
            }
        )
        .drop(columns=["taps", "time", "dura", "dist", "prob", "speed"], errors="ignore")
    )
    mean_psa = (
        tap_response_data.groupby(
            ["dataset", "Gene", "Allele", "Date", "Plate_id", "plate", "Screen"],
            as_index=False,
        )
        .mean(numeric_only=True)
        .round(4)
        .rename(
            columns={
                "prob": "mean_prob",
                "dura": "mean_dura",
                "speed": "mean_speed",
                "PSA Speed": "mean_psa_speed",
                "PSA Bias": "mean_psa_bias",
                "PSA Aspect Ratio": "mean_psa_aspect_ratio",
                "PSA Kink": "mean_psa_kink",
                "PSA Curve": "mean_psa_curve",
                "PSA Crab": "mean_psa_crab",
            }
        )
        .drop(columns=["taps", "time", "dist"], errors="ignore")
    )

    tap_feature_data = pd.merge(
        tap_feature_data,
        max_psa,
        on=["dataset", "plate", "Plate_id", "Screen", "Gene", "Allele", "Date"],
        how="outer",
    )
    tap_feature_data = pd.merge(
        tap_feature_data,
        mean_psa,
        on=["dataset", "plate", "Plate_id", "Screen", "Gene", "Allele", "Date"],
        how="outer",
    )

    for metric in ["psa_speed", "psa_bias", "psa_aspect_ratio", "psa_kink", "psa_curve", "psa_crab"]:
        tap_feature_data[f"sensitization_{metric}"] = (
            tap_feature_data[f"max_{metric}"] - tap_feature_data[f"init_{metric}"]
        )

    if config.screen in {"Neuron_Genes_Screen", "G-Proteins_Screen"}:
        required = [
            "dataset",
            "Gene",
            "Allele",
            "Date",
            "Plate_id",
            "plate",
            "Screen",
            "init_dura",
            "init_prob",
            "init_speed",
            "final_dura",
            "final_prob",
            "final_speed",
            "habit_dura",
            "habit_prob",
            "habit_speed",
        ]
        tap_feature_data = tap_feature_data.dropna(subset=required)
    else:
        tap_feature_data = tap_feature_data.dropna()

    return tap_feature_data


def summarize_psa_by_plate(tap_response_data: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    group_cols = ["Screen", "dataset", "Gene", "Allele", "Plate_id", "Date", "plate"]
    records: list[dict[str, Any]] = []

    for _, group in tap_response_data.groupby(group_cols, dropna=False):
        row = {column: group.iloc[0][column] for column in group_cols}
        for metric in PSA_FEATURE_COLUMNS:
            short_name = metric.replace("PSA ", "").lower().replace(" ", "_")
            series = group[["taps", metric]].dropna().sort_values("taps")
            if series.empty:
                continue
            initial = series.loc[series["taps"] == 1, metric].mean()
            final = series.loc[series["taps"].between(config.number_of_taps - 2, config.number_of_taps), metric].mean()
            recovery = series.loc[series["taps"] == config.recovery_tap, metric].mean()
            peak = series[metric].max()
            peak_tap = series.loc[series[metric].idxmax(), "taps"]
            mean_value = series[metric].mean()
            sensitization = peak - initial
            habituation = peak - final
            if metric in POSITIVE_RECOVERY_METRICS:
                spontaneous_recovery = safe_percentage_change(recovery - initial, initial)
                memory_retention = recovery - final
            else:
                spontaneous_recovery = safe_percentage_change(initial - recovery, initial)
                memory_retention = final - recovery

            row.update(
                {
                    f"Initial PSA {to_title(short_name)}": initial,
                    f"Final PSA {to_title(short_name)}": final,
                    f"Recovery PSA {to_title(short_name)}": recovery,
                    f"Peak PSA {to_title(short_name)}": peak,
                    f"Peak Tap Number of PSA {to_title(short_name)}": peak_tap,
                    f"Average PSA {to_title(short_name)}": mean_value,
                    f"Sensitization of PSA {to_title(short_name)}": sensitization,
                    f"Habituation of PSA {to_title(short_name)}": habituation,
                    f"Spontaneous Recovery of PSA {to_title(short_name)}": spontaneous_recovery,
                    f"Memory Retention of PSA {to_title(short_name)}": memory_retention,
                }
            )
        records.append(row)

    return pd.DataFrame(records)


def get_output_byplate(output: pd.DataFrame, dataset_type: str, allele: bool) -> pd.DataFrame:
    drop_columns = {
        "baseline": ["Plate_id", "n", "Number", "Time", "Screen", "Date", "Allele"],
        "tap": ["Plate_id", "Screen", "Date", "Allele", "dist", "plate", "time", "taps", "recov_dura", "recov_prob", "recov_speed"],
    }[dataset_type][:]
    drop_columns.append("Gene" if allele else "dataset")
    return (
        output.groupby(["Plate_id", "Date", "Screen", "dataset", "Gene", "Allele"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=drop_columns, errors="ignore")
    )


def extract_phenotypes(df: pd.DataFrame) -> list[pd.DataFrame]:
    first = df.columns[0]
    return [df[[first, column]].copy() for column in df.columns[1:]]


def ci95(df: pd.DataFrame) -> pd.DataFrame:
    for metric in df.columns.levels[0]:
        if metric == "Gene":
            continue
        ci_hi: list[float] = []
        ci_lo: list[float] = []
        for index in df[metric].index:
            mean_value = df[metric]["mean"].loc[index]
            count_value = df[metric]["count"].loc[index]
            sem_value = df[metric]["sem"].loc[index]
            interval = stats.t.interval(confidence=0.95, df=count_value - 1, loc=mean_value, scale=sem_value)
            ci_hi.append(interval[1])
            ci_lo.append(interval[0])
        df[(metric, "ci95_hi")] = ci_hi
        df[(metric, "ci95_lo")] = ci_lo
    return df


def calculate_msd(list_of_dfs: list[pd.DataFrame], by: str, screen: str) -> list[pd.DataFrame]:
    results: list[pd.DataFrame] = []
    for frame in list_of_dfs:
        phenotype = frame.columns[1]
        stats_frame = frame.groupby(by)[frame.columns[1]].agg(["mean", "count", "sem"])
        if not isinstance(stats_frame.columns, pd.MultiIndex):
            stats_frame.columns = pd.MultiIndex.from_tuples([(phenotype, column) for column in stats_frame.columns])
        stats_frame = ci95(stats_frame)
        control_mask = control_selector(stats_frame.index, by, screen)
        control_frame = stats_frame[control_mask]
        stats_frame.iloc[:, 0] -= control_frame.iloc[0, 0]
        stats_frame.iloc[:, 3] -= control_frame.iloc[0, 0]
        stats_frame.iloc[:, 4] -= control_frame.iloc[0, 0]
        results.append(stats_frame)
    return results


def get_msd(list_msd: list[pd.DataFrame]) -> pd.DataFrame:
    current = list_msd[0]
    for frame in list_msd[1:]:
        current = current.join(frame)
    return current


def get_combined_msd(baseline_byplate: pd.DataFrame, tap_byplate: pd.DataFrame, by: str, screen: str) -> pd.DataFrame:
    baseline_msd = get_msd(calculate_msd(extract_phenotypes(baseline_byplate), by, screen))
    tap_msd = get_msd(calculate_msd(extract_phenotypes(tap_byplate), by, screen))
    combined = pd.merge(baseline_msd, tap_msd, on=by, how="outer")
    combined = combined.rename(columns=MSD_RENAMES).reset_index()
    combined.columns = combined.columns.to_flat_index().str.join("-")
    combined = combined.rename(columns={f"{by}-": by})
    combined["Screen"] = screen
    return combined


def do_ttest(data: pd.DataFrame, by: str, dataset_type: str, screen: str) -> pd.DataFrame:
    metrics = BASELINE_TSTAT_METRICS if dataset_type == "baseline" else TAP_TSTAT_METRICS
    outputs = [pd.DataFrame(columns=[by, pretty_metric_name(metric), f"{pretty_metric_name(metric)} p-value"]) for metric in metrics]

    for entity in data[by].dropna().unique():
        if is_control(entity, by, screen):
            continue
        subset = data[data[by] == entity]
        matching_dates = data[data["Date"].isin(subset["Date"].unique())]
        comparison = filter_for_entity_and_control(matching_dates, entity, by, screen)
        for metric, output in zip(metrics, outputs):
            append_ttest(metric, comparison, output, by)

    merged = pd.DataFrame()
    for output in outputs:
        averaged = output.groupby([by], as_index=False).mean(numeric_only=True)
        if merged.empty:
            merged = averaged
        else:
            merged = merged.join(averaged.iloc[:, 1:3])
    return merged.set_index(by)


def pair_pvals(df: pd.DataFrame) -> pd.DataFrame:
    pvals = df.iloc[:, 1::2].copy()
    values = df.iloc[:, ::2].copy()
    pvals.columns = [column if "p-value" in column else f"{column} p-value" for column in pvals.columns]
    corrected = pvals.apply(
        lambda row: pd.Series(fdrcorrection(row.dropna(), alpha=0.1)[1], index=row.dropna().index),
        axis=1,
    )
    corrected.columns = [column.replace(" p-value", "") for column in corrected.columns]
    result = pd.DataFrame(index=values.index, columns=values.columns)
    for column in values.columns:
        result[column] = list(zip(values[column], corrected[column] if column in corrected.columns else [None] * len(values)))
    return result


def merge_tstats(baseline: pd.DataFrame, tap: pd.DataFrame, by: str, screen: str) -> pd.DataFrame:
    merged = pd.merge(baseline, tap, on=by, how="left").sort_index().rename(columns=TSTAT_RENAMES)
    merged["Screen"] = screen
    return reorder_screen_column(merged)


def upload_outputs(
    config: PipelineConfig,
    tap_response_data: pd.DataFrame,
    baseline_output: pd.DataFrame,
    tstat_gene_data: pd.DataFrame,
    tstat_allele_data: pd.DataFrame,
    gene_msd: pd.DataFrame,
    allele_msd: pd.DataFrame,
    psa_summarised_data: pd.DataFrame,
) -> None:
    if config.database_ini is None:
        raise ValueError("Database upload requested, but no database.ini path was provided.")
    database_config = load_database_config(config.database_ini)
    if not database_config.get("user") or not database_config.get("password"):
        raise ValueError(f"Missing database credentials in {config.database_ini}")

    engine = create_engine(
        "postgresql+psycopg://{user}:{password}@{host}:{port}/{database}".format(**database_config)
    )

    table_payloads: dict[str, pd.DataFrame] = {
        "tap_response_data": tap_response_data,
        "tap_baseline_data": baseline_output,
        "tstat_gene_data": tstat_gene_data.dropna(thresh=10),
        "tstat_allele_data": tstat_allele_data.dropna(thresh=10),
        "gene_MSD": gene_msd,
        "allele_MSD": allele_msd,
        "psa_summarised_data": psa_summarised_data,
    }

    if config.db_upload_mode == "replace":
        selected_tables = set(config.replace_tables)
        if not selected_tables:
            raise ValueError("Replace mode requires at least one selected table.")
        for table_name, dataframe in table_payloads.items():
            if table_name in selected_tables:
                dataframe.to_sql(table_name, engine, if_exists="replace", index=False, method=None)
        return

    for table_name, dataframe in table_payloads.items():
        dataframe.to_sql(table_name, engine, if_exists="append", index=False, method=postgres_skip_on_duplicate)


def postgres_skip_on_duplicate(pd_table: Any, conn: Any, keys: list[str], data_iter: Any) -> None:
    data = [dict(zip(keys, row)) for row in data_iter]
    conn.execute(insert(pd_table.table).on_conflict_do_nothing(), data)


def load_database_config(path: Path) -> dict[str, str]:
    parser = ConfigParser()
    parser.read(path)
    if not parser.has_section("postgresql"):
        raise ValueError(f"Section 'postgresql' not found in {path}")
    return {key: value for key, value in parser.items("postgresql")}


def find_files(root: Path, suffix: str) -> list[Path]:
    return sorted(path for path in root.rglob(f"*{suffix}") if path.is_file() and not path.name.startswith("._"))


def build_tap_tolerances(number_of_taps: int, first_tap: int, isi_seconds: int) -> list[tuple[float, float]]:
    lower = np.arange(first_tap - 2, first_tap - 2 + (number_of_taps * isi_seconds), isi_seconds)
    upper = np.arange(first_tap + 2, first_tap + 2 + (number_of_taps * isi_seconds), isi_seconds)
    tolerances = [(float(low), float(high)) for low, high in zip(lower, upper)]
    tolerances.append((1188.0, 1191.0))
    return tolerances


def build_psa_tolerances(number_of_taps: int, first_tap: int, isi_seconds: int) -> list[tuple[float, float]]:
    lower = np.arange(first_tap + 7.0, first_tap + 7.0 + (number_of_taps * isi_seconds), isi_seconds)
    upper = np.arange(first_tap + 9.5, first_tap + 9.5 + (number_of_taps * isi_seconds), isi_seconds)
    tolerances = [(float(low), float(high)) for low, high in zip(lower, upper)]
    tolerances.append((1197.5, 1199.0))
    return tolerances


def get_sorted_datasets(files: list[Path], screen: str) -> list[str]:
    datasets = sorted({extract_dataset(path) for path in files if extract_dataset(path)})
    if screen == "Neuron_Genes_Screen":
        for control in ["N2_XJ1", "N2_N2"]:
            if control in datasets:
                datasets.insert(0, datasets.pop(datasets.index(control)))
    elif "N2" in datasets:
        datasets.insert(0, datasets.pop(datasets.index("N2")))
    return datasets


def process_trv_dataset(dataset: str, trv_files: list[Path], screen: str, tolerances: list[tuple[float, float]]) -> pd.DataFrame:
    matching_files = [path for path in trv_files if dataset == extract_dataset(path)]
    if not matching_files:
        raise AssertionError(f"{dataset} is not a good identifier as number of plates = 0")

    frames = [read_trv_file(path, screen) for path in matching_files]
    combined = pd.concat(frames, ignore_index=True).rename(columns=TRV_COLUMNS)
    combined["plate"] = 0
    combined["prob"] = combined["stim_rev"] / (combined["no_rev"] + combined["stim_rev"])
    combined["speed"] = combined["dist"] / combined["dura"]
    final = combined[["time", "dura", "dist", "prob", "speed", "plate", "Date", "Plate_id", "Screen"]].copy()
    final["dataset"] = dataset
    assign_taps(final, "time", tolerances)
    insert_plate_numbers(final)
    return final


def read_trv_file(path: Path, screen: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=" ", header=None, encoding_errors="ignore")
    frame["Plate_id"] = build_plate_id(path)
    frame["Date"] = extract_date(path)
    frame["Screen"] = screen
    return frame


def read_dat_file(path: Path, screen: str, experiment_counter: int) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=" ", header=None, encoding_errors="ignore").rename(columns=DAT_COLUMNS)
    frame["Plate_id"] = build_plate_id(path)
    frame["Date"] = extract_date(path)
    frame["Screen"] = screen
    frame["Experiment"] = experiment_counter
    frame["plate"] = 0
    return frame


def assign_taps(df: pd.DataFrame, time_column: str, tolerances: list[tuple[float, float]]) -> None:
    df["taps"] = np.nan
    if not df.empty:
        df.loc[df.index[0], "taps"] = 0
    for tap_number, tolerance in enumerate(tolerances, start=1):
        lower, upper = tolerance
        in_range = df[time_column].between(lower, upper, inclusive="both")
        df.loc[in_range, "taps"] = tap_number


def insert_plate_numbers(df: pd.DataFrame) -> None:
    df["plate"] = (df["taps"] == 1).cumsum()


def extract_dataset(path: Path) -> str:
    try:
        return path.parents[1].name
    except IndexError as exc:
        raise ValueError(f"Could not infer dataset from {path}") from exc


def extract_date(path: Path) -> str:
    return path.parent.name.split("_")[0]


def build_plate_id(path: Path) -> str:
    return f"{path.parent.name}_{path.stem.split('_')[-1]}"


def append_ttest(metric: str, frame: pd.DataFrame, output: pd.DataFrame, by: str) -> None:
    for entity in frame[by].dropna().unique():
        if by == "Gene":
            sample = frame[frame["Gene"] == entity][metric]
            control = frame[frame["Gene"] == "N2"][metric]
        else:
            sample = frame[frame["dataset"] == entity][metric]
            control = frame[frame["Allele"].isin(["XJ1", "N2"])][metric]
        t_stat, p_value = ttest_ind(sample, control, equal_var=False, nan_policy="omit")
        output.loc[len(output)] = [entity, t_stat, p_value]


def filter_for_entity_and_control(data: pd.DataFrame, entity: str, by: str, screen: str) -> pd.DataFrame:
    if screen == "Neuron_Genes_Screen":
        if by == "Gene":
            return data[data["Gene"].isin(["N2", entity])]
        return data[data["dataset"].isin(["N2_N2", "N2_XJ1", entity])]
    if by == "Gene":
        return data[data["Gene"].isin(["N2", entity])]
    return data[data["dataset"].isin(["N2", entity])]


def control_selector(index: pd.Index, by: str, screen: str) -> pd.Series:
    if screen == "Neuron_Genes_Screen":
        return index == "N2" if by == "Gene" else index.isin(["N2_XJ1", "N2_N2"])
    return index == "N2"


def is_control(entity: str, by: str, screen: str) -> bool:
    if screen == "Neuron_Genes_Screen":
        return entity in (["N2"] if by == "Gene" else ["N2_XJ1", "N2_N2"])
    return entity == "N2"


def pretty_metric_name(metric: str) -> str:
    mapping = {
        "init_dura": "Initial Response Duration",
        "init_prob": "Initial Response Probability",
        "init_speed": "Initial Response Speed",
        "final_dura": "Final Response Duration",
        "final_prob": "Final Response Probability",
        "final_speed": "Final Response Speed",
        "habit_dura": "Habituation of Response Duration",
        "habit_prob": "Habituation of Response Probability",
        "habit_speed": "Habituation of Response Speed",
        "recovery_dura": "Spontaneous Recovery of Response Duration",
        "recovery_prob": "Spontaneous Recovery of Response Probability",
        "recovery_speed": "Spontaneous Recovery of Response Speed",
        "memory_retention_dura": "Memory Retention of Response Duration",
        "memory_retention_prob": "Memory Retention of Response Probability",
        "memory_retention_speed": "Memory Retention of Response Speed",
        "init_psa_speed": "Initial PSA Speed",
        "final_psa_speed": "Final PSA Speed",
        "recov_psa_speed": "Recovery PSA Speed",
        "max_psa_speed": "Peak PSA Speed",
        "mean_psa_speed": "Average PSA Speed",
        "sensitization_psa_speed": "Sensitization of PSA Speed",
        "habit_psa_speed": "Habituation of PSA Speed",
        "recovery_psa_speed": "Spontaneous Recovery of PSA Speed",
        "memory_retention_psa_speed": "Memory Retention of PSA Speed",
        "init_psa_bias": "Initial PSA Bias",
        "final_psa_bias": "Final PSA Bias",
        "recov_psa_bias": "Recovery PSA Bias",
        "max_psa_bias": "Peak PSA Bias",
        "mean_psa_bias": "Average PSA Bias",
        "sensitization_psa_bias": "Sensitization of PSA Bias",
        "habit_psa_bias": "Habituation of PSA Bias",
        "recovery_psa_bias": "Spontaneous Recovery of PSA Bias",
        "memory_retention_psa_bias": "Memory Retention of PSA Bias",
        "init_psa_aspect_ratio": "Initial PSA Aspect Ratio",
        "final_psa_aspect_ratio": "Final PSA Aspect Ratio",
        "recov_psa_aspect_ratio": "Recovery PSA Aspect Ratio",
        "max_psa_aspect_ratio": "Peak PSA Aspect Ratio",
        "mean_psa_aspect_ratio": "Average PSA Aspect Ratio",
        "sensitization_psa_aspect_ratio": "Sensitization of PSA Aspect Ratio",
        "habit_psa_aspect_ratio": "Habituation of PSA Aspect Ratio",
        "recovery_psa_aspect_ratio": "Spontaneous Recovery of PSA Aspect Ratio",
        "memory_retention_psa_aspect_ratio": "Memory Retention of PSA Aspect Ratio",
        "init_psa_kink": "Initial PSA Kink",
        "final_psa_kink": "Final PSA Kink",
        "recov_psa_kink": "Recovery PSA Kink",
        "max_psa_kink": "Peak PSA Kink",
        "mean_psa_kink": "Average PSA Kink",
        "sensitization_psa_kink": "Sensitization of PSA Kink",
        "habit_psa_kink": "Habituation of PSA Kink",
        "recovery_psa_kink": "Spontaneous Recovery of PSA Kink",
        "memory_retention_psa_kink": "Memory Retention of PSA Kink",
        "init_psa_curve": "Initial PSA Curve",
        "final_psa_curve": "Final PSA Curve",
        "recov_psa_curve": "Recovery PSA Curve",
        "max_psa_curve": "Peak PSA Curve",
        "mean_psa_curve": "Average PSA Curve",
        "sensitization_psa_curve": "Sensitization of PSA Curve",
        "habit_psa_curve": "Habituation of PSA Curve",
        "recovery_psa_curve": "Spontaneous Recovery of PSA Curve",
        "memory_retention_psa_curve": "Memory Retention of PSA Curve",
        "init_psa_crab": "Initial PSA Crab",
        "final_psa_crab": "Final PSA Crab",
        "recov_psa_crab": "Recovery PSA Crab",
        "max_psa_crab": "Peak PSA Crab",
        "mean_psa_crab": "Average PSA Crab",
        "sensitization_psa_crab": "Sensitization of PSA Crab",
        "habit_psa_crab": "Habituation of PSA Crab",
        "recovery_psa_crab": "Spontaneous Recovery of PSA Crab",
        "memory_retention_psa_crab": "Memory Retention of PSA Crab",
    }
    return mapping.get(metric, metric)


def percentage_change(current: pd.Series, baseline: pd.Series) -> pd.Series:
    return ((current - baseline) / baseline) * 100


def safe_percentage_change(numerator: float, denominator: float) -> float:
    if pd.isna(denominator) or denominator == 0:
        return math.nan
    return 100 * numerator / denominator


def reorder_screen_column(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns.tolist()
    if "Screen" in columns:
        columns.insert(0, columns.pop(columns.index("Screen")))
    return df[columns]


def reorder_first_columns(df: pd.DataFrame, first_columns: list[str]) -> pd.DataFrame:
    ordered = [column for column in first_columns if column in df.columns]
    ordered.extend(column for column in df.columns if column not in ordered)
    return df[ordered]


def to_title(value: str) -> str:
    return value.replace("_", " ").title().replace("Psa", "PSA")
