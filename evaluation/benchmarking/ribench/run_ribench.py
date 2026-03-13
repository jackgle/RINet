"""
RIBench benchmark evaluation for RINet V1 and V2 models (pure Python).

Reproduces the RIBench evaluation suite (Ammer et al., Clin Chem 2022) without
requiring R. Generates test data from the specification CSV, runs models, and
computes z-score deviation metrics.

Usage:
    python evaluation/benchmarking/run_ribench.py \
        --v1-model-path /path/to/v1/model/ \
        --v2-model-path /path/to/v2/model/ \
        --subset test

    # With pre-generated R data:
    python evaluation/benchmarking/run_ribench.py \
        --v1-model-path /path/to/v1/model/ \
        --data-dir tmp/benchmark/results/Data/ \
        --subset all --outliers
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rinet.models import RINetV1Pipeline, RINetV2Pipeline
from rinet.utils import stats_to_ri

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEC_CSV = None  # resolved from --data-dir at runtime

BIOMARKERS = ["Hb", "Ca", "FT4", "AST", "LACT", "GGT", "TSH", "IgE", "CRP", "LDH"]
DIST_TYPES = ["normal", "skewed", "heavilySkewed", "shifted"]
PERCENTILES = [0.025, 0.975]
CUTOFF_Z = 5.0


# ---------------------------------------------------------------------------
# Box-Cox transforms (matching R's RIbench implementation)
# ---------------------------------------------------------------------------
def box_cox(x, lam):
    x = np.asarray(x, dtype=float)
    if abs(lam) < 1e-20:
        return np.log(x)
    return (x ** lam - 1.0) / lam


def inv_box_cox(x, lam):
    x = np.asarray(x, dtype=float)
    if abs(lam) < 1e-20:
        return np.exp(x)
    return (lam * x + 1.0) ** (1.0 / lam)


# ---------------------------------------------------------------------------
# Data generation (reproducing R's generateDataFiles)
# ---------------------------------------------------------------------------
def generate_test_data(spec_row, rng):
    """Generate a single test set from its specification row.

    Uses the provided numpy RandomState to match the sequential RNG
    consumption pattern of R's generateDataFiles.
    """
    N = int(spec_row["N"])
    frac_pathol = spec_row["fractionPathol"]
    bg_fraction = spec_row["bg_fraction"]
    analyte = spec_row["Analyte"]
    decimals = int(spec_row["decimals"])

    # Non-pathological distribution
    n_phys = int(N * (1 - frac_pathol - bg_fraction))
    phys_raw = inv_box_cox(
        rng.normal(spec_row["nonp_mu"], spec_row["nonp_sigma"], size=N),
        spec_row["nonp_lambda"],
    ) + spec_row["nonp_shift"]

    # Resample if not enough valid (>= 0) values
    if np.sum(phys_raw >= 0) < n_phys:
        phys_raw = inv_box_cox(
            rng.normal(spec_row["nonp_mu"], spec_row["nonp_sigma"], size=N * 3),
            spec_row["nonp_lambda"],
        ) + spec_row["nonp_shift"]

    valid = phys_raw[phys_raw >= 0]
    phys = rng.choice(valid, size=n_phys, replace=False)

    # Background distribution
    n_bg = int(N * bg_fraction)
    background = rng.uniform(spec_row["bg_min"], spec_row["bg_max"], size=n_bg)

    # Left pathological
    left_ratio = spec_row["left_ratio"]
    right_ratio = spec_row["right_ratio"]
    ratio_sum = spec_row["ratio_sum"]

    if analyte == "CRP":
        patho_left = np.array([])
        # Still consume RNG for consistency (R generates but discards for CRP)
    else:
        n_left = int(N * frac_pathol * (left_ratio / ratio_sum))
        left_raw = rng.normal(spec_row["left_mu"], spec_row["left_sigma"], size=N)
        if np.sum(left_raw >= 0) < n_left:
            left_raw = rng.normal(spec_row["left_mu"], spec_row["left_sigma"], size=N * 3)
        valid_left = left_raw[left_raw >= 0]
        if len(valid_left) < n_left:
            patho_left = rng.choice(valid_left, size=n_left, replace=True)
        else:
            patho_left = rng.choice(valid_left, size=n_left, replace=False)

    # Right pathological
    n_right = int(N * frac_pathol * (right_ratio / ratio_sum))
    right_raw = rng.normal(spec_row["right_mu"], spec_row["right_sigma"], size=N)
    if np.sum(right_raw >= 0) < n_right:
        right_raw = rng.normal(spec_row["right_mu"], spec_row["right_sigma"], size=N * 3)
    valid_right = right_raw[right_raw >= 0]
    patho_right = rng.choice(valid_right, size=n_right, replace=False)

    # Combine and shuffle
    dd = np.concatenate([phys, patho_left, patho_right, background])
    rng.shuffle(dd)

    # Round
    dd = np.round(dd, decimals=decimals)

    return dd


def generate_all_test_data(testsets_df, all_testsets_df):
    """Generate test data for the specified subset, consuming RNG in the
    same order as R's generateDataFiles (which iterates over ALL test sets
    but only saves those in the subset).

    Returns dict: (Index, Analyte) -> data array
    """
    subset_indices = set(testsets_df["Index"].values)
    data = {}

    old_marker = all_testsets_df.iloc[0]["Analyte"]
    seed = int(all_testsets_df.iloc[0]["startSeed"])
    rng = np.random.RandomState(seed)

    for _, row in all_testsets_df.iterrows():
        marker = row["Analyte"]
        if marker != old_marker:
            seed = int(all_testsets_df[all_testsets_df["Analyte"] == marker].iloc[0]["startSeed"])
            rng = np.random.RandomState(seed)
            old_marker = marker

        dd = generate_test_data(row, rng)

        if row["Index"] in subset_indices:
            data[(int(row["Index"]), marker)] = dd

    return data


# ---------------------------------------------------------------------------
# Load pre-generated R data from CSV files
# ---------------------------------------------------------------------------
def load_r_data(data_dir, testsets_df):
    """Load test data from pre-generated R CSV files."""
    data = {}
    missing = []
    for _, row in testsets_df.iterrows():
        idx = int(row["Index"])
        analyte = row["Analyte"]
        seed = int(row["startSeed"])
        csv_path = Path(data_dir) / analyte / f"{idx}_{analyte}_seed_{seed}.csv"
        if csv_path.exists():
            data[(idx, analyte)] = pd.read_csv(csv_path, header=None).values.ravel()
        else:
            missing.append(str(csv_path))
    if missing:
        print(f"WARNING: {len(missing)} data files not found. First few:")
        for m in missing[:5]:
            print(f"  {m}")
    return data


# ---------------------------------------------------------------------------
# Model prediction
# ---------------------------------------------------------------------------
def predict_v1(pipeline, data_vec, log_scale=True):
    """Run V1 model and return (lower_ri, upper_ri) or None on failure."""
    try:
        results = pipeline.predict(data_vec, log_scale=log_scale, verbose=0)
        r = results[0]
        return np.array([r[0.025], r[0.975]])
    except Exception as e:
        return None


def predict_v2(pipeline, data_vec, log_scale=True):
    """Run V2 model and return (lower_ri, upper_ri) or None on failure."""
    try:
        results = pipeline.predict(data_vec, log_scale=log_scale, verbose=0)
        r = results[0]
        mean_val = r["mean"][0]
        std_val = r["std"][0]
        if log_scale:
            ri = np.exp(stats_to_ri(mean_val, std_val))
        else:
            ri = stats_to_ri(mean_val, std_val)
        return ri
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Z-score deviation metrics
# ---------------------------------------------------------------------------
def compute_zdev(gt_lrl, gt_url, est_lrl, est_url, nonp_mu, nonp_sigma, nonp_lambda, nonp_shift, analyte):
    """Compute z-score deviations for a single test set."""

    def to_zscore(val, shift, lam, mu, sigma):
        v = val - shift
        if v <= 0:
            v = 1e-20
        return (box_cox(v, lam) - mu) / sigma

    zz_gt_lrl = to_zscore(gt_lrl, nonp_shift, nonp_lambda, nonp_mu, nonp_sigma)
    zz_gt_url = to_zscore(gt_url, nonp_shift, nonp_lambda, nonp_mu, nonp_sigma)

    if est_lrl is None or est_url is None or np.isnan(est_lrl) or np.isnan(est_url):
        return {"zz_dev_lrl": np.nan, "zz_dev_url": np.nan, "zz_dev_ov": np.nan,
                "zz_dev_abs_ov": np.nan, "zz_dev_abs_cutoff_ov": np.nan}

    zz_est_lrl = to_zscore(est_lrl, nonp_shift, nonp_lambda, nonp_mu, nonp_sigma)
    zz_est_url = to_zscore(est_url, nonp_shift, nonp_lambda, nonp_mu, nonp_sigma)

    zz_dev_lrl = zz_gt_lrl - zz_est_lrl
    zz_dev_url = zz_gt_url - zz_est_url

    if analyte == "CRP":
        zz_dev_ov = zz_dev_url
        zz_dev_abs_ov = abs(zz_dev_url)
    else:
        zz_dev_ov = (zz_dev_lrl + zz_dev_url) / 2
        zz_dev_abs_ov = (abs(zz_dev_lrl) + abs(zz_dev_url)) / 2

    zz_dev_abs_cutoff_ov = zz_dev_abs_ov if zz_dev_abs_ov <= CUTOFF_Z else np.nan

    return {
        "zz_dev_lrl": zz_dev_lrl,
        "zz_dev_url": zz_dev_url,
        "zz_dev_ov": zz_dev_ov,
        "zz_dev_abs_ov": zz_dev_abs_ov,
        "zz_dev_abs_cutoff_ov": zz_dev_abs_cutoff_ov,
    }


def _summarize_group(df, label):
    """Compute summary stats for a group of results."""
    vals = df["zz_dev_abs_cutoff_ov"].dropna()
    return {
        "Group": label,
        "N": len(df),
        "N_excluded": len(df) - len(vals),
        "Mean": vals.mean() if len(vals) > 0 else np.nan,
        "Median": vals.median() if len(vals) > 0 else np.nan,
        "P95": vals.quantile(0.95) if len(vals) > 0 else np.nan,
    }


def compute_benchmark_score(results_df):
    """Compute per-biomarker, per-distribution, and overall benchmark scores."""
    rows = []

    # Per biomarker
    for analyte in BIOMARKERS:
        df_a = results_df[results_df["Analyte"] == analyte]
        if len(df_a) > 0:
            rows.append(_summarize_group(df_a, analyte))

    # Overall
    rows.append(_summarize_group(results_df, "Overall"))

    biomarker_df = pd.DataFrame(rows)

    # Per distribution type
    dist_rows = []
    for dist in DIST_TYPES:
        df_d = results_df[results_df["Distribution"] == dist]
        if len(df_d) > 0:
            dist_rows.append(_summarize_group(df_d, dist))
    dist_rows.append(_summarize_group(results_df, "Overall"))
    dist_df = pd.DataFrame(dist_rows)

    return biomarker_df, dist_df


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------
def remove_outliers_tukey_log(data_vec):
    """Tukey 1.5x IQR outlier removal on log-scale."""
    log_vec = np.log(data_vec)
    q1, q3 = np.percentile(log_vec, [25, 75])
    iqr = q3 - q1
    mask = (log_vec >= q1 - 1.5 * iqr) & (log_vec <= q3 + 1.5 * iqr)
    return data_vec[mask]


# ---------------------------------------------------------------------------
# Subset filtering
# ---------------------------------------------------------------------------
def load_and_filter_testsets(subset, spec_csv):
    """Load test set specifications and apply subset filter."""
    all_testsets = pd.read_csv(spec_csv)

    if subset == "all":
        return all_testsets, all_testsets
    elif subset == "test":
        # 3 per biomarker (matching R's defineSubset with N=3, seed=123)
        rng = np.random.RandomState(123)
        indices = []
        for analyte in all_testsets["Analyte"].unique():
            analyte_indices = all_testsets[all_testsets["Analyte"] == analyte]["Index"].values
            chosen = rng.choice(analyte_indices, size=min(3, len(analyte_indices)), replace=False)
            indices.extend(chosen)
        filtered = all_testsets[all_testsets["Index"].isin(indices)]
        return filtered, all_testsets
    elif subset in DIST_TYPES:
        filtered = all_testsets[all_testsets["Distribution"] == subset]
        return filtered, all_testsets
    elif subset in BIOMARKERS:
        filtered = all_testsets[all_testsets["Analyte"] == subset]
        return filtered, all_testsets
    else:
        try:
            n = int(subset)
            rng = np.random.RandomState(123)
            indices = []
            for analyte in all_testsets["Analyte"].unique():
                analyte_indices = all_testsets[all_testsets["Analyte"] == analyte]["Index"].values
                chosen = rng.choice(analyte_indices, size=min(n, len(analyte_indices)), replace=False)
                indices.extend(chosen)
            filtered = all_testsets[all_testsets["Index"].isin(indices)]
            return filtered, all_testsets
        except ValueError:
            raise ValueError(f"Unknown subset: {subset}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_benchmark(args):
    models_to_run = {}
    if args.v1_model_path:
        models_to_run["rinet_v1"] = args.v1_model_path
    if args.v2_model_path:
        models_to_run["rinet_v2"] = args.v2_model_path

    if not models_to_run:
        print("ERROR: At least one of --v1-model-path or --v2-model-path must be provided.")
        sys.exit(1)

    if not args.data_dir:
        print("ERROR: --data-dir is required (path to directory containing analyte subfolders "
              "and SpecificationTestSets.csv).")
        sys.exit(1)

    # Resolve spec CSV from the data directory (parent of analyte subfolders)
    data_dir = Path(args.data_dir)
    spec_csv = data_dir / "SpecificationTestSets.csv"
    if not spec_csv.exists():
        # Try one level up (if --data-dir points to Data/ subfolder)
        spec_csv = data_dir.parent / "SpecificationTestSets.csv"
    if not spec_csv.exists():
        print(f"ERROR: SpecificationTestSets.csv not found in {data_dir} or {data_dir.parent}")
        sys.exit(1)

    # Load test set specs
    testsets, all_testsets = load_and_filter_testsets(args.subset, spec_csv)
    if args.exclude:
        testsets = testsets[~testsets["Analyte"].isin(args.exclude)]
    print(f"=== RIBench Evaluation (Python) ===")
    print(f"Models: {', '.join(models_to_run.keys())}")
    print(f"Subset: {args.subset} ({len(testsets)} test sets)")
    if args.exclude:
        print(f"Excluded: {', '.join(args.exclude)}")
    if args.log_scale_explicit:
        print(f"Log-scale: {'ON' if args.log_scale else 'OFF'} (forced for all models)")
    else:
        print(f"Log-scale: per-model default (V1=OFF, V2=ON)")
    print(f"Outlier removal: {'ON' if args.outliers else 'OFF'}")
    print()

    # Load or generate data
    if args.data_dir:
        print(f"Loading pre-generated data from: {args.data_dir}")
        test_data = load_r_data(args.data_dir, testsets)
        print(f"Loaded {len(test_data)} test sets.\n")
    else:
        print("Generating test data from specification...")
        test_data = generate_all_test_data(testsets, all_testsets)
        print(f"Generated {len(test_data)} test sets.\n")

    # Load models
    pipelines = {}
    for model_name, model_path in models_to_run.items():
        print(f"Loading {model_name} from {model_path}...")
        if model_name == "rinet_v1":
            pipeline = RINetV1Pipeline()
            pipeline.load(model_path, model_file=args.v1_model_file)
        elif model_name == "rinet_v2":
            pipeline = RINetV2Pipeline(ndim=1)
            pipeline.load(model_path, model_file=args.v2_model_file)
        pipelines[model_name] = pipeline
        # Warm up with dummy prediction
        try:
            dummy = np.exp(np.random.normal(2, 0.5, size=100))
            if model_name == "rinet_v1":
                warmup_log = args.log_scale if args.log_scale_explicit else False
            else:
                warmup_log = args.log_scale if args.log_scale_explicit else True
            pipeline.predict(dummy, log_scale=warmup_log, verbose=0)
        except Exception:
            pass
    print()

    # Run predictions
    all_results = []

    for model_name, pipeline in pipelines.items():
        print(f"Running {model_name} on {len(test_data)} test sets...")
        t_start = time.time()
        n_success = 0
        n_fail = 0

        for i, (_, row) in enumerate(testsets.iterrows()):
            idx = int(row["Index"])
            analyte = row["Analyte"]
            key = (idx, analyte)

            if key not in test_data:
                n_fail += 1
                continue

            data_vec = test_data[key].copy()

            # Remove non-positive values (for log-scale prediction)
            data_vec = data_vec[data_vec > 0]

            # Optional outlier removal
            if args.outliers and len(data_vec) > 10:
                data_vec = remove_outliers_tukey_log(data_vec)

            # Predict — V1 uses linear std (no log), V2 uses log-scale by default
            if model_name == "rinet_v1":
                log_scale = args.log_scale if args.log_scale_explicit else False
                ri = predict_v1(pipeline, data_vec, log_scale=log_scale)
            elif model_name == "rinet_v2":
                log_scale = args.log_scale if args.log_scale_explicit else True
                ri = predict_v2(pipeline, data_vec, log_scale=log_scale)
            else:
                ri = None

            if ri is not None and not np.any(np.isnan(ri)):
                est_lrl, est_url = ri[0], ri[1]
                n_success += 1
            else:
                est_lrl, est_url = np.nan, np.nan
                n_fail += 1

            # Compute z-score deviations
            zdev = compute_zdev(
                gt_lrl=row["GT_LRL"], gt_url=row["GT_URL"],
                est_lrl=est_lrl, est_url=est_url,
                nonp_mu=row["nonp_mu"], nonp_sigma=row["nonp_sigma"],
                nonp_lambda=row["nonp_lambda"], nonp_shift=row["nonp_shift"],
                analyte=analyte,
            )

            result_row = {
                "Model": model_name,
                "Index": idx,
                "Analyte": analyte,
                "Distribution": row["Distribution"],
                "fractionPathol": row["fractionPathol"],
                "N": int(row["N"]),
                "GT_LRL": row["GT_LRL"],
                "GT_URL": row["GT_URL"],
                "Est_LRL": est_lrl,
                "Est_URL": est_url,
                **zdev,
            }
            all_results.append(result_row)

            # Progress
            if (i + 1) % 50 == 0 or (i + 1) == len(testsets):
                print(f"\r  Progress: {i + 1}/{len(testsets)} ({100 * (i + 1) / len(testsets):.0f}%)", end="")

        elapsed = time.time() - t_start
        print(f"\n  Done in {elapsed:.1f}s. Success: {n_success}, Failed: {n_fail}\n")

    results_df = pd.DataFrame(all_results)

    # Output results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-model benchmark scores
    for model_name in pipelines:
        model_results = results_df[results_df["Model"] == model_name]
        by_biomarker, by_dist = compute_benchmark_score(model_results)

        print(f"=== Benchmark Score: {model_name} ===")
        print()
        print("By biomarker:")
        print(by_biomarker.to_string(index=False))
        print()
        print("By distribution:")
        print(by_dist.to_string(index=False))
        print()

        # Save
        by_biomarker.to_csv(output_dir / f"score_by_biomarker_{model_name}.csv", index=False)
        by_dist.to_csv(output_dir / f"score_by_distribution_{model_name}.csv", index=False)
        model_results.to_csv(output_dir / f"detailed_results_{model_name}.csv", index=False)

    # Comparison table if both models ran
    if len(pipelines) > 1:
        print("=== Model Comparison (Mean / Median |z-dev|) ===")
        comparison_rows = []
        for label in BIOMARKERS + ["Overall"]:
            row_data = {"Group": label}
            for model_name in pipelines:
                if label == "Overall":
                    vals = results_df[results_df["Model"] == model_name]["zz_dev_abs_cutoff_ov"].dropna()
                else:
                    mask = (results_df["Model"] == model_name) & (results_df["Analyte"] == label)
                    vals = results_df[mask]["zz_dev_abs_cutoff_ov"].dropna()
                if len(vals) > 0:
                    row_data[f"{model_name}_mean"] = f"{vals.mean():.4f}"
                    row_data[f"{model_name}_median"] = f"{vals.median():.4f}"
                else:
                    row_data[f"{model_name}_mean"] = "N/A"
                    row_data[f"{model_name}_median"] = "N/A"
            comparison_rows.append(row_data)
        comparison_df = pd.DataFrame(comparison_rows)
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
        print()

    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="RIBench benchmark evaluation for RINet V1 and V2 models"
    )
    parser.add_argument("--v1-model-path", type=str, default=None,
                        help="Path to V1 model directory (weights, scaler, quantiles)")
    parser.add_argument("--v2-model-path", type=str, default=None,
                        help="Path to V2 model directory (weights/model, scaler)")
    parser.add_argument("--v1-model-file", type=str, default="model.keras",
                        help="V1 model filename within the model directory (default: model.keras)")
    parser.add_argument("--v2-model-file", type=str, default="model.keras",
                        help="V2 model filename within the model directory (default: model.keras)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to pre-generated RIBench data CSVs (from R)")
    parser.add_argument("--subset", type=str, default="all",
                        help="Subset: 'all', 'test', integer, biomarker name, or distribution type")
    parser.add_argument("--exclude", type=str, nargs="+", default=None,
                        help="Analytes to exclude (e.g. --exclude CRP LDH)")
    parser.add_argument("--outliers", action="store_true",
                        help="Enable Tukey 1.5x IQR outlier removal on log-scale")
    parser.add_argument("--no-log-scale", dest="log_scale", action="store_false",
                        help="Force linear standardization (no log) for all models. "
                             "Without this flag, V1 defaults to linear and V2 to log-scale.")
    parser.add_argument("--log-scale", dest="log_scale", action="store_true",
                        help="Force log-scale preprocessing for all models.")
    parser.set_defaults(log_scale=None)
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "evaluation" / "benchmarking" / "results"),
                        help="Directory to save results")

    args = parser.parse_args()
    # Track whether log_scale was explicitly set by the user
    args.log_scale_explicit = args.log_scale is not None
    if args.log_scale is None:
        args.log_scale = True  # fallback default (overridden per-model when not explicit)
    run_benchmark(args)


if __name__ == "__main__":
    main()
