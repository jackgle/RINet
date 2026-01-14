import numpy as np
import pandas as pd
from itertools import combinations


def bootstrap_paired_diff(errors_dict, n_boot=1000, ci=95, stat_func=np.mean):
    """
    Pairwise bootstrap comparison of errors, optionally accounting for group structure,
    using a specified statistic (mean, median, etc.).

    Parameters:
        errors_dict (dict): Keys are method names, values are lists or arrays of errors.
        n_boot (int): Number of bootstrap samples.
        ci (float): Confidence interval percentage (default 95).
        stat_func (callable): Function to compute the statistic of interest (default np.mean).

    Returns:
        pd.DataFrame with columns:
            'method_1', 'method_2', 'stat_diff', 'ci_lower', 'ci_upper', 'significant'
    """
    methods = list(errors_dict.keys())
    results = []

    for method1, method2 in combinations(methods, 2):
        errors1 = np.array(errors_dict[method1])
        errors2 = np.array(errors_dict[method2])

        # handle NaNs pairwise
        mask = ~np.isnan(errors1) & ~np.isnan(errors2)
        errors1 = errors1[mask]
        errors2 = errors2[mask]

        diff = errors1 - errors2
        boot_stats = []

        for _ in range(n_boot):
            idx = np.random.choice(len(diff), size=len(diff), replace=True)
            boot_stats.append(stat_func(diff[idx]))

        lower = np.percentile(boot_stats, (100-ci)/2)
        upper = np.percentile(boot_stats, 100-(100-ci)/2)
        stat_diff = stat_func(diff)

        significant = (lower > 0) or (upper < 0)

        results.append({
            'method_1': method1,
            'method_2': method2,
            'stat_diff': stat_diff,
            'ci_lower': lower,
            'ci_upper': upper,
            'significant': significant
        })

    return pd.DataFrame(results)


def bootstrap_method_mean(errors, groups=None, n_boot=1000, ci=95):
    """
    Compute mean and bootstrap confidence interval for a single method, optionally grouped.

    Parameters:
        errors (list or np.array): List of errors for the method.
        groups (list or np.array, optional): Group assignment for each error. Must be same length as errors.
        n_boot (int): Number of bootstrap samples.
        ci (float): Confidence interval percentage.

    Returns:
        dict: {'mean': ..., 'ci_lower': ..., 'ci_upper': ...}
    """
    errors = np.array(errors)
    
    # Remove NaNs
    mask = ~np.isnan(errors)
    errors = errors[mask]
    if groups is not None:
        groups = np.array(groups)[mask]

    if groups is not None:
        unique_groups = np.unique(groups)
        # Compute mean per group
        group_means = np.array([errors[groups == g].mean() for g in unique_groups])
    else:
        # Treat all errors as independent
        group_means = errors

    # Bootstrap group means
    boot_means = []
    for _ in range(n_boot):
        idx = np.random.choice(len(group_means), size=len(group_means), replace=True)
        boot_means.append(group_means[idx].mean())

    lower = np.percentile(boot_means, (100-ci)/2)
    upper = np.percentile(boot_means, 100-(100-ci)/2)
    mean_val = group_means.mean()

    return {'mean': mean_val, 'ci_lower': lower, 'ci_upper': upper}


