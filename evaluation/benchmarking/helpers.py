import pickle
import numpy as np
import pandas as pd
from scipy.stats import skew, boxcox
from rinet.utils import stats_to_ri, correlation_to_covariance
from itertools import combinations


def save_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def flag_outliers(x):
    """
    Remove's outliers using Tukey's rule:
        < q1-1.5*iqr
        > q3+1.5*iqr

    Log transform is applied to normalize before filtering outliers

    Args:
        data: numpy array (n_samples, n_variables)
    Returns:
        is_outlier: boolean array of predicted outliers
    """
    data = x.copy()
    is_outlier = np.zeros(data.shape[0], dtype=bool)
    for i in range(data.shape[1]):
        if data[:, i].min() <= 0:  # handle non-positive data
            data[:, i] = data[:, i] - data[:, i].min() + 1e-9
        skewness = skew(data[:, i])
        tdata, _ = boxcox(data[:, i])
        q1 = np.quantile(tdata, 0.25)
        q3 = np.quantile(tdata, 0.75)
        iqr = q3 - q1
        if skewness >= 1:  # if positively skewed, only remove from right tail
            outlier_i = np.array(tdata > (q3+(iqr*1.5)))
        elif skewness <= -1:  # if negatively skewed, only remove from left tail
            outlier_i = np.array(tdata < (q1-(iqr*1.5)))
        else:
            outlier_i = np.array(tdata < (q1-(iqr*1.5))) | np.array(tdata > (q3+(iqr*1.5)))
        is_outlier = is_outlier | outlier_i
    return is_outlier


def get_targets_simulated(path):
    """Prepares targets for simulated dataset"""
    means = load_pickle(f"{path}/means.pkl")
    stds = load_pickle(f"{path}/stds.pkl")
    if '2d' in path:
        corr_mats = load_pickle(f"{path}/corr_mats.pkl")
    if '1d' in path:
        targets_stats = [[i[0], j[0]] for i, j in zip(means, stds)]
        targets_ri = [np.squeeze(stats_to_ri(i[0], i[1])) for i in targets_stats]
        return targets_stats, targets_ri
    elif '2d' in path:
        targets_stats = [
            [
                i[0],
                correlation_to_covariance(j[0], k[0])
            ] for i, j, k in zip(means, corr_mats, stds)
        ]
        return targets_stats


def out_to_ri(model, prediction):
    if prediction is None:
        return None
    if model in ['rinet_v1', 'rinet_v1_log']:
        p = np.array([prediction[0.025], prediction[0.975]])
    elif model in ['rinet_v2', 'gmm']:
        p = stats_to_ri(prediction['mean'][0], prediction['std'][0])
    elif model in ['refineR', 'reflimR']:
        p = np.array([prediction['ri_low'], prediction['ri_high']])
    else:
        raise ValueError(f"Unknown model: {model}")
    return p


class BaseDatasetAdapter:
    """Default: identity transforms, fully compatible."""

    def transform_in(self, data, model_key):
        return data

    def transform_out(self, preds, data, model_key):
        return preds

    def is_compatible(self, model_key):
        return True


class Liver1DAdapter(BaseDatasetAdapter):
    def transform_in(self, data, model_key):
        if model_key in ["gmm", "rinet_v2"]:
            return np.log(data)
        return data

    def transform_out(self, preds, data, model_key):
        if model_key in ["gmm", "rinet_v2"]:
            return np.exp(preds)
        return preds

    def extra_kwargs(self, model_key):
        if model_key == "rinet_v1_log":
            return {"log_scale": True}
        return {}


class Simulated1DAdapter(BaseDatasetAdapter):
    def transform_in(self, data, model_key):
        if model_key in ["refineR", "reflimR"]:
            offset = data.min() - 1e-9
            return data - offset
        return data

    def transform_out(self, preds, data, model_key):
        if model_key in ["refineR", "reflimR"]:
            offset = data.min() - 1e-9
            return preds + offset
        return preds

    def extra_kwargs(self, model_key):
        if model_key == "refineR":
            return {"fix_lambda": True}
        return {}


# all bivariate analyses are done in log space
class Liver2DAdapter(BaseDatasetAdapter):
    def is_compatible(self, model_key):
        return model_key not in ["rinet_v1", "rinet_v1_log", "refineR", "reflimR"]


class Simulated2DAdapter(BaseDatasetAdapter):
    def is_compatible(self, model_key):
        return model_key not in ["rinet_v1", "rinet_v1_log", "refineR", "reflimR"]



