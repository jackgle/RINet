# RINet

RINet is a deep learning framework for indirect estimation of clinical reference intervals (RIs) using synthetic data–driven training.

The repository contains code used in the accompanying publication and supports data simulation, model training, hyperparameter search, and benchmarking.

For R implementation: Install from CRAN with `install.packages("rinet")` or see the [source repository](https://github.com/jackgle/rinet_r_package)

For RINetv1 code, please see this [repository](https://github.com/jackgle/rinet_v1)


## Project Structure

```
src/rinet/            # Main package code
data/
  simulated/scripts/  # Data simulation scripts
  liver/              # Real-world data preprocessing
modeling/             # Bayesian hyperparameter search
evaluation/
  benchmarking/       # Method benchmarking
```

## Installation

```bash
./setup.sh
```

## Citation

If you use this code in academic work, please cite the associated paper:

LeBien, J., Velev, J., & Roche-Lima, A. (2026). RINet: synthetic data training for indirect estimation of clinical reference distributions. Journal of biomedical informatics, 104980. https://doi.org/10.1016/j.jbi.2026.104980
