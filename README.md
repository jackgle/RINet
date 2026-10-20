# RINet

RINet is a deep learning framework for indirect estimation of clinical reference intervals (RIs) using synthetic data–driven training.

The repository contains code used in the accompanying publication and supports data simulation, model training, hyperparameter search, and benchmarking.

For R implementation: see the [source repository](https://github.com/jackgle/rinet_r_package)

For RINetv1 code, please see this [repository](https://github.com/jackgle/rinet_v1)

## Update: model v2.1

The simulation and training code in this repository has been updated since the publication. Pathological component shifts and widths are now drawn relative to the reference component, and the 1D training set is larger, which gives an updated 1D model (v2.1).

- **R package:** the new version of the [rinet R package](https://github.com/jackgle/rinet_r_package) uses the v2.1 model by default, and the model from the publication remains available as `model_version = "v2.0"`. This version is under review at CRAN. Until it is released there (`install.packages("rinet")` currently installs the previous release), install it from the main branch of the source repository:

  ```r
  install.packages("remotes")
  remotes::install_github("jackgle/rinet_r_package")
  ```

- **Reproducing the publication:** the code that reproduces the results reported in the publication is tagged [`jbi-2026`](https://github.com/jackgle/RINet/tree/jbi-2026) (`git checkout jbi-2026`).


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
./setup.sh   # creates a "rinet" conda env; then pick the conda_rinet kernel
```

## Citation

If you use this code in academic work, please cite the associated paper:

LeBien, J., Velev, J., & Roche-Lima, A. (2026). RINet: synthetic data training for indirect estimation of clinical reference distributions. Journal of biomedical informatics, 104980. https://doi.org/10.1016/j.jbi.2026.104980
