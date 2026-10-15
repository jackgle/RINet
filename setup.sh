#!/usr/bin/env bash
# Creates an isolated conda env (Python + R) for RINet and installs the package.
# Run from the repo root in a terminal; no env needs to be active beforehand.
# On SageMaker the env then shows up as the "conda_rinet" kernel.
set -euo pipefail

ENV_NAME="${ENV_NAME:-rinet}"

echo "=== Setting up environment: $ENV_NAME ==="

eval "$(conda shell.bash hook)"

# keep everything from conda-forge so Python and R share one consistent set of libs
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" -c conda-forge --override-channels \
        python=3.11 r-base=4.5.2 ipykernel
fi
conda activate "$ENV_NAME"

# R packages go into the env's own R library
if [ ! -d ./refineR ]; then
    git clone https://github.com/jackgle/refineR.git
fi
Rscript -e "install.packages(c('remotes', 'ash', 'future', 'reflimR'), repos='https://cloud.r-project.org/')"
Rscript -e "remotes::install_local('./refineR/', upgrade='never')"

# Python package; pull in pip-packaged CUDA libs if a GPU is present
python -m pip install --upgrade pip
if command -v nvidia-smi > /dev/null 2>&1; then
    python -m pip install -e ".[gpu]"
else
    python -m pip install -e .
fi

echo "=== Setup complete: select the 'conda_$ENV_NAME' kernel in Jupyter ==="
