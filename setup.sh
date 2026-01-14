#!/usr/bin/env bash
set -e  # stop on error

echo "=== Setting up environment ==="

conda update -n base -c conda-forge conda -y
conda install -c conda-forge r-base=4.5.1 -y

mkdir -p ~/R/library
echo 'R_LIBS_USER="~/R/library"' >> ~/.Renviron

# # Rscript -e "install.packages('refineR', repos='https://cloud.r-project.org/', lib='~/R/library')"
# # git clone https://github.com/jackgle/refineR.git
Rscript -e "install.packages('remotes', repos='https://cloud.r-project.org/', lib='~/R/library')"
Rscript -e "remotes::install_local('./refineR/', lib='~/R/library', upgrade='never')"
Rscript -e "install.packages('ash', repos='https://cloud.r-project.org/', lib='~/R/library')"
Rscript -e "install.packages('future', repos='https://cloud.r-project.org/', lib='~/R/library')"
Rscript -e "install.packages('reflimR', repos='https://cloud.r-project.org/', lib='~/R/library')"

pip install -e .

echo "=== Setup complete ==="
