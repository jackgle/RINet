import numpy as np
from scipy.stats import norm
from scipy.stats import chi2
from matplotlib.patches import Ellipse


def correlation_to_covariance(correlation_matrix, std_devs):
    '''
    Converts a correlation matrix to covariance matrix

    Args:
        correlation_matrix: correlation matrix
        std_devs: vector of variable standard deviations
    Returns:
        covariance_matrix
    '''

    # calculate covariance matrix using the formula: cov(X, Y) = corr(X, Y) * std_dev(X) * std_dev(Y)
    covariance_matrix = np.outer(std_devs, std_devs) * correlation_matrix

    return covariance_matrix


def stats_to_ri(mean, std_dev):
    """
    Converts mean and standard deviation to reference interval (central 95%)

    """
    return np.array([norm.ppf(0.025, loc=mean, scale=std_dev), norm.ppf(0.975, loc=mean, scale=std_dev)])


def plot_cov_ellipse(mean, cov, percentage=0.95, **kwargs):
    """
    Plots an ellipse representing the covariance matrix `cov` centered at `mean`.

    Parameters:
    - cov: 2x2 covariance matrix
    - mean: 2-element array-like representing the center of the ellipse
    - P: Percentage of density to enclose
    - kwargs: Additional keyword arguments to be passed to Ellipse patch.

    Returns:
    - The matplotlib Ellipse patch object.
    """
    # Get number of stds for ellipse axes based on percentage argument
    nstd = np.sqrt(chi2.ppf(percentage, 2))

    # Eigenvalues and eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Compute the angle of the ellipse
    angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))

    # Width and height of the ellipse
    width, height = 2 * nstd * np.sqrt(eigvals)

    # Create the ellipse patch
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)

    return ellipse

