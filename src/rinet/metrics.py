# functions used for evaluating Gaussian or RI estimates

import numpy as np


def bhattacharyya_gaussian_distance(mu1, cov1, mu2, cov2):
    """
    Compute the Bhattacharyya distance between two Gaussian distributions.

    Parameters:
    mu1 : array_like
        Mean of the first Gaussian distribution.
    cov1 : array_like
        Covariance matrix of the first Gaussian distribution.
    mu2 : array_like
        Mean of the second Gaussian distribution.
    cov2 : array_like
        Covariance matrix of the second Gaussian distribution.

    Returns:
    float
        Bhattacharyya distance between the two Gaussian distributions.
    """
    
    if cov1.shape==(2,2):
        assert mu1.shape==(2,), 'Error: single value given for mean'

    cov = (1 / 2) * (cov1 + cov2)

    T1 = (1 / 8) * (
        (mu1 - mu2) @ np.linalg.inv(cov) @ (mu1 - mu2).T
    )
    T2 = (1 / 2) * np.log(
        np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
    )

    return T1 + T2


def kl_gaussian_divergence(mu1, cov1, mu2, cov2):
    """
    Compute the Kullback-Leibler divergence between two Gaussian distributions.

    Parameters:
    mu1 : array_like
        Mean of the first Gaussian distribution.
    cov1 : array_like
        Covariance matrix of the first Gaussian distribution.
    mu2 : array_like
        Mean of the second Gaussian distribution.
    cov2 : array_like
        Covariance matrix of the second Gaussian distribution.

    Returns:
    float
        Kullback-Leibler divergence between the two Gaussian distributions.
    """

    k = len(mu1)
    cov2_inv = np.linalg.inv(cov2)
    
    T1 = np.trace(
        cov2_inv @ cov1
    )
    
    T2 = (mu2 - mu1) @ cov2_inv @ (mu2 - mu1).T
    
    T3 = np.log(
        np.linalg.det(cov2) / np.linalg.det(cov1)
    )

    return 1/2 * (T1 - k + T2 + T3)


def norm_err(y_test, p_test):
    """
    Calculates the normalized error between a true and predicted reference interval as proposed in CA-125 paper submitted to Sci. Reports
    
    Arguments:
        y_test:   true RI
        p_test:   predicted RI
        
    Returns:
        normalized error
    """
    assert len(y_test)==2, 'Error: len(y_test) not equal to 2'
    assert len(p_test)==2, 'Error: len(p_test) not equal to 2'
    
    return np.mean([np.abs(i-j) for i,j in zip([-1, 1], (p_test-y_test.mean())/y_test.std())])/2


