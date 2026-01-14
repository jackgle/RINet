import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import sqrtm
from rinet.utils import plot_cov_ellipse


def random_correlation_matrix(n_dimensions, max_corr=1.0):
    """Generates a random correlation matrix in N dimensions"""
    # for 2D, just sample uniformly
    if n_dimensions == 2:
        correlation = np.random.uniform(-max_corr, max_corr)
        return np.array([[1.0, correlation], [correlation, 1.0]])
    # for >2D, use Onion Method
    else:
        if max_corr == 1.0:
            return random_corr_mat_onion(n_dimensions)
        else:  # to limit max correlation, repeat sampling until it's satisfied
            corr_mat = random_corr_mat_onion(n_dimensions)
            while np.abs(corr_mat[~np.eye(n_dimensions, dtype=bool)]).max() > max_corr:
                corr_mat = random_corr_mat_onion(n_dimensions)
            return corr_mat


def random_corr_mat_onion(n_dimensions):
    """
    Implements the onion method for uniform sampling in correlation matrix space

    Taken from: https://gmarti.gitlab.io/stats/2018/10/05/onion-method.html
    """
    d = n_dimensions + 1
    prev_corr = np.matrix(np.ones(1))
    for k in range(2, d):
        # sample y = r^2 from a beta distribution
        # with alpha_1 = (k-1)/2 and alpha_2 = (d-k)/2
        y = np.random.beta((k - 1) / 2, (d - k) / 2)
        r = np.sqrt(y)

        # sample a unit vector theta uniformly
        # from the unit ball surface B^(k-1)
        v = np.random.randn(k-1)
        theta = v / np.linalg.norm(v)

        # set w = r theta
        w = np.dot(r, theta)

        # set q = prev_corr**(1/2) w
        q = np.dot(sqrtm(prev_corr), w)

        next_corr = np.zeros((k, k))
        next_corr[:(k-1), :(k-1)] = prev_corr
        next_corr[k-1, k-1] = 1
        next_corr[k-1, :(k-1)] = q
        next_corr[:(k-1), k-1] = q

        prev_corr = next_corr
    return next_corr


def correlation_to_covariance(corr, stds):
    """Converts a correlation matrix and standard deviation array to a covariance matrix"""
    cov = corr * np.outer(stds, stds)
    return cov


def random_covariance_matrix(n_dimensions, std_range=[0.5, 1.5], max_corr=0.9):
    """Generates a random n-dimensional covariance matrix."""
    correlation_matrix = random_correlation_matrix(n_dimensions, max_corr)
    std_devs = np.random.uniform(std_range[0], std_range[1], n_dimensions)
    covariance_matrix = correlation_to_covariance(correlation_matrix, std_devs)
    return covariance_matrix, correlation_matrix, std_devs


def random_gaussian_sample(
    n_dimensions, mean_range=[-1.5, 1.5], 
    std_range=[0.5, 1.5], max_corr=0.9, 
    n_samples=1000
):
    """
    Generates a sample from a random multivariate Gaussian 

    Args:
        n_dimensions:       matrix dimensionality
        mean_range:         min/max mean of Gaussian
        std_range:          min/max std of Gaussian
        n_samples:          sample size
    Returns:
        sample:             vector of points
        mean_vector:        mean vector
        correlation_matrix: correlation matrix of random Gaussian
    """
    # randomly sample mean
    mean_vector = np.random.uniform(mean_range[0], mean_range[1], n_dimensions)

    if n_dimensions >= 2:
        # generate a random correlation matrix
        covariance_matrix, correlation_matrix, std_devs = random_covariance_matrix(n_dimensions, std_range, max_corr)
        # take sample
        sample = np.random.multivariate_normal(mean_vector, covariance_matrix, n_samples)
    else:
        # generate random standard deviations
        std_devs = np.random.uniform(std_range[0], std_range[1], n_dimensions)
        # take sample
        sample = np.random.normal(mean_vector, std_devs, (n_samples, n_dimensions))
        correlation_matrix = None

    return sample, mean_vector, std_devs, correlation_matrix


def random_deviation_mean_std(array, stat_range, percent_range):
    """Adds random variation to a given mean or standard deviation array.
       The deltas are within a percent of the original possible range of values (e.g. 5% of [-1, 1] = 0.1)
    """
    # calculate the lower and upper possible absolute changes
    delta_interval = (percent_range / 100) * (stat_range[1]-stat_range[0])
    deltas = np.random.uniform(delta_interval[0], delta_interval[1], size=array.shape)  # random sample changes
    deltas *= np.random.choice([-1, 1], size=array.shape)  # randomize positive or negative deltas
    # add deltas to get the new array
    new_array = array + deltas
    # limit values?
    new_array = np.maximum(stat_range[0], np.minimum(new_array, stat_range[1]))  # ensure changes don't exceed chosen limits
    return new_array


def quantize_data(data, step):
    """
    Quantize a numpy array to a certain bin size.

    Parameters:
    data (numpy.ndarray): The input data to be quantized.
    step (float): The bin size for quantization.

    Returns:
    numpy.ndarray: The quantized data.
    """
    return np.round(data / step) * step


def random_mixture_sample(
    n_dimensions=1,             # number of dimensions
    n_components=2,             # number of pathological components
    p_frac_range=[0.01, 0.40],  # min/max pathological fraction
    bg_frac_range=[0.01, 0.1],  # min/max background noise fraction
    bg_scale_range=[5, 10],     # min/max stds from mixture mean for background noise
    max_corr=0.8,               # max variable correlation
    std_range=[0.5, 2.0],       # min/max Gaussian stdevs
    mean_range=[-2.5, 2.5],     # min/max Gaussian means
    n_samples=10000,            # no. samples
    quantize_step=0,            # bin size for quantizing
    pathological_dev_percentage_range=[10, 40]  # percentage pathological component statistics will deviate from reference for >=1D
):
    """
    Generates a sample from a mixture of random Gaussians

    Args:
        See above
    Returns:
        mixture_sample:   points sampled from a random mixture of Gaussians
        component_sizes:   size of each Gaussian component, beginning with reference
        mean_vectors:     list of mean vectors of each Gaussian
        corr_matrices:    list of correlation matrices of each Gaussian
    """

    assert n_dimensions <= 2, (
        "Warning: 3-D data generation is in progress "
        "and the current method may lead to unrealistically small determinants"
    )

    # hold component for each point
    components = []

    # hold statistics for each component
    corr_matrices = []
    std_vectors = []
    mean_vectors = []

    # random reference component
    if n_components > 0:
        p_frac = np.random.uniform(p_frac_range[0], p_frac_range[1])
    else:
        p_frac = 0
    bg_frac = np.random.uniform(bg_frac_range[0], bg_frac_range[1])
    ref_size = int(n_samples*(1-p_frac-bg_frac))
    mixture_sample, mean_vector, std_dev, corr_mat = random_gaussian_sample(
        n_dimensions=n_dimensions,
        mean_range=mean_range,
        std_range=std_range,
        max_corr=max_corr,
        n_samples=ref_size
    )

    components += list(np.zeros(len(mixture_sample)))  # store parameters
    corr_matrices.append(corr_mat)
    mean_vectors.append(mean_vector)
    std_vectors.append(std_dev)

    # pathological components
    p_ratios = np.random.uniform(0, 1, n_components)
    p_ratios /= np.sum(p_ratios)
    for i in range(len(p_ratios)):

        p_size = int(n_samples*p_frac*p_ratios[i])  # get pathological component size

        # deviate reference parameters based on a 
        # percentage of the given range
        if pathological_dev_percentage_range is not None:
            pathological_dev_percentage_range = np.array(pathological_dev_percentage_range)

            # deviate the mean and standard deviation
            mv = random_deviation_mean_std(
                mean_vector, mean_range,
                pathological_dev_percentage_range
            )
            std = random_deviation_mean_std(
                std_dev, std_range,
                pathological_dev_percentage_range
            )

            if n_dimensions > 1:  # convert correlation to covariance for >1D
                cvm = correlation_to_covariance(corr_mat, std)

            if n_dimensions > 1:  # take sample
                gauss_sample = np.random.multivariate_normal(mv, cvm, p_size)
            else:
                gauss_sample = np.random.normal(mv, std, (p_size, n_dimensions))

        else:  # continue to sample uniformly across the given ranges
            gauss_sample, mv, std, cm = random_gaussian_sample(
                n_dimensions=n_dimensions, 
                mean_range=mean_range, 
                std_range=std_range, 
                n_samples=p_size
            )

        corr_matrices.append(corr_mat)  # store parameters
        mean_vectors.append(mv)
        std_vectors.append(std)

        mixture_sample = np.vstack([mixture_sample, gauss_sample])
        components += list(np.ones(len(gauss_sample))*(i+1))

    # background noise
    mixture_mean = mixture_sample.mean(axis=0)
    mixture_std = mixture_sample.std(axis=0)
    bg_noise = np.zeros((int(n_samples*bg_frac), n_dimensions))
    for i in range(n_dimensions):
        bg_scale = np.random.uniform(bg_scale_range[0], bg_scale_range[1])
        bg_noise[:, i] = np.random.uniform(
            mixture_mean[i]-mixture_std[i]*bg_scale, 
            mixture_mean[i]+mixture_std[i]*bg_scale, 
            bg_noise.shape[0]
        )
    mixture_sample = np.vstack([mixture_sample, bg_noise])

    if quantize_step:
        mixture_sample = quantize_data(mixture_sample, quantize_step)

    components += list(np.ones(bg_noise.shape[0])*-1)
    components = np.array(components)

    component_sizes = [
        len(np.where(components == i)[0]) for i in
        sorted(list(set(components)))[1:]
    ]  # gaussian component sizes
    component_sizes.append(
        len(np.where(components == -1)[0])
    )  # background noise size

    if n_dimensions >= 2:
        return mixture_sample, mean_vectors, std_vectors, \
            corr_matrices, components, component_sizes
    else:
        return mixture_sample, mean_vectors, std_vectors, \
            components, component_sizes


def plot_1d_mixture(sample, sizes, means, stds, legend=False, scope=[-4, 4]):
    """
    Plots a histogram of a 1D Gaussian mixture sample, along with scaled PDFs for each Gaussian

    Arguments:
        sample:   array or list of sampled values
        sizes:    list containing size of each Gaussian components
        means:    list of Gaussian means
        stds:     list of Gaussian standard deviations
        legend:   whether to include a legend
        scope:    range of values to plot
    """

    # range for plotting distributions
    x_range = np.linspace(*scope, 51)

    # standardize
    stds = np.array([i/sample.std() for i in stds])
    means = np.array([(i-sample.mean())/sample.std() for i in means])
    sample = (sample-sample.mean())/sample.std()

    # plot the histogram
    plt.hist(
        sample, bins=x_range,
        density=True, alpha=0.7, color='lightskyblue', edgecolor='white',
        label='Mixture Sample Histogram'
    )

    # plot the PDF of the reference Gaussian
    pdf_values = sizes[0]/sum(sizes) * norm.pdf(x_range, means[0], stds[0])
    plt.plot(
        x_range, pdf_values, 'r-', 
        label='Reference Component', linewidth=2
    )

    # plot the PDFs of other Gaussians
    for i in range(len(means)-1):
        other_pdf_values = sizes[i+1]/sum(sizes) * \
            norm.pdf(x_range, means[i+1], stds[i+1])
        plt.plot(
            x_range, other_pdf_values, '--',
            label='Pathological Component '+str(i+1))

    # display legend and labels
    if legend:
        plt.legend()
    plt.xlim([min(x_range), max(x_range)]);
    plt.yticks([])


def plot_2d_mixture(
    sample, means, cov_mats,
    legend=False, scope=[-4, 4],
    standardize=True, scatter=True
):
    """
    Plots a 2D histogram of a 2D Gaussian mixture sample, along with contour lines for each Gaussian

    Arguments:
        sample:   array of sampled values
        means:    list of Gaussian mean vectors
        cov_mats: list of Gaussian covariance matrices
        legend:   whether to include a legend
        scope:    range of values to plot for both axes
        scatter:  whether to make a scatter plot instead of 2D histogram
    """

    # setup colors
    colors_ellipses = [
        'red', 'tab:blue', 'tab:orange'
    ]

    # standardize
    if standardize:
        a = 1/sample.std(axis=0)  # reciprocal of sample std for scaling
        b = -sample.mean(axis=0)  # negative of sample mean for shifting
        for i in range(len(cov_mats)):
            cov_mats[i] = [
                [cov_mats[i][0, 0]*a[0]**2, cov_mats[i][0, 1]*a[0]*a[1]],
                [cov_mats[i][1, 0]*a[0]*a[1], cov_mats[i][1, 1]*a[1]**2]
            ]
        means = [a*(i+b) for i in means]
        sample = a*(sample+b)

    if scatter:
        # do scatter plot instead
        plt.scatter(
            x=sample[:, 0],
            y=sample[:, 1],
            alpha=0.1,
            marker='.',
            color='black',
            s=10,
        )
    else:
        # 2D histogram visualization
        hist, xedges, yedges = np.histogram2d(
            sample[:, 0], sample[:, 1],
            bins=np.linspace(scope[0], scope[1], 51)
        )
        with np.errstate(divide='ignore'):
            plt.imshow(
                np.log(hist.T),
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap='binary', origin='lower'
            )

    # ellipses for reference distribution
    for comp in range(len(means)):  # don't plot background noise
        linestyle = '-'
        linewidth = 2
        label = 'Reference Gaussian'
        if comp > 0:
            linestyle = '--'
            linewidth = 1
            label = 'Pathological Gaussian'
        ellipse = plot_cov_ellipse(
            means[comp],
            cov_mats[comp],
            percentage=0.95,
            fc='none',
            edgecolor=colors_ellipses[comp],
            linestyle=linestyle,
            linewidth=linewidth)
        plt.gca().add_patch(ellipse)

    # create legend
    if legend:
        if scatter:
            plt.scatter(
                [], [],
                color='k', label='Sampled Values'
            )  # empty object for points
        else:
            plt.scatter(
                [], [],
                color='k', marker='s',
                alpha=0.5, label='Histogram Density'
            )  # empty object for points
        # objects for ellipses
        ellipse_sizes = [2, *(1,)*len(means)]
        ellipse_styles = ['-', *('--')*len(means)]
        for c, (size, color, style) in enumerate(
            zip(
                ellipse_sizes,
                colors_ellipses[:len(means)],
                ellipse_styles
            )
        ):
            if c == 0:
                label = 'Reference Gaussian'
            else:
                label = 'Pathological Gaussian '+str(c)
            plt.plot([], [], color=color, linestyle=style, label=label)
        plt.legend()
    plt.xlim(scope)
    plt.ylim(scope)



