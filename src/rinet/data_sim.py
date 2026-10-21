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


def log_uniform(low, high, size=None):
    """Samples uniformly on a log scale between low and high."""
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))


def quantize_original_scale(data, step, log_sigma_range):
    """
    Quantize log-scale data as laboratory values are rounded: on the original scale.

    The sample is treated as log-scale data whose reference component has unit
    SD. It is mapped to the original scale with a random log-scale SD sigma
    (so the reference is log-normal there), rounded to a multiple of
    step * (original-scale reference SD), and mapped back. The lowest
    recordable value is one step, as zeros cannot be log-transformed. Because
    rounding is uniform on the original scale, it is coarser at low values after
    the log transform, as in real data.

    Parameters:
    data (numpy.ndarray): Samples (n, d) in which the reference component is N(0, 1).
    step (float): Bin size in SDs of the reference component on the original scale.
    log_sigma_range (list): Min/max log-scale SD of the reference (log-uniform, per dimension).

    Returns:
    numpy.ndarray: Quantized data, with the reference component again approximately N(0, 1).
    """
    sigma = log_uniform(log_sigma_range[0], log_sigma_range[1], data.shape[1])
    x = np.exp(sigma * data)
    reference_sd = np.sqrt((np.exp(sigma**2) - 1) * np.exp(sigma**2))
    width = step * reference_sd
    x = np.maximum(np.round(x / width) * width, width)
    return np.log(x) / sigma


def random_mixture_sample(
    n_dimensions=1,             # number of dimensions
    n_components=2,             # number of pathological components
    p_frac_range=[0, 0.40],     # min/max pathological fraction
    bg_frac_range=[0, 0.1],     # min/max background noise fraction
    clean_prob=0.05,             # probability of a clean sample: no pathological components or background noise
    bg_scale_range=[5, 10],     # min/max stds from mixture mean for background noise
    max_corr=0.8,               # max variable correlation
    shift_range=[1, 8],         # min/max |pathological mean - reference mean| per dimension, in reference SDs (log-uniform)
    std_ratio_range=[0.4, 2.5], # min/max pathological SD / reference SD per dimension (log-uniform)
    n_samples=10000,            # no. samples
    quantize_step=0,            # bin size for quantizing on the original scale, in reference SDs
    log_sigma_range=[0.05, 1.0],  # min/max log-scale SD of the reference, used to place the data on the original scale for quantizing
):
    """
    Generates a sample from a Gaussian mixture: a standard normal reference
    component plus pathological components drawn relative to it.

    The reference is fixed at N(0, I) because standardized inputs and targets
    are invariant to its location and scale, so drawing pathological shifts and
    widths relative to it gives even coverage of what the model actually sees.

    Args:
        See above
    Returns:
        mixture_sample:   points sampled from the mixture
        mean_vectors:     list of mean vectors of each Gaussian, beginning with reference
        std_vectors:      list of std vectors of each Gaussian, beginning with reference
        corr_matrices:    list of correlation matrices of each Gaussian (>=2D only)
        components:       component label for each point (-1 = background noise)
        component_sizes:  size of each Gaussian component, beginning with reference,
                          followed by the background noise size
    """

    assert n_dimensions <= 2, (
        "Warning: 3-D data generation is in progress "
        "and the current method may lead to unrealistically small determinants"
    )

    clean = np.random.uniform() < clean_prob
    p_frac = 0 if clean or n_components == 0 else np.random.uniform(p_frac_range[0], p_frac_range[1])
    bg_frac = 0 if clean else np.random.uniform(bg_frac_range[0], bg_frac_range[1])
    ref_size = int(n_samples*(1-p_frac-bg_frac))

    # reference component
    mean_vector = np.zeros(n_dimensions)
    std_dev = np.ones(n_dimensions)
    if n_dimensions > 1:
        corr_mat = random_correlation_matrix(n_dimensions, max_corr)
        mixture_sample = np.random.multivariate_normal(mean_vector, corr_mat, ref_size)
    else:
        corr_mat = None
        mixture_sample = np.random.normal(0, 1, (ref_size, n_dimensions))

    components = [np.zeros(ref_size)]
    corr_matrices = [corr_mat]
    mean_vectors = [mean_vector]
    std_vectors = [std_dev]

    # pathological components
    p_ratios = np.random.uniform(0, 1, n_components)
    p_ratios /= np.sum(p_ratios)
    for i in range(len(p_ratios)):

        p_size = int(n_samples*p_frac*p_ratios[i])  # get pathological component size

        mv = log_uniform(shift_range[0], shift_range[1], n_dimensions) * \
            np.random.choice([-1, 1], n_dimensions)
        std = log_uniform(std_ratio_range[0], std_ratio_range[1], n_dimensions)

        if n_dimensions > 1:
            cvm = correlation_to_covariance(corr_mat, std)
            gauss_sample = np.random.multivariate_normal(mv, cvm, p_size)
        else:
            gauss_sample = np.random.normal(mv, std, (p_size, n_dimensions))

        corr_matrices.append(corr_mat)  # store parameters
        mean_vectors.append(mv)
        std_vectors.append(std)

        mixture_sample = np.vstack([mixture_sample, gauss_sample])
        components.append(np.ones(len(gauss_sample))*(i+1))

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
        mixture_sample = quantize_original_scale(mixture_sample, quantize_step, log_sigma_range)

    components.append(np.ones(bg_noise.shape[0])*-1)
    components = np.concatenate(components)

    # count by component index so sizes stay aligned with mean_vectors even
    # when a small pathological component rounds down to zero points
    component_sizes = [int(np.sum(components == i)) for i in range(n_components + 1)]
    component_sizes.append(int(np.sum(components == -1)))  # background noise size

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



