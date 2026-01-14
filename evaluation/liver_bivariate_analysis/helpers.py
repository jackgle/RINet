import pickle
import numpy as np
from scipy.stats import skew, multivariate_normal, boxcox  # only for testing numerical BC estimate
from shapely.geometry import Point, Polygon
from scipy.stats import chi2


def save_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def remove_outliers(data):
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
    is_outlier = np.zeros(data.shape[0], dtype=bool)
    for i in range(data.shape[1]):
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


def get_data(df, analytes, gender, log_analytes, outlier_removal=False, transform=False):
    """ Preprocess data for a pair of analytes and get RI region
    
    Args:
        df: dataframe with analyte data as formatted in ./data/format_datasets.ipynb
        pair: the pair of analytes to retrieve preprocessed data for
        gender: gender to filter results for
        log_analytes: list of analytes for which normalization will be applied
    Returns:
        data: preprocessed data
        labels: labels associated with data (0=healthy, 1=pathological)
        lambdas: BoxCox lambdas determined for any data transforms
    """
    # filter gender and drop NA
    dfp = df[df.gender == gender].dropna(subset=analytes)
    data = dfp[analytes].values
    labels = dfp.label.values
    
    # remove outliers
    if outlier_removal:
        outliers = remove_outliers(data)
        data = data[~outliers]
        labels = labels[~outliers]
    
    # transform certain analytes
    if transform:
        for c, i in enumerate(analytes):
            if i in log_analytes:
                data[:, c] = np.log(data[:, c])

    return data, labels


# def get_ellipse_vertices(ellipse, num_points=2000):
#     """ Gets the polygon vertices for the ellipse defined by predicted mean and covariance
#     """
#     t = np.linspace(0, 2*np.pi, num_points)
#     vertices = np.vstack([np.cos(t), np.sin(t)]).T
#     vertices = ellipse.get_patch_transform().transform(vertices)
#     return vertices


def get_ellipse_vertices(mean, cov, percentage=0.95, num_points=2000):
    """
    Generate vertices for the covariance ellipse.
    """
    # Confidence scaling
    nstd = np.sqrt(chi2.ppf(percentage, 2))

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Parametric angles
    t = np.linspace(0, 2*np.pi, num_points)
    circle = np.array([np.cos(t), np.sin(t)])  # shape (2, num_points)

    # Scale by sqrt of eigenvalues (axis lengths)
    ellipse = nstd * eigvecs @ np.diag(np.sqrt(eigvals)) @ circle

    # Translate to mean
    ellipse = ellipse.T + mean

    return ellipse


def iou(region1, region2):
    """ Intersection over union of two polygonal regions
        Very common object detection performance metric
    Args:
        region1: region array containing (vertices, dimensions)
        region2: region array containing (vertices, dimensions)
    """
    # define two polygons as lists of (x, y) coordinates
    poly1 = Polygon(region1)
    poly2 = Polygon(region2)

    # compute the intersection and union areas
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    iou = intersection_area / union_area
    return iou


def predict_labels(data_points, verts):
    """ Given data points and a polygonal decision boundary, predicts points as positive (outside) or negative (inside) 
    """
    poly = Polygon(verts)
    points = [Point(p) for p in data_points]
    p = []
    for point in points:
        p.append(poly.contains(point))
    p = np.array(p)
    return ~p


def estimate_bc_numerical(data, mean1, cov1, mean2, cov2):
    """ Previously used for numerical estimation of bhattacharyya coeffecient to double check
    """
    # Generate a grid of points
    grid_size = 1000
    x = np.linspace(data[:, 0].min(), data[:, 0].max(), grid_size)
    y = np.linspace(data[:, 1].min(), data[:, 1].max(), grid_size)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Evaluate densities
    g1 = multivariate_normal(mean1, cov1)
    g2 = multivariate_normal(mean2, cov2)
    d1 = g1.pdf(positions.T)
    d2 = g2.pdf(positions.T)

    # Compute Bhattacharyya coefficient
    bc = np.sum(np.sqrt(d1 * d2)) * (x[1] - x[0]) * (y[1] - y[0])
    return bc

    
def get_ri_vertices(ris, analyte_pair, gender=None):
    """ Gets the polygon vertices for the box defined by a pair of RIs
    
    Args:
        ris: dictionary with format:
            {
                analyte_name: {'unit': unit, 'ri': [lower, upper]} # for no gender difference
                analyte_name: {'unit': unit, 'ri': {'F': [lower, upper], 'M': [lower upper]}} # for gender differences
                ...
            }
        analyte_pair: list of 2 analytes
        gender: gender to get RIs for, gender-specific RIs will be averaged in the case of None
    Returns:
        verts: vertices of rectangular region defined by RIs for each analyte
    """
    ri_pair = np.zeros((2, 2))
    for c, i in enumerate(analyte_pair):
        if isinstance(ris[i]['ri'], dict):  # if RIs are gender-specific
            if gender is None:  # if no gender specified, take average
                ri_pair[0, c] = np.mean([ris[i]['ri']['M'][0], ris[i]['ri']['F'][0]])
                ri_pair[1, c] = np.mean([ris[i]['ri']['M'][1], ris[i]['ri']['F'][1]])
            else:
                ri_pair[0, c] = ris[i]['ri'][gender][0]
                ri_pair[1, c] = ris[i]['ri'][gender][1] 
        else:
            ri_pair[0, c] = ris[i]['ri'][0]
            ri_pair[1, c] = ris[i]['ri'][1] 
    verts = np.array([
        [ri_pair[0, 0], ri_pair[0, 1]],  # low x, low y
        [ri_pair[1, 0], ri_pair[0, 1]],  # high x, low y
        [ri_pair[1, 0], ri_pair[1, 1]],  # high x, high y
        [ri_pair[0, 0], ri_pair[1, 1]],  # low x, high y
    ])
    return verts
    


