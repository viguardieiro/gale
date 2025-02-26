import itertools
import multiprocessing
import networkx as nx
import numpy as np

import gudhi as gd
from gudhi.cover_complex import MapperComplex

from sklearn.cluster import AgglomerativeClustering

def create_mapper(
    X: np.ndarray,
    f: np.ndarray,
    resolution: int,
    gain: float,
    dist_thresh=None,
    estimate_scale_N=100,
    estimate_scale_beta=0.001,
    clusterer=AgglomerativeClustering(n_clusters=None, linkage="single"),
    min_points_per_node=0,
    colors=None,
) -> MapperComplex:
    """Runs Mapper on given some data, a filter function, and resolution + gain parameters.

    Args:
        X (np.ndarray): Array of data. For GALE, this is the feature attribution output (n x k), where there are n samples with k feature attributions each.
        f (np.ndarray): Filter (lens) function. For GALE, the predicted probabilities are the lens function.
        resolution (int): Resolution (how wide each window is)
        gain (float): Gain (how much overlap between windows)
        dist_thresh (float): If using AgglomerativeClustering, this sets the distance threshold. Ignored if clusterer is not AgglomerativeClustering. If dist_thresh is None, then the distance threshold is estimated using the estimate_scale_N and estimate_scale_beta parameters.
        estimate_scale_N (int): Number of runs to estimate the distance threshold. Ignored if dist_thresh is not None. See MapperComplex.estimate_scale for more details.
        estimate_scale_beta (float): Beta parameter for estimating the distance threshold. Ignored if dist_thresh is not None. See MapperComplex.estimate_scale for more details.
        clusterer (sklearn.base.ClusterMixin, optional): Clustering method from sklearn. Defaults to AgglomerativeClustering(n_clusters=None, linkage="single").
        min_points_per_node (int, optional): Minimum number of points per node. Defaults to 0.
        colors (np.ndarray, optional): Function values to color mapper nodes. Defaults to f.

    Returns:
        MapperComplex: MapperComplex object
    """
    if dist_thresh is None:
        mapper = MapperComplex(input_type="point cloud")
        dist_thresh = mapper.estimate_scale(X, N=estimate_scale_N, beta=estimate_scale_beta)

    clusterer.distance_threshold = dist_thresh
    mapper = MapperComplex(input_type="point cloud", min_points_per_node=min_points_per_node,
                           clustering=clusterer,
                           resolutions=[resolution], gains=[gain])
    if colors is None:
        colors = f

    mapper.fit(X, filters=f, colors=colors)
    return mapper


def create_pd(mapper: MapperComplex, return_d=False) -> list:
    """Creates a persistence diagram from Mapper output.

    Args:
        mapper (MapperComplex): Mapper output from `create_mapper`

    Returns:
        list: List of the topographical features
    """
    st = mapper.simplex_tree_.copy()
    G = mapper.get_networkx(set_attributes_from_colors=True)
    filtration = nx.get_node_attributes(G, "attr_name")
    for k in filtration.keys():
        st.assign_filtration([k], filtration[k][0])
    st.make_filtration_non_decreasing()
    st.extend_filtration()
    dgms = st.extended_persistence()
    pdgms = []
    for dgm in dgms:
        pdgms += [d[1] for d in dgm]
    if return_d:
        return dgms, pdgms
    else:
        return pdgms


def bottleneck_distance(mapper_a: MapperComplex, mapper_b: MapperComplex) -> float:
    """Calculates the bottleneck distance between two Mapper outputs (denoted A and B)

    Args:
        mapper_a (MapperComplex): Mapper A, from `create_mapper`
        mapper_b (MapperComplex): Mapper B, from `create_mapper`

    Returns:
        float: the bottleneck distance
    """
    pd_a = create_pd(mapper_a)
    pd_b = create_pd(mapper_b)
    return gd.bottleneck_distance(pd_a, pd_b)

# Sub function to run the bootstrap sequence
def _bootstrap_sub(params):
    # Fix random seed to use the same bootstrap samples for all parameters
    if params[9] is not None:
        np.random.seed(params[9])

    M = create_mapper(X=params[0],
        f=params[1],
        resolution=params[2],
        gain=params[3],
        dist_thresh=params[4],
        min_points_per_node=params[5],
        clusterer=params[6]
    )
    n_samples = params[0].shape[0]
    distribution, cc = [], []
    for bootstrap in range(params[8]):
        # Randomly select points with replacement
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        Xboot = params[0][idxs, :]
        fboot = params[1][idxs]
        # Fit mapper
        M_boot = create_mapper(Xboot, fboot, 
                                resolution=params[2],
                                gain=params[3],
                                dist_thresh=params[4],
                                min_points_per_node=params[5],
                                clusterer=params[6]
                                )
        distribution.append(bottleneck_distance(M_boot, M))
        G_boot = M_boot.get_networkx(set_attributes_from_colors=True)
        G_cc = nx.number_connected_components(G_boot)
        cc.append(G_cc)
        
    distribution = np.sort(distribution)
    distribution_thresh = distribution[int(params[7] * len(distribution))]
    cc = np.sort(cc)
    cc_thresh = cc[int(params[7] * len(cc))]
    return params[2], params[3], params[4], distribution_thresh, cc_thresh


def bootstrap_mapper_params(
    X: np.ndarray,
    f: np.ndarray,
    resolutions: list,
    gains: list,
    distances: list,
    clusterer=AgglomerativeClustering(n_clusters=None, linkage="single"),
    min_points_per_node=[0],
    ci=0.95,
    n=30,
    n_jobs=1,
    seed=None,
    selection_type='min_stability'
) -> MapperComplex:
    """Bootstraps the data to figure out the best Mapper parameters through a greedy search.

    Args:
        X (np.ndarray): Array of data. For GALE, this is the feature attribution output (n x k), where there are n samples with k feature attributions each.
        f (np.ndarray): Filter (lens) function. For GALE, the predicted probabilities are the lens function.
        resolutions (list): List of resolutions to test.
        gains (list): List of gains to test.
        distances (list): If using AgglomerativeClustering, this sets the distance threshold as (X.max() - X.min())*thresh.
        clusterer (sklearn.base.ClusterMixin, optional): Clustering method from sklearn. Defaults to AgglomerativeClustering(n_clusters=None, linkage="single").
        min_points_per_node (list): List of min points per node to test.
        ci (float, optional): Confidence interval to create. Defaults to 0.95.
        n (int, optional): Number of bootstraps to run. Defaults to 30.
        n_jobs (int, optional): Number of processes for multiprocessing. Defaults to CPU count. -1 for all cores.
        seed (int, optional): Random seed for boostrap samples.
        selection_type: Method to pick the best params. Options are 'min_stability' (default) or 'min_distance'.

    Returns:
        MapperComplex: Dictionary containing the Mapper parameters found in a greedy search
    """
    # Create parameter list
    paramlist = list(
        itertools.product(
            [X], [f], resolutions, gains, distances, min_points_per_node, [clusterer], [ci], [n], [seed]
        )
    )

    # Create MP pool
    if n_jobs < 1:
        pool = multiprocessing.Pool()
    else:
        pool = multiprocessing.Pool(processes=n_jobs)

    results = pool.map(_bootstrap_sub, paramlist)

    # Find "best" parameters by scaling stability and components between [0,1]
    # then, calculate the distance to (0,0). Select the params with minimum distance.
    best_stability = None
    best_component = None
    best_r = None
    best_g = None
    best_d = None

    # Put stabilities, components, etc. into lists
    stability, component, resolution, gain, distance = [], [], [], [], []
    for res in results:
        stability.append(res[3])
        component.append(res[4])
        resolution.append(res[0])
        gain.append(res[1])
        distance.append(res[2])

    if selection_type=='min_stability':
        best_stability = min(stability)
        index = stability.index(best_stability)
        best_component = component[index]
        best_r = resolution[index]
        best_g = gain[index]
        best_d = distance[index]

    elif selection_type=='min_distance':
        # Find min/max for stability and components, for scaling purposes
        min_distance = 999
        min_stability = min(stability)
        max_stability = max(stability)
        min_component = min(component)
        max_component = max(component)

        # Calculate distance to (0,0) and take the smallest
        for s, c, r, g, d in zip(stability, component, resolution, gain, distance):
            if max_stability == min_stability:
                stab = 0
            else:
                stab = (s - min_stability) / (max_stability - min_stability)
            if max_component == min_component:
                comp = 0
            else:
                comp = (c - min_component) / (max_component - min_component)
            dist = np.sqrt(stab**2 + comp**2)
            if dist < min_distance:
                best_stability = s
                best_component = c
                best_r = r
                best_g = g
                best_d = d

    return {
        "stability": best_stability,
        "components": best_component,
        "resolution": best_r,
        "gain": best_g,
        "distance_threshold": best_d,
    }


def mapper_to_networkx(mapper: MapperComplex) -> nx.classes.graph.Graph:
    """Takes the Mapper output and transforms it to a networkx graph.

    Args:
        mapper (MapperComplex): Mapper output from `create_mapper`

    Returns:
        nx.classes.graph.Graph: Networkx graph produced by the Mapper output.
    """
    G = mapper.get_networkx(set_attributes_from_colors=True)

    return G
