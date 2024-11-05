import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
import time


def estimate_lambda(X, k=8):
    """
    Estimate lambda parameter for clustering based on k-nearest neighbors.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Input data
    k : int, default=8
        Number of nearest neighbors to use

    Returns:
    --------
    float
        Estimated lambda value
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    avg_distances = np.mean(distances[:, 1:], axis=1)  # 排除自身距离
    lambda_value = np.median(avg_distances)
    return lambda_value


def exp_local_scaling_transform(distances, k=7):
    """
    Apply exponential local scaling transform to distances.

    Parameters:
    -----------
    distances : array-like
        Distance matrix
    k : int, default=7
        Number of neighbors for local scaling

    Returns:
    --------
    array-like
        Transformed distances
    """
    sigma = np.sort(distances, axis=1)[:, k]
    sigma = sigma.reshape(-1, 1)
    transformed = np.exp(-distances ** 2 / (sigma * sigma.T))
    return transformed


def rknn_with_distance_transform(data, k, transform_func):
    """
    Compute reverse k-nearest neighbors with distance transformation.

    Parameters:
    -----------
    data : array-like
        Input data
    k : int
        Number of neighbors
    transform_func : callable
        Function to transform distances

    Returns:
    --------
    array-like
        Border values for each point
    """
    nbrs = NearestNeighbors(n_neighbors=data.shape[0]).fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Transform distances
    transformed_distances = transform_func(distances, k)

    # Calculate border values
    border_values = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        knn_indices = indices[i, 1:k + 1]
        border_values[i] = np.sum([1 for j in knn_indices if i in indices[j, 1:k + 1]])

    return border_values / k


def dynamic_3w(data, border_func, threshold_func, max_iterations=150, mean_border_eps=-1,
               plot_debug_output_dir=None, k=20, precentile=0.1, dist_threshold=3,
               link_dist_expansion_factor=3, verbose=True, vis_data=None,
               min_cluster_size=3, stopping_precentile=0, should_merge_core_points=True,
               debug_marker_size=70, core_points_threshold=1, dvalue_threshold=1):
    """
    Perform dynamic 3W clustering.

    Parameters:
    -----------
    data : array-like
        Input data
    border_func : callable
        Function to compute border values
    threshold_func : callable
        Function to determine thresholds
    max_iterations : int, default=150
        Maximum number of iterations
    mean_border_eps : float, default=-1
        Epsilon value for mean border
    plot_debug_output_dir : str, optional
        Directory for debug output plots
    k : int, default=20
        Number of neighbors
    precentile : float, default=0.1
        Percentile for border values
    dist_threshold : float, default=3
        Distance threshold
    link_dist_expansion_factor : float, default=3
        Factor for link distance expansion
    verbose : bool, default=True
        Whether to print progress
    vis_data : array-like, optional
        Data for visualization
    min_cluster_size : int, default=3
        Minimum cluster size
    stopping_precentile : float, default=0
        Percentile for stopping condition
    should_merge_core_points : bool, default=True
        Whether to merge core points
    debug_marker_size : int, default=70
        Size of markers in debug plots
    core_points_threshold : float, default=1
        Threshold for core points
    dvalue_threshold : float, default=1
        Threshold for d-values

    Returns:
    --------
    tuple
        Clustering results including labels, core points, etc.
    """
    if vis_data is None:
        vis_data = data

    data_sets_by_iterations = []
    border_values_per_iteration = []
    current_data = data.copy()
    current_vis_data = vis_data.copy()

    # Track indices through iterations
    current_indices = np.arange(len(data))

    for iteration in range(max_iterations):
        if len(current_data) < min_cluster_size:
            break

        # Calculate border values
        border_values = border_func(current_data)
        border_values_per_iteration.append(border_values)

        # Determine threshold
        if threshold_func is None:
            threshold = np.percentile(border_values, precentile * 100)
        else:
            threshold = threshold_func(border_values)

        # Select points based on threshold
        selected_points = border_values > threshold

        if np.sum(selected_points) < min_cluster_size:
            break

        # Update data sets
        current_data = current_data[selected_points]
        current_vis_data = current_vis_data[selected_points]
        current_indices = current_indices[selected_points]

        data_sets_by_iterations.append((current_indices, current_data, current_vis_data))

        if verbose:
            print(f"Iteration {iteration}: {len(current_data)} points remaining")

        # Plot debug output if requested
        if plot_debug_output_dir:
            plt.figure(figsize=(10, 10))
            plt.scatter(current_vis_data[:, 0], current_vis_data[:, 1],
                        c=border_values, cmap='viridis', s=debug_marker_size)
            plt.colorbar()
            plt.title(f'Iteration {iteration}')
            plt.savefig(os.path.join(plot_debug_output_dir, f'iteration_{iteration}.png'))
            plt.close()

    # Process results
    core_points = data_sets_by_iterations[-1][1]
    core_points_indices = data_sets_by_iterations[-1][0]

    # Initialize labels
    labels = np.full(len(data), -1)

    # Assign cluster labels
    if should_merge_core_points:
        # Merge close core points
        core_clusters = DBSCAN(eps=dist_threshold, min_samples=1).fit(core_points)
        core_labels = core_clusters.labels_

        # Update labels for core points
        for i, idx in enumerate(core_points_indices):
            labels[idx] = core_labels[i]
    else:
        # Each core point forms its own cluster
        for i, idx in enumerate(core_points_indices):
            labels[idx] = i

    return (labels, core_points, None, data_sets_by_iterations,
            None, None, border_values_per_iteration, core_points_indices)


def border_peel_rknn_exp_transform_local(data, k, threshold, iterations, debug_output_dir=None,
                                         dist_threshold=3, link_dist_expansion_factor=3, precentile=0, verbose=True):
    """
    Perform border peeling with reverse k-nearest neighbors and exponential transform.

    Parameters:
    -----------
    data : array-like
        Input data
    k : int
        Number of neighbors
    threshold : float
        Threshold value
    iterations : int
        Number of iterations
    debug_output_dir : str, optional
        Directory for debug output
    dist_threshold : float, default=3
        Distance threshold
    link_dist_expansion_factor : float, default=3
        Factor for link distance expansion
    precentile : float, default=0
        Percentile value
    verbose : bool, default=True
        Whether to print progress

    Returns:
    --------
    tuple
        Results of dynamic_3w clustering
    """
    border_func = lambda data: rknn_with_distance_transform(data, k, exp_local_scaling_transform)
    threshold_func = lambda value: value > threshold
    return dynamic_3w(data, border_func, threshold_func,
                      plot_debug_output_dir=debug_output_dir, k=k, precentile=precentile,
                      dist_threshold=dist_threshold, link_dist_expansion_factor=link_dist_expansion_factor,
                      verbose=verbose)