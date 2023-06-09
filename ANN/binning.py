import numpy as np

def n_dimensional_binning(data, bin_edges):
    """
    Bins data in n dimensions using the specified bin edges.

    Parameters:
        data (ndarray): An n-dimensional array of data points to bin.
        bin_edges (list of ndarrays): A list of ndarrays, where each ndarray
            contains the bin edges for a single dimension.

    Returns:
        ndarray: An n-dimensional array containing the counts for each bin.
    """
    # Get the number of dimensions and the shape of the bin counts array
    n = data.shape[1]
    counts_shape = tuple(len(edges) for edges in bin_edges)

    # Initialize the n-dimensional bin counts array
    counts = np.zeros(counts_shape)

    # Loop over each data point and increment the counts for the corresponding bin
    for point in data:
        bin_indices = [np.digitize(point[i], bin_edges[i]) - 1 for i in range(n)]
        counts[tuple(bin_indices)] += 1

    return counts
