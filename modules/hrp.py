import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def correlation_to_distance(
    correlation: np.ndarray
) -> np.ndarray:
    """
    Convert correlation matrix into distance matrix.

    Higher correlation → smaller distance.
    Lower correlation → larger distance.
    """

    correlation = np.clip(
        correlation,
        -1.0,
        1.0
    )

    correlation = correlation.copy()
    np.fill_diagonal(correlation, 1.0)

    distance = np.sqrt(
        2.0 * (1.0 - correlation)
    )

    np.fill_diagonal(distance, 0.0)

    return distance


def run_hierarchical_clustering(
    distance: np.ndarray
):
    """
    Perform hierarchical clustering using Ward linkage.
    """

    condensed_distance = squareform(
        distance,
        checks=False
    )

    return linkage(
        condensed_distance,
        method="ward"
    )


def get_leaf_ordering(
    linkage_matrix: np.ndarray
) -> list:
    """
    Get the ordering of assets produced by
    hierarchical clustering.
    """

    dendrogram_data = dendrogram(
        linkage_matrix,
        no_plot=True
    )

    return dendrogram_data["leaves"]


def cluster_variance(covariance, indices):
    """
    Calculate the variance of a cluster using
    inverse-variance portfolio weights.

    The inverse-variance weights determine how much
    each asset contributes to the cluster.

    The FULL covariance matrix is then used to calculate
    the actual cluster variance, including correlations
    between assets.
    """

    sub_cov = covariance[np.ix_(indices, indices)]

    # Individual asset variances
    variances = np.diag(sub_cov)

    # Protect against zero / invalid variance
    variances = np.maximum(variances, 1e-12)

    # Inverse-variance weights
    inv_variance = 1.0 / variances
    weights = inv_variance / np.sum(inv_variance)

    # IMPORTANT:
    # Use the complete covariance matrix here.
    # This includes both variance and covariance between assets.
    cluster_var = weights @ sub_cov @ weights

    return float(cluster_var)

def calculate_hrp_weights(
    covariance: np.ndarray,
    ordering: list
) -> np.ndarray:
    """
    Calculate Hierarchical Risk Parity weights.
    """

    n_assets = covariance.shape[0]

    weights = np.ones(n_assets)

    clusters = [list(ordering)]

    while clusters:

        new_clusters = []

        for cluster in clusters:

            if len(cluster) <= 1:
                continue

            midpoint = len(cluster) // 2

            left = cluster[:midpoint]
            right = cluster[midpoint:]

            variance_left = cluster_variance(
                covariance,
                left
            )

            variance_right = cluster_variance(
                covariance,
                right
            )

            # Allocate more weight to
            # the lower-risk cluster.
            alpha = variance_right / max(
                variance_left + variance_right,
                1e-10
            )

            for index in left:
                weights[index] *= alpha

            for index in right:
                weights[index] *= (1.0 - alpha)

            if len(left) > 1:
                new_clusters.append(left)

            if len(right) > 1:
                new_clusters.append(right)

        clusters = new_clusters

    # Normalize to exactly 100%
    weights /= weights.sum()

    return weights


def build_covariance_matrix(
    correlation: np.ndarray,
    garch_results: dict,
    sectors: list
) -> np.ndarray:
    """
    Construct today's covariance matrix from
    GARCH volatility and DCC correlation.
    """

    volatility = np.zeros(len(sectors))

    for i, sector in enumerate(sectors):

        volatility[i] = np.sqrt(
            max(
                garch_results[sector]["sigma2"][-1],
                1e-10
            )
        )

    volatility_matrix = np.diag(volatility)

    covariance = (
        volatility_matrix
        @ correlation
        @ volatility_matrix
    )

    covariance = (
        covariance + covariance.T
    ) / 2.0

    covariance += (
        1e-8 * np.eye(len(sectors))
    )

    return covariance