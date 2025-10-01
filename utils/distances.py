import torch
import numpy as np
import plotly.graph_objects as go
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from jaxtyping import Float
from torch import Tensor
from ete3 import Tree
from multiprocessing import Pool

def build_knn_graph(points: Float[Tensor, "n d"], k: int, distance: str = 'euclidean', weighted: bool = True) -> csr_matrix:
    """
    Build KNN graph from points.
    
    Args:
        points: torch tensor of shape (n, d)
        k: number of nearest neighbours
        distance: 'euclidean' or 'cosine'
        weighted: if True, edges weighted by distance; if False, all weights = 1
    
    Returns:
        sparse adjacency matrix (scipy csr_matrix)
    """
    n = points.shape[0]
    
    # Calculate distances
    if distance == 'euclidean':
        distances = torch.cdist(points, points, p=2)
    elif distance == 'cosine':
        # Cosine distance = 1 - cosine similarity
        points_norm = points / points.norm(dim=1, keepdim=True)
        cosine_sim = points_norm @ points_norm.T
        distances = torch.arccos(cosine_sim)  # convert similarity to distance
    else:
        raise ValueError(f"Unknown distance: {distance}")
    
    # Get k+1 nearest (includes self at distance 0)
    _, indices = torch.topk(distances, k + 1, largest=False, dim=1)
    
    # Build sparse adjacency matrix
    row_indices = []
    col_indices = []
    weights = []
    
    for i in range(n):
        for j in indices[i, 1:]:  # skip self
            j = j.item()
            weight = 1.0 if not weighted else distances[i, j].item()
            row_indices.append(i)
            col_indices.append(j)
            weights.append(weight)
    
    adjacency_matrix = csr_matrix(
        (weights, (row_indices, col_indices)),
        shape=(n, n)
    )
    
    return adjacency_matrix

def geodesic_distance(adjacency_matrix: csr_matrix, start: int, end: int) -> tuple[float, list[int]]:
    """
    Find shortest path between two nodes (indices start and end) using Dijkstra's algorithm.
    
    Returns:
        distance: geodesic distance
        path: list of node indices from start to end
    """
    # Run Dijkstra from start node
    distances, predecessors = dijkstra(
        adjacency_matrix, 
        indices=start, 
        return_predecessors=True
    )
    
    # Check if path exists
    if np.isinf(distances[end]):
        return float('inf'), []
    
    # Reconstruct path
    path = []
    current = end
    while current != -9999:  # scipy uses -9999 for no predecessor
        path.append(current)
        current = predecessors[current]
    path.reverse()
    
    return distances[end], path

def geodesic_distance_matrix(adjacency_matrix: csr_matrix) -> np.ndarray:
    """
    Compute geodesic distances between all pairs of points.
    
    Args:
        adjacency_matrix: sparse adjacency matrix from build_knn_graph
    
    Returns:
        distance_matrix: (n, n) array of geodesic distances
    """
    distance_matrix = dijkstra(adjacency_matrix, return_predecessors=False)
    return distance_matrix

def cosine_similarity_matrix(X: Float[Tensor, "n d"]) -> Float[Tensor, "n n"]:
    """
    Compute pairwise cosine similarity between all vectors.
    
    Args:
        X: tensor of shape (n, d)
    
    Returns:
        similarity matrix of shape (n, n)
    """
    X_norm = X / X.norm(dim=1, keepdim=True)
    return X_norm @ X_norm.T

def phylogenetic_distance_matrix(accession_ids: list[str], tree: Tree) -> torch.Tensor:
    """
    Compute pairwise phylogenetic distances between sequences given a series of accession IDs and a phylogenetic tree.
    """
    n = len(accession_ids)
    distance_matrix = torch.zeros((n, n), dtype=torch.float32)

    for i in range(n):
        for j in range(i + 1, n):
            dist = tree.get_distance(accession_ids[i], accession_ids[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def _compute_distance_chunk(args):
    """Worker function to compute distances for a chunk of row indices."""
    row_indices, accession_ids, tree_path = args
    
    # Each worker loads its own tree
    tree = Tree(tree_path, format=1, quoted_node_names=True)
    leaves = tree.get_leaf_names()
    
    results = []
    for i in row_indices:
        for j in range(i + 1, len(accession_ids)):
            if accession_ids[i] in leaves and accession_ids[j] in leaves:
                dist = tree.get_distance(accession_ids[i], accession_ids[j])
            else:
                dist = float('nan')
            results.append((i, j, dist))
    
    return results

def mp_phylogenetic_distance_matrix(accession_ids: list[str], tree_path: str, n_processes: int = 12) -> torch.Tensor:
    """
    Compute pairwise phylogenetic distances using multiprocessing.
    
    Args:
        accession_ids: List of accession IDs
        tree_path: Path to the phylogenetic tree file
        n_processes: Number of worker processes
    """
    n = len(accession_ids)
    distance_matrix = torch.zeros((n, n), dtype=torch.float32)
    
    # Divide rows among workers
    chunk_size = n // n_processes
    chunks = []
    for proc in range(n_processes):
        start = proc * chunk_size
        end = n if proc == n_processes - 1 else (proc + 1) * chunk_size
        row_indices = list(range(start, end))
        chunks.append((row_indices, accession_ids, tree_path))
    
    # Process in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(_compute_distance_chunk, chunks)
    
    # Fill the matrix
    for chunk_results in results:
        for i, j, dist in chunk_results:
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix




if __name__ == "__main__":

    # Example: noisy sphere
    n_points = 1000
    R = 1.0

    # Sample points uniformly on sphere
    u = torch.rand(n_points) * 2 * np.pi
    v = torch.arccos(2 * torch.rand(n_points) - 1)  # uniform on sphere

    x = R * torch.sin(v) * torch.cos(u)
    y = R * torch.sin(v) * torch.sin(u)
    z = R * torch.cos(v)

    # Add radial noise
    noise_scale = 0.05
    radial_noise = 1 + noise_scale * torch.randn(n_points)
    x = x * radial_noise
    y = y * radial_noise
    z = z * radial_noise

    points = torch.stack([x, y, z], dim=1)

    # Build KNN graphs
    k = 15
    graph_euclidean = build_knn_graph(points, k, distance='cosine', weighted=False)

    # Calculate geodesics between antipodal points
    start_idx = 0
    end_idx = torch.argmin(torch.sum(points * points[start_idx], dim=1)).item()

    dist_euc, path_euc = geodesic_distance(graph_euclidean, start_idx, end_idx)

    print(f"Euclidean geodesic distance: {dist_euc:.3f}")
    print(f"Path length: {len(path_euc)} points")

    # Visualise
    points_np = points.numpy()
    path_points_euc = points_np[path_euc]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=points_np[:, 0], y=points_np[:, 1], z=points_np[:, 2],
        mode='markers', marker=dict(size=2, color='lightblue', opacity=0.5),
        name='All points'
    ))

    fig.add_trace(go.Scatter3d(
        x=path_points_euc[:, 0], y=path_points_euc[:, 1], z=path_points_euc[:, 2],
        mode='lines+markers', line=dict(color='red', width=4),
        marker=dict(size=3), name='Geodesic'
    ))

    fig.add_trace(go.Scatter3d(
        x=[points_np[start_idx, 0]], y=[points_np[start_idx, 1]], z=[points_np[start_idx, 2]],
        mode='markers', marker=dict(size=10, color='green'), name='Start'
    ))

    fig.add_trace(go.Scatter3d(
        x=[points_np[end_idx, 0]], y=[points_np[end_idx, 1]], z=[points_np[end_idx, 2]],
        mode='markers', marker=dict(size=10, color='orange'), name='End'
    ))

    fig.update_layout(
        scene=dict(aspectmode='data'),
        title=f'Geodesic on Noisy Sphere (k={k})<br>Distance: {dist_euc:.2f}'
    )
    fig.show()
    # %%
