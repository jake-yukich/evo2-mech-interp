import numpy as np
import torch
import umap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def save_plotly_figure(fig: go.Figure, filepath: Path, formats: list[str] = ["html", "png"]):
    """
    Save a Plotly figure to multiple formats.
    
    Args:
        fig: Plotly Figure object
        filepath: Base filepath (without extension)
        formats: List of formats to save (e.g., ["html", "png"])
    
    Note:
        PNG export requires kaleido package: pip install kaleido
    """
    saved_formats = []
    
    for fmt in formats:
        try:
            output_path = filepath.with_suffix(f".{fmt}")
            
            if fmt == "html":
                fig.write_html(output_path)
                saved_formats.append(fmt)
            elif fmt == "png":
                # PNG requires kaleido
                fig.write_image(output_path, width=1200, height=900)
                saved_formats.append(fmt)
            elif fmt == "svg":
                fig.write_image(output_path, width=1200, height=900)
                saved_formats.append(fmt)
            elif fmt == "pdf":
                fig.write_image(output_path, width=1200, height=900)
                saved_formats.append(fmt)
            else:
                print(f"Warning: Unknown format '{fmt}', skipping")
                
        except Exception as e:
            if "kaleido" in str(e).lower() or "image export" in str(e).lower():
                print(f"Warning: Cannot save as {fmt} (kaleido not installed). Install with: pip install kaleido")
            else:
                print(f"Warning: Failed to save as {fmt}: {e}")
    
    return saved_formats


def umap_reduce_3d(embeddings: torch.Tensor, random_state: int = 42) -> torch.Tensor:
    """Fit UMAP on embeddings and return 3D reduced embeddings."""
    reducer = umap.UMAP(n_components=3, random_state=random_state)
    umap_embeddings = reducer.fit_transform(embeddings.to(torch.float32).cpu().numpy())
    return torch.tensor(umap_embeddings, dtype=torch.float32)


def plot_umap_3d(
    embedding_3d: torch.Tensor, title: str, labels: list[str], palette: list[str] = None
) -> go.Figure:
    """Plot 3D UMAP embedding using Plotly, coloring by categorical label."""
    # Ensure data is NumPy for Plotly rendering
    coords = (
        embedding_3d.detach().cpu().numpy()
        if isinstance(embedding_3d, torch.Tensor)
        else np.asarray(embedding_3d)
    )
    assert coords.shape[1] == 3, "Embedding must be 3D"
    assert len(labels) == coords.shape[0], "Labels length must match number of points"

    # Map labels to colors (cycle palette if needed)
    unique_labels = sorted(set(labels))
    if palette is None:
        palette = px.colors.qualitative.Light24
    if len(unique_labels) > len(palette):
        repeats = (len(unique_labels) // len(palette)) + 1
        palette = (palette * repeats)[: len(unique_labels)]
    color_map = {label: color for label, color in zip(unique_labels, palette)}
    colors = [color_map[label] for label in labels]

    # Create main trace with points
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    opacity=0.7,
                    color=colors,
                ),
                text=[
                    f"Genome {i}<br>Label: {labels[i]}" for i in range(coords.shape[0])
                ],
                hovertemplate="<b>%{text}</b><br>UMAP 1: %{x}<br>UMAP 2: %{y}<br>UMAP 3: %{z}<extra></extra>",
                showlegend=False,
            )
        ]
    )

    # Add legend-only entries for categories
    for label, color in color_map.items():
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(
                    size=4,
                    opacity=0.7,
                    color=color,
                ),
                name=label,
                showlegend=True,
                visible="legendonly",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
        width=800,
        height=600,
        legend=dict(itemsizing="constant"),
    )
    return fig


def plot_distance_scatter(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    marker_size: int = 5,
    marker_opacity: float = 0.3,
) -> go.Figure:
    """
    Create a scatter plot comparing two distance metrics.

    Args:
        x: X-axis values (e.g., phylogenetic distances)
        y: Y-axis values (e.g., embedding distances)
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
        marker_size: Size of scatter points
        marker_opacity: Opacity of scatter points

    Returns:
        Plotly Figure object
    """
    fig = px.scatter(
        x=x,
        y=y,
        labels={"x": x_label, "y": y_label},
        title=title,
        trendline_color_override="red",
    )
    fig.update_traces(marker=dict(size=marker_size, opacity=marker_opacity))
    return fig
