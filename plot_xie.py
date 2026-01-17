"""
Visualization functions for Xie et al. subspace analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List
from pathlib import Path


def set_style():
    """Set plotting style."""
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            pass  # Use default style
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['figure.facecolor'] = 'white'


def plot_angle_matrix(
    results: Dict,
    seq_len: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot principal angles as a matrix heatmap.

    Args:
        results: Results from xie_subspace_analysis
        seq_len: Sequence length
        save_path: Optional path to save figure

    Returns:
        fig: Matplotlib figure
    """
    set_style()

    # Create angle matrix
    angle_matrix = np.zeros((seq_len, seq_len))
    angle_matrix[:] = np.nan

    mean_angles = results.get('mean_angles_deg', {})
    for (i, j), angle in mean_angles.items():
        angle_matrix[i, j] = angle
        angle_matrix[j, i] = angle

    # Fill diagonal with 0 (same subspace)
    np.fill_diagonal(angle_matrix, 0)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Heatmap
    mask = np.triu(np.ones_like(angle_matrix, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        angle_matrix,
        mask=~np.tril(np.ones_like(angle_matrix, dtype=bool)),
        annot=True,
        fmt='.1f',
        cmap='viridis',
        vmin=0,
        vmax=90,
        ax=ax,
        cbar_kws={'label': 'Mean Principal Angle (°)'},
        square=True,
    )

    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_xticklabels([f'Pos {i+1}' for i in range(seq_len)])
    ax.set_yticklabels([f'Pos {i+1}' for i in range(seq_len)])
    ax.set_title(f'Subspace Angles Between Positions (seq_len={seq_len})\n(90° = orthogonal)')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_principal_angles_detail(
    results: Dict,
    seq_len: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot all principal angles (not just mean) for each pair.

    Args:
        results: Results from xie_subspace_analysis
        seq_len: Sequence length
        save_path: Optional path to save figure

    Returns:
        fig: Matplotlib figure
    """
    set_style()

    principal_angles = results.get('principal_angles_deg', {})
    n_pairs = len(principal_angles)

    if n_pairs == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=(8, 5))

    x_positions = []
    colors = plt.cm.Set2(np.linspace(0, 1, n_pairs))

    for idx, ((i, j), angles) in enumerate(sorted(principal_angles.items())):
        x = np.arange(len(angles)) + idx * (len(angles) + 1)
        x_positions.append((x.mean(), f'Pos {i+1}\nvs\nPos {j+1}'))

        bars = ax.bar(x, angles, color=colors[idx], edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar, angle in zip(bars, angles):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{angle:.0f}°', ha='center', va='bottom', fontsize=9)

    # Add 90° reference line
    ax.axhline(90, color='red', linestyle='--', alpha=0.7, label='Orthogonal (90°)')

    ax.set_ylabel('Principal Angle (°)')
    ax.set_ylim(0, 100)
    ax.set_xticks([pos for pos, _ in x_positions])
    ax.set_xticklabels([label for _, label in x_positions])
    ax.set_title(f'Principal Angles Between Position Subspaces (seq_len={seq_len})')
    ax.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_angle_timecourse(
    timecourse_results: Dict,
    seq_len: int,
    task=None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot how subspace angles evolve through the delay period.

    Args:
        timecourse_results: Results from run_xie_analysis_over_time
        seq_len: Sequence length
        task: Task object (for timing info)
        save_path: Optional path to save figure

    Returns:
        fig: Matplotlib figure
    """
    set_style()

    angle_timecourse = timecourse_results.get('angle_timecourse', {})
    time_indices = timecourse_results.get('time_indices', [])

    if not angle_timecourse or not time_indices:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No timecourse data', ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(angle_timecourse)))

    for idx, ((i, j), angles) in enumerate(sorted(angle_timecourse.items())):
        ax.plot(range(len(angles)), angles, 'o-', color=colors[idx],
               label=f'Pos {i+1} vs Pos {j+1}', linewidth=2, markersize=6)

    # Add 90° reference line
    ax.axhline(90, color='red', linestyle='--', alpha=0.5, label='Orthogonal')

    ax.set_xlabel('Delay Timestep')
    ax.set_ylabel('Mean Principal Angle (°)')
    ax.set_ylim(0, 100)
    ax.set_title(f'Subspace Angle Evolution During Delay (seq_len={seq_len})')
    ax.legend(loc='lower right', ncol=2)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_coding_planes_3d(
    results: Dict,
    seq_len: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize coding planes projected into common 3D space.

    Projects the 2D coding planes for each position into a shared
    3D space to show their relative orientations.

    Args:
        results: Results from xie_subspace_analysis
        seq_len: Sequence length
        save_path: Optional path to save figure

    Returns:
        fig: Matplotlib figure
    """
    set_style()

    planes = results.get('planes', [])
    position_betas = results.get('position_betas', [])

    if not planes or not position_betas:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No plane data', ha='center', va='center')
        return fig

    # Stack all betas and do joint PCA to get common 3D space
    all_betas = np.vstack([pb.T for pb in position_betas])  # (n_items * seq_len, n_neurons)

    from sklearn.decomposition import PCA
    pca_3d = PCA(n_components=3)
    pca_3d.fit(all_betas)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.Set1(np.linspace(0, 1, seq_len))

    for pos in range(seq_len):
        # Project this position's item representations into common space
        item_reps = position_betas[pos].T  # (n_items, n_neurons)
        projected = pca_3d.transform(item_reps)  # (n_items, 3)

        # Plot points
        ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                  c=[colors[pos]], s=100, alpha=0.8, label=f'Position {pos+1}')

        # Connect points to show the "plane"
        for i in range(len(projected)):
            for j in range(i+1, len(projected)):
                ax.plot([projected[i, 0], projected[j, 0]],
                       [projected[i, 1], projected[j, 1]],
                       [projected[i, 2], projected[j, 2]],
                       color=colors[pos], alpha=0.2, linewidth=0.5)

        # Draw plane as a surface
        if planes[pos] is not None and planes[pos].shape[1] >= 2:
            # Project plane basis vectors
            plane_vecs = planes[pos][:, :2]  # (n_neurons, 2)

            # Create grid in plane coordinates
            u = np.linspace(-1, 1, 10)
            v = np.linspace(-1, 1, 10)
            U, V = np.meshgrid(u, v)

            # Center of this position's representations
            center = item_reps.mean(axis=0)

            # Scale factor based on spread of points
            scale = np.std(projected) * 2

            # Points on the plane in neuron space, then project to 3D
            plane_points = []
            for ui, vi in zip(U.flatten(), V.flatten()):
                point = center + scale * (ui * plane_vecs[:, 0] + vi * plane_vecs[:, 1])
                plane_points.append(point)
            plane_points = np.array(plane_points)
            plane_3d = pca_3d.transform(plane_points)

            X = plane_3d[:, 0].reshape(U.shape)
            Y = plane_3d[:, 1].reshape(U.shape)
            Z = plane_3d[:, 2].reshape(U.shape)

            ax.plot_surface(X, Y, Z, color=colors[pos], alpha=0.15)

    ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
    ax.set_title(f'Item Coding Planes in Common Neural Space (seq_len={seq_len})')
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_regression_weights(
    results: Dict,
    seq_len: int,
    n_stimuli: int = 8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the regression weight matrix.

    Args:
        results: Results from xie_subspace_analysis
        seq_len: Sequence length
        n_stimuli: Number of stimuli
        save_path: Optional path to save figure

    Returns:
        fig: Matplotlib figure
    """
    set_style()

    betas = results.get('betas')
    if betas is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No beta data', ha='center', va='center')
        return fig

    fig, axes = plt.subplots(1, seq_len + 1, figsize=(4 * (seq_len + 1), 5),
                             gridspec_kw={'width_ratios': [1] * seq_len + [0.05]})

    # Split betas by position
    vmax = np.abs(betas).max()

    for pos in range(seq_len):
        ax = axes[pos]
        start_idx = pos * n_stimuli
        end_idx = (pos + 1) * n_stimuli
        pos_betas = betas[:, start_idx:end_idx]

        im = ax.imshow(pos_betas, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('Item')
        ax.set_ylabel('Neuron' if pos == 0 else '')
        ax.set_title(f'Position {pos + 1}')
        ax.set_xticks(range(n_stimuli))
        ax.set_xticklabels([f'{i+1}' for i in range(n_stimuli)])

    # Colorbar
    plt.colorbar(im, cax=axes[-1], label='Regression Weight')

    fig.suptitle(f'Regression Weights: Item Identity at Each Position (seq_len={seq_len})')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_full_xie_analysis(
    full_results: Dict,
    save_dir: str,
) -> None:
    """
    Generate all visualization plots for Xie analysis.

    Args:
        full_results: Results from run_full_xie_analysis
        save_dir: Directory to save figures
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    seq_len = full_results['seq_len']
    n_stimuli = full_results['n_stimuli']

    # 1. Angle matrix
    plot_angle_matrix(
        full_results['delay_analysis'],
        seq_len,
        save_path=save_path / f'angle_matrix_sl{seq_len}.png'
    )
    print(f"  Saved angle_matrix_sl{seq_len}.png")

    # 2. Principal angles detail
    plot_principal_angles_detail(
        full_results['delay_analysis'],
        seq_len,
        save_path=save_path / f'principal_angles_sl{seq_len}.png'
    )
    print(f"  Saved principal_angles_sl{seq_len}.png")

    # 3. Angle timecourse
    plot_angle_timecourse(
        full_results['delay_timecourse'],
        seq_len,
        save_path=save_path / f'angle_timecourse_sl{seq_len}.png'
    )
    print(f"  Saved angle_timecourse_sl{seq_len}.png")

    # 4. 3D coding planes
    plot_coding_planes_3d(
        full_results['delay_analysis'],
        seq_len,
        save_path=save_path / f'coding_planes_3d_sl{seq_len}.png'
    )
    print(f"  Saved coding_planes_3d_sl{seq_len}.png")

    # 5. Regression weights
    plot_regression_weights(
        full_results['delay_analysis'],
        seq_len,
        n_stimuli=n_stimuli,
        save_path=save_path / f'regression_weights_sl{seq_len}.png'
    )
    print(f"  Saved regression_weights_sl{seq_len}.png")


if __name__ == "__main__":
    import torch
    from task import SequenceWorkingMemoryTask
    from model import create_model
    from train import load_model
    from analysis_xie import run_full_xie_analysis

    # Find GRU model
    results_dir = Path("results")
    model_path = None
    model_type = 'vanilla'

    if results_dir.exists():
        for exp_dir in sorted(results_dir.iterdir(), reverse=True):
            if (exp_dir / "final_model.pt").exists():
                checkpoint = torch.load(exp_dir / "final_model.pt", map_location='cpu')
                keys = list(checkpoint['model_state_dict'].keys())
                if any('gru' in k for k in keys):
                    model_path = exp_dir / "final_model.pt"
                    model_type = 'gru'
                    break

        if model_path is None:
            for exp_dir in sorted(results_dir.iterdir(), reverse=True):
                if (exp_dir / "final_model.pt").exists():
                    model_path = exp_dir / "final_model.pt"
                    model_type = 'vanilla'
                    break

    # Create task
    task = SequenceWorkingMemoryTask(n_stimuli=8, seq_lengths=[2, 3, 4])

    if model_path:
        print(f"Loading model from {model_path} (type: {model_type})")
        model, checkpoint = load_model(str(model_path), model_type=model_type, device='cpu')
    else:
        print("No trained model found, using random model")
        model = create_model('gru', task.input_dim, 128, task.output_dim)

    # Run analysis and plot for each sequence length
    figures_dir = Path("results/xie_figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    for seq_len in [2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"Generating figures for seq_len = {seq_len}")
        print('='*60)

        results = run_full_xie_analysis(
            model, task, seq_len,
            n_trials=200,
            device='cpu',
            n_components=2
        )

        plot_full_xie_analysis(results, str(figures_dir))

    print(f"\nAll figures saved to {figures_dir}")
