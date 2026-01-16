"""
Utility functions for plotting and data handling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def set_plotting_style():
    """Set consistent plotting style."""
    plt.rcParams.update({
        'figure.figsize': (10, 8),
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training history.

    Args:
        history: Training history dict
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Training loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')

    # Training accuracy
    ax = axes[0, 1]
    ax.plot(history['train_accuracy'], alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.set_ylim([0, 1])

    # Validation accuracy by sequence length
    ax = axes[1, 0]
    for seq_len, metrics_list in history['val_metrics'].items():
        if metrics_list:
            epochs = history['epochs'][:len(metrics_list)]
            accs = [m['item_accuracy'] for m in metrics_list]
            ax.plot(epochs, accs, label=f'seq_len={seq_len}', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy by Sequence Length')
    ax.legend()
    ax.set_ylim([0, 1])

    # Curriculum schedule
    ax = axes[1, 1]
    ax.plot(history['seq_len_schedule'], alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sequence Length')
    ax.set_title('Curriculum Schedule')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_pca_trajectories(
    hidden_states: np.ndarray,
    infos: List[dict],
    task,
    seq_len: int,
    n_trials: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot PCA trajectories of hidden states with stimulus onset markers.

    Args:
        hidden_states: (batch, seq_len+1, hidden_dim) hidden states
        infos: Trial info dicts
        task: Task object
        seq_len: Sequence length
        n_trials: Number of trials to plot
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    from sklearn.decomposition import PCA
    from analysis import get_timepoint_indices

    # Fit PCA on all data
    batch, total_time, hidden_dim = hidden_states.shape
    pca = PCA(n_components=3)
    all_states = hidden_states.reshape(-1, hidden_dim)
    pca.fit(all_states)

    # Get phase indices
    indices = get_timepoint_indices(task, seq_len)

    # Create figure
    fig = plt.figure(figsize=(16, 5))

    # 2D plot (PC1 vs PC2)
    ax1 = fig.add_subplot(131)
    # 2D plot (PC1 vs PC3)
    ax2 = fig.add_subplot(132)
    # 3D plot
    ax3 = fig.add_subplot(133, projection='3d')

    # Stimulus colors
    stim_colors = plt.cm.tab10(np.linspace(0, 1, seq_len))

    for i in range(min(n_trials, batch)):
        trajectory = hidden_states[i]  # (total_time, hidden_dim)
        projected = pca.transform(trajectory)  # (total_time, 3)

        # Plot trajectory with time coloring
        colors = plt.cm.viridis(np.linspace(0, 1, total_time))
        for t in range(total_time - 1):
            ax1.plot(projected[t:t+2, 0], projected[t:t+2, 1],
                    color=colors[t], alpha=0.3, linewidth=0.5)
            ax2.plot(projected[t:t+2, 0], projected[t:t+2, 2],
                    color=colors[t], alpha=0.3, linewidth=0.5)
            ax3.plot(projected[t:t+2, 0], projected[t:t+2, 1], projected[t:t+2, 2],
                    color=colors[t], alpha=0.3, linewidth=0.5)

        # Mark stimulus onsets with colored markers
        for s, stim_idx in enumerate(indices['stimulus']):
            stim_onset = stim_idx[0] + 1  # +1 for hidden state indexing
            marker_label = f'Stim {s+1}' if i == 0 else None
            ax1.scatter(projected[stim_onset, 0], projected[stim_onset, 1],
                       c=[stim_colors[s]], s=60, marker='^', zorder=10,
                       edgecolors='black', linewidths=0.5, label=marker_label)
            ax2.scatter(projected[stim_onset, 0], projected[stim_onset, 2],
                       c=[stim_colors[s]], s=60, marker='^', zorder=10,
                       edgecolors='black', linewidths=0.5)
            ax3.scatter(projected[stim_onset, 0], projected[stim_onset, 1], projected[stim_onset, 2],
                       c=[stim_colors[s]], s=60, marker='^', zorder=10,
                       edgecolors='black', linewidths=0.5)

        # Mark delay onset
        delay_onset = indices['delay'][0] + 1
        if i == 0:
            ax1.scatter(projected[delay_onset, 0], projected[delay_onset, 1],
                       c='orange', s=80, marker='s', zorder=10,
                       edgecolors='black', linewidths=0.5, label='Delay')
        else:
            ax1.scatter(projected[delay_onset, 0], projected[delay_onset, 1],
                       c='orange', s=80, marker='s', zorder=10,
                       edgecolors='black', linewidths=0.5)

        # Mark start and end
        if i == 0:
            ax1.scatter(projected[0, 0], projected[0, 1], c='green', s=40, zorder=5, label='Start')
            ax1.scatter(projected[-1, 0], projected[-1, 1], c='red', s=40, zorder=5, label='End')
        else:
            ax1.scatter(projected[0, 0], projected[0, 1], c='green', s=40, zorder=5)
            ax1.scatter(projected[-1, 0], projected[-1, 1], c='red', s=40, zorder=5)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_title('PC1 vs PC2')
    ax1.legend(loc='best', fontsize=7)

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax2.set_title('PC1 vs PC3')

    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_zlabel('PC3')
    ax3.set_title('3D Trajectory')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_mean_trajectory_by_phase(
    hidden_states: np.ndarray,
    task,
    seq_len: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot mean PCA trajectory colored by task phase.

    Args:
        hidden_states: (batch, seq_len+1, hidden_dim)
        task: Task object
        seq_len: Sequence length
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    from sklearn.decomposition import PCA
    from analysis import get_timepoint_indices

    # Compute mean trajectory
    mean_trajectory = hidden_states.mean(axis=0)

    # Fit PCA
    pca = PCA(n_components=3)
    projected = pca.fit_transform(mean_trajectory)

    # Get phase indices
    indices = get_timepoint_indices(task, seq_len)
    total_time = mean_trajectory.shape[0]

    # Create phase labels
    phase_colors = {
        'fixation': 'gray',
        'stimulus': 'blue',
        'delay': 'orange',
        'response': 'green',
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 2D plot
    ax = axes[0]
    current_t = 0

    # Plot fixation
    fix_end = len(indices['fixation'])
    ax.plot(projected[current_t:fix_end+1, 0], projected[current_t:fix_end+1, 1],
            color=phase_colors['fixation'], linewidth=2, label='Fixation')
    current_t = fix_end

    # Plot stimuli
    for i, stim_idx in enumerate(indices['stimulus']):
        stim_start = stim_idx[0]
        stim_end = stim_idx[-1] + 1
        ax.plot(projected[stim_start:stim_end+1, 0], projected[stim_start:stim_end+1, 1],
                color=plt.cm.Blues(0.3 + 0.7*i/len(indices['stimulus'])),
                linewidth=2, label=f'Stim {i+1}' if i == 0 else '')
        # Mark stimulus onset
        ax.scatter(projected[stim_start, 0], projected[stim_start, 1],
                  c='blue', s=50, marker='^', zorder=5)

    # Plot delay
    delay_start = indices['delay'][0]
    delay_end = indices['delay'][-1] + 1
    ax.plot(projected[delay_start:delay_end+1, 0], projected[delay_start:delay_end+1, 1],
            color=phase_colors['delay'], linewidth=2, label='Delay')

    # Plot responses
    for i, resp_idx in enumerate(indices['response']):
        resp_start = resp_idx[0]
        resp_end = resp_idx[-1] + 1
        ax.plot(projected[resp_start:resp_end+1, 0], projected[resp_start:resp_end+1, 1],
                color=plt.cm.Greens(0.3 + 0.7*i/len(indices['response'])),
                linewidth=2, label=f'Resp {i+1}' if i == 0 else '')
        # Mark response cue
        ax.scatter(projected[resp_start, 0], projected[resp_start, 1],
                  c='green', s=50, marker='v', zorder=5)

    ax.scatter(projected[0, 0], projected[0, 1], c='black', s=100, marker='o', label='Start', zorder=6)
    ax.scatter(projected[-1, 0], projected[-1, 1], c='red', s=100, marker='x', label='End', zorder=6)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('Mean Trajectory by Phase')
    ax.legend(loc='best', fontsize=8)

    # Time series of PC values
    ax = axes[1]
    time = np.arange(total_time)
    ax.plot(time, projected[:, 0], label='PC1')
    ax.plot(time, projected[:, 1], label='PC2')
    ax.plot(time, projected[:, 2], label='PC3')

    # Shade phases
    fix_end = len(indices['fixation'])
    ax.axvspan(0, fix_end, alpha=0.2, color='gray', label='Fixation')

    for stim_idx in indices['stimulus']:
        ax.axvspan(stim_idx[0], stim_idx[-1]+1, alpha=0.2, color='blue')

    ax.axvspan(indices['delay'][0], indices['delay'][-1]+1, alpha=0.2, color='orange')

    for resp_idx in indices['response']:
        ax.axvspan(resp_idx[0], resp_idx[-1]+1, alpha=0.2, color='green')

    ax.set_xlabel('Time')
    ax.set_ylabel('PC Value')
    ax.set_title('PC Values Over Time')
    ax.legend(loc='best')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_rdm(
    rdm: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = 'RDM',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot representational dissimilarity matrix.

    Args:
        rdm: (n, n) dissimilarity matrix
        labels: Labels for conditions
        title: Plot title
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(rdm, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Dissimilarity')

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_cross_temporal_decoding(
    gen_matrix: np.ndarray,
    time_labels: Optional[List[str]] = None,
    title: str = 'Cross-temporal Decoding',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot cross-temporal generalization matrix.

    Args:
        gen_matrix: (n_times, n_times) generalization accuracy matrix
        time_labels: Labels for timepoints
        title: Plot title
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(gen_matrix, cmap='RdBu_r', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Accuracy')

    if time_labels:
        n = len(time_labels)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(time_labels, rotation=45, ha='right')
        ax.set_yticklabels(time_labels)

    ax.set_xlabel('Test Time')
    ax.set_ylabel('Train Time')
    ax.set_title(title)

    # Add diagonal line
    ax.plot([0, gen_matrix.shape[0]-1], [0, gen_matrix.shape[1]-1],
            'k--', alpha=0.5, linewidth=1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_decoding_results(
    decoding_results: Dict,
    seq_len: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot decoding accuracy results.

    Args:
        decoding_results: Dict from decode_item_and_position
        seq_len: Sequence length
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Item at position (stimulus end)
    ax = axes[0]
    positions = list(range(seq_len))
    accs = [decoding_results['item_at_position'][p]['accuracy'] for p in positions]
    stds = [decoding_results['item_at_position'][p]['std'] for p in positions]

    ax.bar(positions, accs, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
    ax.axhline(1/8, color='gray', linestyle='--', label='Chance (1/8)')
    ax.set_xlabel('Position')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_title('Item Identity Decoding\n(at stimulus offset)')
    ax.set_ylim([0, 1])
    ax.set_xticks(positions)
    ax.set_xticklabels([f'Pos {p+1}' for p in positions])
    ax.legend()

    # Item at position (during delay midpoint)
    ax = axes[1]
    delay_accs = []
    delay_stds = []
    for p in positions:
        key = f'delay_pos{p}'
        if key in decoding_results['item_at_position']:
            delay_accs.append(decoding_results['item_at_position'][key]['accuracy'])
            delay_stds.append(decoding_results['item_at_position'][key]['std'])
        else:
            delay_accs.append(np.nan)
            delay_stds.append(np.nan)

    ax.bar(positions, delay_accs, yerr=delay_stds, capsize=5, color='darkorange', alpha=0.7)
    ax.axhline(1/8, color='gray', linestyle='--', label='Chance (1/8)')
    ax.set_xlabel('Position')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_title('Item Identity Decoding\n(delay midpoint)')
    ax.set_ylim([0, 1])
    ax.set_xticks(positions)
    ax.set_xticklabels([f'Pos {p+1}' for p in positions])
    ax.legend()

    # Delay timecourse
    ax = axes[2]
    if 'delay_timecourse' in decoding_results and decoding_results['delay_timecourse']:
        colors = plt.cm.tab10(np.linspace(0, 1, seq_len))
        for pos in range(seq_len):
            if pos in decoding_results['delay_timecourse']:
                timecourse = decoding_results['delay_timecourse'][pos]
                ax.plot(timecourse, color=colors[pos], linewidth=2,
                       label=f'Position {pos+1}', marker='o', markersize=4)

        ax.axhline(1/8, color='gray', linestyle='--', label='Chance')
        ax.set_xlabel('Delay Timestep')
        ax.set_ylabel('Decoding Accuracy')
        ax.set_title('Item Decoding Throughout Delay')
        ax.set_ylim([0, 1])
        ax.legend(loc='best')
    else:
        ax.text(0.5, 0.5, 'No timecourse data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Item Decoding Throughout Delay')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_subspace_analysis(
    subspace_results: Dict,
    seq_len: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot subspace analysis results.

    Args:
        subspace_results: Dict from subspace_analysis
        seq_len: Sequence length
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Explained variance per position
    ax = axes[0]
    for pos in range(seq_len):
        var = subspace_results['subspaces'][pos]['explained_variance']
        ax.plot(range(1, len(var)+1), np.cumsum(var), marker='o', label=f'Position {pos+1}')

    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('Variance Explained by Position')
    ax.legend()
    ax.set_ylim([0, 1])

    # Principal angles between subspaces
    ax = axes[1]
    angles_data = []
    labels = []
    for (i, j), data in subspace_results['principal_angles'].items():
        angles_data.append(data['angles_deg'][:5])  # First 5 angles
        labels.append(f'{i+1} vs {j+1}')

    if angles_data:
        x = np.arange(len(angles_data[0]))
        width = 0.8 / len(angles_data)
        for idx, (angles, label) in enumerate(zip(angles_data, labels)):
            ax.bar(x + idx*width, angles, width, label=label, alpha=0.7)

        ax.set_xlabel('Principal Angle Index')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Principal Angles Between Position Subspaces')
        ax.legend()
        ax.axhline(90, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def create_analysis_report(
    results: Dict,
    save_dir: str,
) -> None:
    """
    Create a complete analysis report with all figures.

    Args:
        results: Results from run_all_analyses
        save_dir: Directory to save figures
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_plotting_style()

    # Summary figure
    fig, axes = plt.subplots(2, len(results), figsize=(5*len(results), 10))
    if len(results) == 1:
        axes = axes.reshape(-1, 1)

    for i, (seq_len, seq_results) in enumerate(results.items()):
        # Dimensionality
        ax = axes[0, i]
        ax.bar(['Dimensionality'], [seq_results['geometry']['dimensionality']])
        ax.set_title(f'Seq Len = {seq_len}')
        ax.set_ylabel('Participation Ratio')

        # RDM correlations
        ax = axes[1, i]
        item_corr = seq_results['geometry'].get('item_rdm_correlation', np.nan)
        pos_corr = seq_results['geometry'].get('position_rdm_correlation', np.nan)
        ax.bar(['Item', 'Position'], [item_corr, pos_corr])
        ax.set_ylabel('RDM Correlation')
        ax.set_ylim([-1, 1])
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(save_path / 'summary.png', bbox_inches='tight')
    plt.close(fig)

    print(f"Analysis report saved to {save_dir}")


if __name__ == "__main__":
    # Test plotting functions
    set_plotting_style()

    # Create dummy data
    np.random.seed(42)
    n = 10
    rdm = np.random.rand(n, n)
    rdm = (rdm + rdm.T) / 2  # Make symmetric
    np.fill_diagonal(rdm, 0)

    fig = plot_rdm(rdm, labels=[f'Cond {i}' for i in range(n)])
    plt.show()

    gen_matrix = np.random.rand(20, 20)
    fig = plot_cross_temporal_decoding(gen_matrix)
    plt.show()
