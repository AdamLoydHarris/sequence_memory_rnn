"""
Subspace analysis following Xie et al. Science 2022.

This implements the regression-based subspace analysis:
1. Regress neural activity against item-at-position design matrix
2. Split regression weights by position (rank)
3. PCA on each position's weights to get 2D coding planes
4. Compute principal angles between planes
"""

import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import warnings


def create_design_matrix(
    sequences: np.ndarray,
    n_stimuli: int,
) -> np.ndarray:
    """
    Create the design matrix for regression.

    Each trial is represented as a multi-hot vector:
    - First n_stimuli elements: one-hot for item at position 0
    - Next n_stimuli elements: one-hot for item at position 1
    - etc.

    Args:
        sequences: (n_trials, seq_len) array of item indices
        n_stimuli: Number of unique stimuli

    Returns:
        X: (n_trials, n_stimuli * seq_len) design matrix
    """
    n_trials, seq_len = sequences.shape
    n_regressors = n_stimuli * seq_len

    X = np.zeros((n_trials, n_regressors))

    for trial_idx in range(n_trials):
        for pos in range(seq_len):
            item = sequences[trial_idx, pos]
            regressor_idx = pos * n_stimuli + item
            X[trial_idx, regressor_idx] = 1.0

    return X


def regress_activity(
    hidden_states: np.ndarray,
    design_matrix: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Regress each neuron's activity against the design matrix.

    Args:
        hidden_states: (n_trials, n_neurons) activity at a single timepoint
        design_matrix: (n_trials, n_regressors) design matrix
        alpha: Ridge regression regularization parameter

    Returns:
        betas: (n_neurons, n_regressors) regression weights
    """
    n_trials, n_neurons = hidden_states.shape
    n_regressors = design_matrix.shape[1]

    # Ridge regression for each neuron
    # Could do this in one go with multi-output regression
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(design_matrix, hidden_states)

    # betas shape: (n_regressors, n_neurons) from sklearn, transpose to (n_neurons, n_regressors)
    betas = ridge.coef_.T  # (n_neurons, n_regressors)

    return betas


def split_betas_by_position(
    betas: np.ndarray,
    n_stimuli: int,
    seq_len: int,
) -> List[np.ndarray]:
    """
    Split the regression weight matrix by position.

    Args:
        betas: (n_neurons, n_regressors) regression weights
        n_stimuli: Number of unique stimuli
        seq_len: Sequence length

    Returns:
        position_betas: List of (n_neurons, n_stimuli) arrays, one per position
    """
    position_betas = []
    for pos in range(seq_len):
        start_idx = pos * n_stimuli
        end_idx = (pos + 1) * n_stimuli
        position_betas.append(betas[:, start_idx:end_idx])

    return position_betas


def compute_coding_planes(
    position_betas: List[np.ndarray],
    n_components: int = 2,
) -> List[np.ndarray]:
    """
    Compute the coding plane for each position using PCA.

    Args:
        position_betas: List of (n_neurons, n_stimuli) arrays
        n_components: Number of PCA components (defines plane dimensionality)

    Returns:
        planes: List of (n_neurons, n_components) arrays defining each plane
    """
    planes = []

    for pos_beta in position_betas:
        # PCA on the (n_neurons, n_stimuli) matrix
        # We want to find the principal directions in neuron space
        # that capture variance across items at this position

        # Transpose to (n_stimuli, n_neurons) for PCA
        # Each "sample" is an item, each "feature" is a neuron
        data = pos_beta.T  # (n_stimuli, n_neurons)

        n_comp = min(n_components, data.shape[0] - 1, data.shape[1])
        if n_comp < 1:
            planes.append(None)
            continue

        pca = PCA(n_components=n_comp)
        pca.fit(data)

        # Components are (n_components, n_neurons)
        # Transpose to get (n_neurons, n_components) - each column is a PC direction
        plane = pca.components_.T  # (n_neurons, n_components)
        planes.append(plane)

    return planes


def compute_principal_angles(
    plane1: np.ndarray,
    plane2: np.ndarray,
) -> np.ndarray:
    """
    Compute principal angles between two subspaces.

    The principal angles are computed via SVD of the inner product matrix
    between the orthonormal bases of the two subspaces.

    Args:
        plane1: (n_neurons, k1) orthonormal basis for plane 1
        plane2: (n_neurons, k2) orthonormal basis for plane 2

    Returns:
        angles: Array of principal angles in radians
    """
    # Ensure orthonormal (PCA components should already be orthonormal)
    # But let's be safe
    Q1, _ = np.linalg.qr(plane1)
    Q2, _ = np.linalg.qr(plane2)

    # Inner product matrix
    M = Q1.T @ Q2  # (k1, k2)

    # SVD to get principal angles
    _, s, _ = np.linalg.svd(M)

    # Singular values are cosines of principal angles
    # Clip to handle numerical issues
    s = np.clip(s, -1, 1)
    angles = np.arccos(s)

    return angles


def xie_subspace_analysis(
    hidden_states: np.ndarray,
    sequences: np.ndarray,
    n_stimuli: int,
    seq_len: int,
    alpha: float = 1.0,
    n_components: int = 2,
) -> Dict:
    """
    Full Xie et al. subspace analysis pipeline.

    Args:
        hidden_states: (n_trials, n_neurons) activity at delay period
        sequences: (n_trials, seq_len) item indices
        n_stimuli: Number of unique stimuli
        seq_len: Sequence length
        alpha: Ridge regression regularization
        n_components: Number of PCA components for each plane

    Returns:
        results: Dict with regression weights, planes, and angles
    """
    results = {}

    # Step 1: Create design matrix
    design_matrix = create_design_matrix(sequences, n_stimuli)
    results['design_matrix_shape'] = design_matrix.shape

    # Step 2: Regress activity against design matrix
    betas = regress_activity(hidden_states, design_matrix, alpha=alpha)
    results['betas_shape'] = betas.shape
    results['betas'] = betas

    # Step 3: Split betas by position
    position_betas = split_betas_by_position(betas, n_stimuli, seq_len)
    results['position_betas'] = position_betas

    # Step 4: Compute coding planes via PCA
    planes = compute_coding_planes(position_betas, n_components=n_components)
    results['planes'] = planes

    # Store explained variance for each position's PCA
    results['pca_explained_variance'] = {}
    for pos, pos_beta in enumerate(position_betas):
        data = pos_beta.T
        n_comp = min(n_components, data.shape[0] - 1, data.shape[1])
        if n_comp >= 1:
            pca = PCA(n_components=n_comp)
            pca.fit(data)
            results['pca_explained_variance'][pos] = pca.explained_variance_ratio_.tolist()

    # Step 5: Compute principal angles between all pairs of planes
    results['principal_angles'] = {}
    results['principal_angles_deg'] = {}
    results['mean_angles_deg'] = {}

    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if planes[i] is not None and planes[j] is not None:
                angles = compute_principal_angles(planes[i], planes[j])
                angles_deg = np.degrees(angles)

                results['principal_angles'][(i, j)] = angles.tolist()
                results['principal_angles_deg'][(i, j)] = angles_deg.tolist()
                results['mean_angles_deg'][(i, j)] = float(np.mean(angles_deg))

    return results


def run_xie_analysis_over_time(
    hidden_states: np.ndarray,
    sequences: np.ndarray,
    n_stimuli: int,
    seq_len: int,
    time_indices: List[int],
    alpha: float = 1.0,
    n_components: int = 2,
) -> Dict:
    """
    Run Xie analysis at multiple timepoints.

    Args:
        hidden_states: (n_trials, n_timepoints, n_neurons)
        sequences: (n_trials, seq_len) item indices
        n_stimuli: Number of unique stimuli
        seq_len: Sequence length
        time_indices: Which timepoints to analyze
        alpha: Ridge regression regularization
        n_components: Number of PCA components

    Returns:
        results: Dict with analysis at each timepoint
    """
    results = {
        'time_indices': time_indices,
        'timepoint_results': {},
        'angle_timecourse': {},  # Mean angle between each pair over time
    }

    # Initialize angle timecourse storage
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            results['angle_timecourse'][(i, j)] = []

    for t_idx, t in enumerate(time_indices):
        states_t = hidden_states[:, t, :]

        analysis = xie_subspace_analysis(
            states_t, sequences, n_stimuli, seq_len,
            alpha=alpha, n_components=n_components
        )

        results['timepoint_results'][t] = {
            'mean_angles_deg': analysis['mean_angles_deg'],
            'principal_angles_deg': analysis['principal_angles_deg'],
        }

        # Store for timecourse
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                if (i, j) in analysis['mean_angles_deg']:
                    results['angle_timecourse'][(i, j)].append(
                        analysis['mean_angles_deg'][(i, j)]
                    )
                else:
                    results['angle_timecourse'][(i, j)].append(np.nan)

    return results


def run_full_xie_analysis(
    model,
    task,
    seq_len: int,
    n_trials: int = 200,
    device: str = 'cpu',
    alpha: float = 1.0,
    n_components: int = 2,
) -> Dict:
    """
    Run full Xie et al. analysis on a trained model.

    Args:
        model: Trained model
        task: Task object
        seq_len: Sequence length to analyze
        n_trials: Number of trials
        device: Device
        alpha: Ridge regression regularization
        n_components: Number of PCA components for planes

    Returns:
        results: Complete analysis results
    """
    from analysis import extract_hidden_states, get_timepoint_indices

    # Generate trials
    inputs, targets, mask, infos = task.generate_batch(n_trials, seq_len=seq_len)
    inputs = inputs.to(device)

    # Extract hidden states
    hidden_states = extract_hidden_states(model, inputs, device).numpy()

    # Get sequences
    sequences = np.array([info['sequence'] for info in infos])

    # Get timepoint indices
    indices = get_timepoint_indices(task, seq_len)

    results = {
        'seq_len': seq_len,
        'n_trials': n_trials,
        'n_stimuli': task.n_stimuli,
    }

    # Analysis at delay midpoint
    delay_mid = indices['delay'][len(indices['delay']) // 2] + 1
    delay_states = hidden_states[:, delay_mid, :]

    results['delay_analysis'] = xie_subspace_analysis(
        delay_states, sequences, task.n_stimuli, seq_len,
        alpha=alpha, n_components=n_components
    )

    # Analysis over time (through delay period)
    delay_time_indices = [t + 1 for t in indices['delay']]  # +1 for hidden state indexing

    results['delay_timecourse'] = run_xie_analysis_over_time(
        hidden_states, sequences, task.n_stimuli, seq_len,
        time_indices=delay_time_indices,
        alpha=alpha, n_components=n_components
    )

    # Also analyze at end of each stimulus presentation
    stim_end_indices = [stim[-1] + 1 for stim in indices['stimulus']]

    results['stimulus_offset_analysis'] = {}
    for pos, t in enumerate(stim_end_indices):
        states_t = hidden_states[:, t, :]
        results['stimulus_offset_analysis'][pos] = xie_subspace_analysis(
            states_t, sequences, task.n_stimuli, seq_len,
            alpha=alpha, n_components=n_components
        )

    return results


if __name__ == "__main__":
    # Test the analysis
    import torch
    from task import SequenceWorkingMemoryTask
    from model import create_model
    from train import load_model
    from pathlib import Path

    # Try to load a trained model, or create untrained one for testing
    results_dir = Path("results")
    model_path = None
    model_type = 'vanilla'

    # Find GRU model first, then fall back to vanilla
    if results_dir.exists():
        # First look for GRU models
        for exp_dir in sorted(results_dir.iterdir(), reverse=True):
            if (exp_dir / "final_model.pt").exists():
                checkpoint = torch.load(exp_dir / "final_model.pt", map_location='cpu')
                keys = list(checkpoint['model_state_dict'].keys())
                if any('gru' in k for k in keys):
                    model_path = exp_dir / "final_model.pt"
                    model_type = 'gru'
                    break
                elif any('lstm' in k for k in keys):
                    model_path = exp_dir / "final_model.pt"
                    model_type = 'lstm'
                    break

        # Fall back to vanilla if no GRU/LSTM found
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
        print("No trained model found, using random model for testing")
        model = create_model('gru', task.input_dim, 128, task.output_dim)

    # Run analysis for each sequence length
    for seq_len in [2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"Xie et al. Analysis for seq_len = {seq_len}")
        print('='*60)

        results = run_full_xie_analysis(
            model, task, seq_len,
            n_trials=200,
            device='cpu',
            n_components=2
        )

        # Print delay period results
        delay_results = results['delay_analysis']
        print(f"\nDelay period analysis:")
        print(f"  Design matrix shape: {delay_results['design_matrix_shape']}")
        print(f"  Beta matrix shape: {delay_results['betas_shape']}")

        print(f"\n  PCA explained variance per position:")
        for pos, var in delay_results['pca_explained_variance'].items():
            print(f"    Position {pos+1}: {[f'{v:.1%}' for v in var]}")

        print(f"\n  Principal angles between position subspaces (2D planes):")
        for (i, j), angles in delay_results['principal_angles_deg'].items():
            mean_angle = delay_results['mean_angles_deg'][(i, j)]
            print(f"    Pos {i+1} vs Pos {j+1}: {[f'{a:.1f}°' for a in angles]} (mean: {mean_angle:.1f}°)")

        # Summary
        print(f"\n  Summary:")
        all_means = list(delay_results['mean_angles_deg'].values())
        if all_means:
            print(f"    Overall mean angle: {np.mean(all_means):.1f}°")
            print(f"    (0° = identical planes, 90° = orthogonal planes)")
