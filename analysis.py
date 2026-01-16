"""
Representation geometry analysis for sequence working memory task.

Implements analyses inspired by Xie et al. Science 2022:
- State-space geometry and trajectories
- Representational dissimilarity matrices (RDMs)
- Cross-temporal generalization
- Linear decoders for item and position
- Subspace analysis
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional
import warnings


def extract_hidden_states(
    model: nn.Module,
    inputs: torch.Tensor,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Extract hidden states from model for given inputs.

    Args:
        model: Trained model
        inputs: (batch, seq_len, input_dim) inputs

    Returns:
        hidden_states: (batch, seq_len+1, hidden_dim) hidden states
    """
    model = model.to(device)
    model.eval()
    inputs = inputs.to(device)

    with torch.no_grad():
        _, _ = model(inputs, store_hidden=True)
        hidden_states = model.get_hidden_states()

    return hidden_states.cpu()


def get_timepoint_indices(
    task,
    seq_len: int,
) -> Dict[str, List[int]]:
    """
    Get indices for key timepoints in a trial.

    Args:
        task: Task object
        seq_len: Sequence length

    Returns:
        indices: Dict mapping phase names to timestep indices
    """
    indices = {
        'fixation': [],
        'stimulus': [],  # List of lists, one per item
        'delay': [],
        'response': [],  # List of lists, one per item
    }

    t = 0

    # Fixation
    indices['fixation'] = list(range(t, t + task.fixation_duration))
    t += task.fixation_duration

    # Stimulus presentations
    indices['stimulus'] = []
    for i in range(seq_len):
        indices['stimulus'].append(list(range(t, t + task.stimulus_duration)))
        t += task.stimulus_duration

    # Delay
    indices['delay'] = list(range(t, t + task.delay_duration))
    t += task.delay_duration

    # Response periods
    indices['response'] = []
    for i in range(seq_len):
        indices['response'].append(list(range(t, t + task.response_duration)))
        t += task.response_duration

    return indices


def compute_pca(
    hidden_states: torch.Tensor,
    n_components: int = 3,
) -> Tuple[np.ndarray, PCA]:
    """
    Perform PCA on hidden states.

    Args:
        hidden_states: (batch, seq_len, hidden_dim) or (n_samples, hidden_dim)
        n_components: Number of PCA components

    Returns:
        projected: Projected data
        pca: Fitted PCA object
    """
    # Flatten if 3D
    if hidden_states.ndim == 3:
        batch, seq_len, hidden_dim = hidden_states.shape
        data = hidden_states.numpy().reshape(-1, hidden_dim)
    else:
        data = hidden_states.numpy()

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(data)

    return projected, pca


def compute_rdm(
    activations: np.ndarray,
    metric: str = 'correlation',
) -> np.ndarray:
    """
    Compute representational dissimilarity matrix.

    Args:
        activations: (n_conditions, n_features) activation patterns
        metric: Distance metric ('correlation', 'euclidean', 'cosine')

    Returns:
        rdm: (n_conditions, n_conditions) dissimilarity matrix
    """
    if metric == 'correlation':
        # 1 - correlation
        dists = pdist(activations, metric='correlation')
    elif metric == 'euclidean':
        dists = pdist(activations, metric='euclidean')
    elif metric == 'cosine':
        dists = pdist(activations, metric='cosine')
    else:
        raise ValueError(f"Unknown metric: {metric}")

    rdm = squareform(dists)
    return rdm


def cross_temporal_decoding(
    hidden_states: torch.Tensor,
    labels: np.ndarray,
    time_indices: Optional[List[int]] = None,
    cv: int = 5,
) -> np.ndarray:
    """
    Train decoder at each timepoint, test at all other timepoints.

    Args:
        hidden_states: (batch, seq_len, hidden_dim)
        labels: (batch,) labels to decode
        time_indices: Subset of timepoints to use (all if None)
        cv: Number of cross-validation folds

    Returns:
        generalization_matrix: (n_times, n_times) decoding accuracy matrix
    """
    hidden_states = hidden_states.numpy()
    batch_size, seq_len, hidden_dim = hidden_states.shape

    if time_indices is None:
        time_indices = list(range(seq_len))
    n_times = len(time_indices)

    gen_matrix = np.zeros((n_times, n_times))

    for i, t_train in enumerate(time_indices):
        X_train = hidden_states[:, t_train, :]

        # Fit classifier
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')

        # Train on this timepoint
        try:
            clf.fit(X_train, labels)

            # Test on all timepoints
            for j, t_test in enumerate(time_indices):
                X_test = hidden_states[:, t_test, :]
                gen_matrix[i, j] = clf.score(X_test, labels)
        except Exception:
            gen_matrix[i, :] = np.nan

    return gen_matrix


def decode_item_and_position(
    hidden_states: torch.Tensor,
    infos: List[dict],
    task,
    seq_len: int,
) -> Dict[str, Dict]:
    """
    Train linear decoders for item identity and serial position.

    Args:
        hidden_states: (batch, seq_len+1, hidden_dim)
        infos: List of trial info dicts
        task: Task object
        seq_len: Sequence length

    Returns:
        results: Dict with decoding accuracies for item and position
    """
    hidden_states = hidden_states.numpy()
    batch_size = hidden_states.shape[0]

    indices = get_timepoint_indices(task, seq_len)

    results = {
        'item_at_position': {},  # Decode item identity at each position
        'position_of_item': {},  # Decode position for each item
        'delay_timecourse': {},  # Decoding accuracy throughout delay
    }

    # Extract sequences
    sequences = np.array([info['sequence'] for info in infos])

    # Decode item at each position (at stimulus offset)
    for pos in range(seq_len):
        # Labels: which item is at this position
        labels = sequences[:, pos]

        # Get hidden state at end of this item's presentation
        stim_end = indices['stimulus'][pos][-1] + 1  # +1 for hidden state indexing
        X = hidden_states[:, stim_end, :]

        try:
            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            scores = cross_val_score(clf, X, labels, cv=5)
            results['item_at_position'][pos] = {
                'accuracy': scores.mean(),
                'std': scores.std(),
            }
        except Exception as e:
            results['item_at_position'][pos] = {'accuracy': np.nan, 'std': np.nan}

    # Decode at delay midpoint (for backward compatibility)
    delay_mid = indices['delay'][len(indices['delay'])//2] + 1
    X_delay = hidden_states[:, delay_mid, :]

    for pos in range(seq_len):
        labels = sequences[:, pos]
        try:
            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            scores = cross_val_score(clf, X_delay, labels, cv=5)
            results['item_at_position'][f'delay_pos{pos}'] = {
                'accuracy': scores.mean(),
                'std': scores.std(),
            }
        except Exception:
            results['item_at_position'][f'delay_pos{pos}'] = {'accuracy': np.nan, 'std': np.nan}

    # Decode throughout delay period for each position
    delay_indices = indices['delay']
    for pos in range(seq_len):
        labels = sequences[:, pos]
        timecourse = []

        for t_idx in delay_indices:
            t = t_idx + 1  # +1 for hidden state indexing
            X = hidden_states[:, t, :]
            try:
                clf = LogisticRegression(max_iter=1000, solver='lbfgs')
                scores = cross_val_score(clf, X, labels, cv=3)  # Use cv=3 for speed
                timecourse.append(scores.mean())
            except Exception:
                timecourse.append(np.nan)

        results['delay_timecourse'][pos] = timecourse

    return results


def subspace_analysis(
    hidden_states: torch.Tensor,
    infos: List[dict],
    task,
    seq_len: int,
    n_components: int = 10,
) -> Dict:
    """
    Analyze subspaces for different sequence positions.

    Computes:
    - Principal angles between position subspaces
    - Variance explained by position-specific vs shared subspaces

    Args:
        hidden_states: (batch, seq_len+1, hidden_dim)
        infos: List of trial info dicts
        task: Task object
        seq_len: Sequence length
        n_components: Number of PCA components per subspace

    Returns:
        results: Dict with subspace analysis results
    """
    hidden_states = hidden_states.numpy()
    batch_size = hidden_states.shape[0]

    indices = get_timepoint_indices(task, seq_len)
    sequences = np.array([info['sequence'] for info in infos])

    results = {
        'principal_angles': {},
        'subspaces': {},
        'delay_subspace_angles': {},  # Angles during delay period
    }

    # Get hidden states at end of each stimulus presentation
    position_states = []
    for pos in range(seq_len):
        stim_end = indices['stimulus'][pos][-1] + 1
        position_states.append(hidden_states[:, stim_end, :])

    # Compute PCA for each position
    position_pcas = []
    for pos, states in enumerate(position_states):
        pca = PCA(n_components=min(n_components, states.shape[1], states.shape[0]))
        pca.fit(states)
        position_pcas.append(pca)
        results['subspaces'][pos] = {
            'explained_variance': pca.explained_variance_ratio_.tolist(),
        }

    # Compute principal angles between position subspaces (at stimulus offset)
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            # Get principal components
            U_i = position_pcas[i].components_.T  # (hidden_dim, n_components)
            U_j = position_pcas[j].components_.T

            # Compute principal angles via SVD
            M = U_i.T @ U_j
            _, s, _ = np.linalg.svd(M)
            angles = np.arccos(np.clip(s, -1, 1))

            results['principal_angles'][(i, j)] = {
                'angles_rad': angles.tolist(),
                'angles_deg': np.degrees(angles).tolist(),
            }

    # Analyze subspaces during delay period
    # Group trials by item identity at each position, compute mean representation
    delay_mid = indices['delay'][len(indices['delay'])//2] + 1
    delay_states = hidden_states[:, delay_mid, :]

    # For each position, find the subspace that encodes item identity
    delay_position_subspaces = []
    for pos in range(seq_len):
        # Group by item at this position
        item_means = []
        for item in range(task.n_stimuli):
            mask = sequences[:, pos] == item
            if mask.sum() > 0:
                item_means.append(delay_states[mask].mean(axis=0))

        if len(item_means) > 1:
            item_means = np.array(item_means)
            # PCA on item-conditioned means gives the subspace encoding this position's item
            pca = PCA(n_components=min(n_components, len(item_means)-1, item_means.shape[1]))
            pca.fit(item_means)
            delay_position_subspaces.append(pca)
        else:
            delay_position_subspaces.append(None)

    # Compute principal angles between delay subspaces
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            if delay_position_subspaces[i] is not None and delay_position_subspaces[j] is not None:
                U_i = delay_position_subspaces[i].components_.T
                U_j = delay_position_subspaces[j].components_.T

                # Match dimensions
                min_dim = min(U_i.shape[1], U_j.shape[1])
                U_i = U_i[:, :min_dim]
                U_j = U_j[:, :min_dim]

                M = U_i.T @ U_j
                _, s, _ = np.linalg.svd(M)
                angles = np.arccos(np.clip(s, -1, 1))

                results['delay_subspace_angles'][(i, j)] = {
                    'angles_rad': angles.tolist(),
                    'angles_deg': np.degrees(angles).tolist(),
                }

    return results


def compute_geometry_metrics(
    hidden_states: torch.Tensor,
    infos: List[dict],
    task,
    seq_len: int,
) -> Dict:
    """
    Compute various geometry metrics for the representation.

    Args:
        hidden_states: (batch, seq_len+1, hidden_dim)
        infos: List of trial info dicts
        task: Task object
        seq_len: Sequence length

    Returns:
        metrics: Dict with geometry metrics
    """
    hidden_states_np = hidden_states.numpy()
    batch_size, total_time, hidden_dim = hidden_states_np.shape

    indices = get_timepoint_indices(task, seq_len)
    sequences = np.array([info['sequence'] for info in infos])

    metrics = {}

    # 1. Dimensionality (participation ratio)
    delay_mid = indices['delay'][len(indices['delay'])//2] + 1
    delay_states = hidden_states_np[:, delay_mid, :]
    cov = np.cov(delay_states.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]
    participation_ratio = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    metrics['dimensionality'] = float(participation_ratio)

    # 2. Distance between sequence representations
    # Group by unique sequences and compute mean representation
    unique_seqs = {}
    for i, seq in enumerate(sequences):
        seq_tuple = tuple(seq)
        if seq_tuple not in unique_seqs:
            unique_seqs[seq_tuple] = []
        unique_seqs[seq_tuple].append(delay_states[i])

    if len(unique_seqs) > 1:
        seq_means = np.array([np.mean(states, axis=0) for states in unique_seqs.values()])
        seq_rdm = compute_rdm(seq_means, metric='euclidean')
        metrics['mean_seq_distance'] = float(seq_rdm[np.triu_indices_from(seq_rdm, k=1)].mean())
    else:
        metrics['mean_seq_distance'] = np.nan

    # 3. Item vs position coding strength
    # Compute RDM and correlate with model RDMs
    if len(unique_seqs) > 1:
        # Create model RDMs for item identity and position
        seq_list = list(unique_seqs.keys())
        n_seqs = len(seq_list)

        item_model = np.zeros((n_seqs, n_seqs))
        position_model = np.zeros((n_seqs, n_seqs))

        for i in range(n_seqs):
            for j in range(n_seqs):
                # Item similarity: count shared items
                shared_items = len(set(seq_list[i]) & set(seq_list[j]))
                item_model[i, j] = seq_len - shared_items  # dissimilarity

                # Position similarity: count items in same position
                same_pos = sum(seq_list[i][k] == seq_list[j][k] for k in range(min(len(seq_list[i]), len(seq_list[j]))))
                position_model[i, j] = seq_len - same_pos  # dissimilarity

        # Correlate neural RDM with model RDMs
        neural_rdm = seq_rdm
        neural_upper = neural_rdm[np.triu_indices_from(neural_rdm, k=1)]
        item_upper = item_model[np.triu_indices_from(item_model, k=1)]
        position_upper = position_model[np.triu_indices_from(position_model, k=1)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if np.std(item_upper) > 0:
                item_corr, _ = spearmanr(neural_upper, item_upper)
            else:
                item_corr = np.nan
            if np.std(position_upper) > 0:
                pos_corr, _ = spearmanr(neural_upper, position_upper)
            else:
                pos_corr = np.nan

        metrics['item_rdm_correlation'] = float(item_corr) if not np.isnan(item_corr) else np.nan
        metrics['position_rdm_correlation'] = float(pos_corr) if not np.isnan(pos_corr) else np.nan

    return metrics


def analyze_dynamics(
    hidden_states: torch.Tensor,
    task,
    seq_len: int,
    n_components: int = 3,
) -> Dict:
    """
    Analyze dynamics of hidden state trajectories.

    Args:
        hidden_states: (batch, seq_len+1, hidden_dim)
        task: Task object
        seq_len: Sequence length
        n_components: Number of PCA components for visualization

    Returns:
        dynamics: Dict with trajectory analysis results
    """
    hidden_states_np = hidden_states.numpy()
    batch_size, total_time, hidden_dim = hidden_states_np.shape

    indices = get_timepoint_indices(task, seq_len)

    # Fit PCA on all states
    all_states = hidden_states_np.reshape(-1, hidden_dim)
    pca = PCA(n_components=n_components)
    pca.fit(all_states)

    dynamics = {
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
        'pca_components': pca.components_.tolist(),
    }

    # Compute mean trajectory and project
    mean_trajectory = hidden_states_np.mean(axis=0)  # (total_time, hidden_dim)
    projected_trajectory = pca.transform(mean_trajectory)
    dynamics['mean_trajectory_pca'] = projected_trajectory.tolist()

    # Compute velocity (rate of change) in state space
    velocities = np.diff(mean_trajectory, axis=0)
    speed = np.linalg.norm(velocities, axis=1)
    dynamics['trajectory_speed'] = speed.tolist()

    # Phase labels
    phase_labels = []
    for t in range(total_time):
        if t in indices['fixation']:
            phase_labels.append('fixation')
        elif any(t in stim for stim in indices['stimulus']):
            phase_labels.append('stimulus')
        elif t in indices['delay']:
            phase_labels.append('delay')
        elif any(t in resp for resp in indices['response']):
            phase_labels.append('response')
        else:
            phase_labels.append('unknown')
    dynamics['phase_labels'] = phase_labels

    return dynamics


def run_all_analyses(
    model: nn.Module,
    task,
    n_trials: int = 200,
    device: str = 'cpu',
) -> Dict:
    """
    Run all representation analyses.

    Args:
        model: Trained model
        task: Task object
        n_trials: Number of trials to analyze
        device: Device to use

    Returns:
        results: Dict with all analysis results
    """
    results = {}

    for seq_len in task.seq_lengths:
        print(f"Analyzing seq_len={seq_len}...")

        # Generate trials
        inputs, targets, mask, infos = task.generate_batch(n_trials, seq_len=seq_len)
        inputs = inputs.to(device)

        # Extract hidden states
        hidden_states = extract_hidden_states(model, inputs, device)

        # Run analyses
        seq_results = {}

        # Decoding
        seq_results['decoding'] = decode_item_and_position(
            hidden_states, infos, task, seq_len
        )

        # Subspace analysis
        seq_results['subspace'] = subspace_analysis(
            hidden_states, infos, task, seq_len
        )

        # Geometry metrics
        seq_results['geometry'] = compute_geometry_metrics(
            hidden_states, infos, task, seq_len
        )

        # Dynamics
        seq_results['dynamics'] = analyze_dynamics(
            hidden_states, task, seq_len
        )

        results[seq_len] = seq_results

    return results


if __name__ == "__main__":
    # Test analysis functions
    import torch
    from task import SequenceWorkingMemoryTask
    from model import VanillaRNN

    # Create task and model
    task = SequenceWorkingMemoryTask(n_stimuli=8, seq_lengths=[2, 3])
    model = VanillaRNN(
        input_dim=task.input_dim,
        hidden_dim=64,
        output_dim=task.output_dim,
    )

    # Generate test data
    inputs, targets, mask, infos = task.generate_batch(50, seq_len=2)

    # Extract hidden states
    hidden_states = extract_hidden_states(model, inputs)
    print(f"Hidden states shape: {hidden_states.shape}")

    # Test PCA
    projected, pca = compute_pca(hidden_states)
    print(f"PCA projected shape: {projected.shape}")
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # Test decoding
    decoding_results = decode_item_and_position(hidden_states, infos, task, seq_len=2)
    print(f"Decoding results: {decoding_results}")

    # Test subspace analysis
    subspace_results = subspace_analysis(hidden_states, infos, task, seq_len=2)
    print(f"Subspace results keys: {subspace_results.keys()}")

    # Test geometry metrics
    geometry = compute_geometry_metrics(hidden_states, infos, task, seq_len=2)
    print(f"Geometry metrics: {geometry}")

    # Test dynamics
    dynamics = analyze_dynamics(hidden_states, task, seq_len=2)
    print(f"Dynamics keys: {dynamics.keys()}")
