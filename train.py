"""
Training utilities for sequence working memory task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import json
from tqdm import tqdm

from task import SequenceWorkingMemoryTask, CurriculumScheduler
from model import VanillaRNN, create_model


def compute_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Compute accuracy on response timesteps.

    Args:
        outputs: (batch, seq_len, n_classes) model outputs (logits)
        targets: (batch, seq_len) target indices
        mask: (batch, seq_len) binary mask for response timesteps

    Returns:
        accuracy: Fraction correct on response timesteps
    """
    predictions = outputs.argmax(dim=-1)  # (batch, seq_len)
    correct = (predictions == targets).float() * mask
    accuracy = correct.sum() / mask.sum()
    return accuracy.item()


def compute_sequence_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    infos: List[dict],
) -> float:
    """
    Compute whole-sequence accuracy (all items correct).

    Args:
        outputs: (batch, seq_len, n_classes) model outputs
        targets: (batch, seq_len) target indices
        mask: (batch, seq_len) binary mask
        infos: List of trial info dicts

    Returns:
        accuracy: Fraction of sequences entirely correct
    """
    predictions = outputs.argmax(dim=-1)
    correct = (predictions == targets).float() * mask

    # Check if entire sequence is correct for each trial
    batch_size = outputs.shape[0]
    n_correct = 0
    for i in range(batch_size):
        trial_mask = mask[i] > 0.5  # Boolean mask
        masked_correct = correct[i][trial_mask]
        if masked_correct.sum().item() == masked_correct.numel():
            n_correct += 1

    return n_correct / batch_size


def train_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Perform one training step.

    Args:
        model: The model
        inputs: (batch, seq_len, input_dim)
        targets: (batch, seq_len)
        mask: (batch, seq_len)
        optimizer: Optimizer
        criterion: Loss function

    Returns:
        loss: Training loss
        accuracy: Training accuracy
    """
    model.train()
    optimizer.zero_grad()

    outputs, _ = model(inputs, store_hidden=False)

    # Compute loss only on response timesteps
    # Reshape for cross entropy: (batch * seq_len, n_classes) vs (batch * seq_len,)
    batch_size, seq_len, n_classes = outputs.shape
    outputs_flat = outputs.view(-1, n_classes)
    targets_flat = targets.view(-1)
    mask_flat = mask.view(-1)

    # Compute weighted loss
    loss_per_timestep = criterion(outputs_flat, targets_flat)
    loss = (loss_per_timestep * mask_flat).sum() / mask_flat.sum()

    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    accuracy = compute_accuracy(outputs, targets, mask)

    return loss.item(), accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    infos: List[dict],
    criterion: nn.Module,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: The model
        inputs: (batch, seq_len, input_dim)
        targets: (batch, seq_len)
        mask: (batch, seq_len)
        infos: List of trial info dicts
        criterion: Loss function

    Returns:
        metrics: Dict with loss, item_accuracy, sequence_accuracy
    """
    model.eval()

    outputs, _ = model(inputs, store_hidden=False)

    # Compute loss
    batch_size, seq_len, n_classes = outputs.shape
    outputs_flat = outputs.view(-1, n_classes)
    targets_flat = targets.view(-1)
    mask_flat = mask.view(-1)

    loss_per_timestep = criterion(outputs_flat, targets_flat)
    loss = (loss_per_timestep * mask_flat).sum() / mask_flat.sum()

    item_acc = compute_accuracy(outputs, targets, mask)
    seq_acc = compute_sequence_accuracy(outputs, targets, mask, infos)

    return {
        'loss': loss.item(),
        'item_accuracy': item_acc,
        'sequence_accuracy': seq_acc,
    }


def train_model(
    model: nn.Module,
    task: SequenceWorkingMemoryTask,
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    use_curriculum: bool = True,
    curriculum_mix_ratio: float = 0.3,
    curriculum_threshold: float = 0.9,
    eval_every: int = 10,
    save_dir: Optional[str] = None,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict:
    """
    Train the model.

    Args:
        model: The model to train
        task: The task object
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        use_curriculum: Whether to use curriculum learning
        curriculum_mix_ratio: Fraction of batch from previous seq lengths (prevents forgetting)
        curriculum_threshold: Accuracy threshold to advance curriculum
        eval_every: Evaluate every N epochs
        save_dir: Directory to save checkpoints
        device: Device to train on
        verbose: Print progress

    Returns:
        history: Training history dict
    """
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    # Curriculum scheduler
    if use_curriculum:
        scheduler = CurriculumScheduler(
            task,
            start_seq_len=min(task.seq_lengths),
            accuracy_threshold=curriculum_threshold,
        )
    else:
        scheduler = None

    # History
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_metrics': {sl: [] for sl in task.seq_lengths},
        'epochs': [],
        'seq_len_schedule': [],
    }

    # Create save directory
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    epoch_iter = tqdm(range(n_epochs), desc='Training') if verbose else range(n_epochs)

    for epoch in epoch_iter:
        # Determine sequence length for this epoch
        if scheduler:
            seq_len = scheduler.get_seq_len()
        else:
            seq_len = np.random.choice(task.seq_lengths)

        history['seq_len_schedule'].append(seq_len)

        # Mixed curriculum: train on current level + previous levels to prevent forgetting
        epoch_losses = []
        epoch_accs = []

        if scheduler and curriculum_mix_ratio > 0:
            # Get all sequence lengths up to current
            min_sl = min(task.seq_lengths)
            active_seq_lengths = [sl for sl in task.seq_lengths if sl <= seq_len]

            # Distribute batch across sequence lengths
            # Current level gets (1 - mix_ratio), previous levels share mix_ratio
            if len(active_seq_lengths) > 1:
                n_current = int(batch_size * (1 - curriculum_mix_ratio))
                n_previous = batch_size - n_current
                n_per_previous = max(1, n_previous // (len(active_seq_lengths) - 1))
            else:
                n_current = batch_size
                n_per_previous = 0

            # Train on current level
            inputs, targets, mask, infos = task.generate_batch(n_current, seq_len=seq_len)
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            loss, accuracy = train_step(model, inputs, targets, mask, optimizer, criterion)
            epoch_losses.append(loss)
            epoch_accs.append(accuracy)

            # Train on previous levels (replay to prevent forgetting)
            for prev_sl in active_seq_lengths[:-1]:
                inputs, targets, mask, infos = task.generate_batch(n_per_previous, seq_len=prev_sl)
                inputs = inputs.to(device)
                targets = targets.to(device)
                mask = mask.to(device)
                loss, accuracy = train_step(model, inputs, targets, mask, optimizer, criterion)
                epoch_losses.append(loss)
                epoch_accs.append(accuracy)
        else:
            # Standard training on single sequence length
            inputs, targets, mask, infos = task.generate_batch(batch_size, seq_len=seq_len)
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            loss, accuracy = train_step(model, inputs, targets, mask, optimizer, criterion)
            epoch_losses.append(loss)
            epoch_accs.append(accuracy)

        # Record average loss/accuracy for this epoch
        history['train_loss'].append(np.mean(epoch_losses))
        history['train_accuracy'].append(np.mean(epoch_accs))

        # Evaluation
        if (epoch + 1) % eval_every == 0:
            history['epochs'].append(epoch + 1)

            # Evaluate on each sequence length
            for sl in task.seq_lengths:
                val_inputs, val_targets, val_mask, val_infos = task.generate_batch(
                    batch_size, seq_len=sl
                )
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                val_mask = val_mask.to(device)

                metrics = evaluate(model, val_inputs, val_targets, val_mask, val_infos, criterion)
                history['val_metrics'][sl].append(metrics)

            # Update curriculum
            if scheduler:
                current_sl_metrics = history['val_metrics'][seq_len][-1]
                advanced = scheduler.update(current_sl_metrics['item_accuracy'])
                if advanced and verbose:
                    tqdm.write(f"  Advancing to seq_len={scheduler.get_seq_len()}")

            if verbose:
                val_accs = {sl: history['val_metrics'][sl][-1]['item_accuracy']
                           for sl in task.seq_lengths}
                tqdm.write(f"  Epoch {epoch+1}: train_loss={loss:.4f}, val_acc={val_accs}")

    # Save final model
    if save_dir:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'task_config': {
                'n_stimuli': task.n_stimuli,
                'seq_lengths': task.seq_lengths,
            },
        }, save_path / 'final_model.pt')

        # Save history as JSON
        with open(save_path / 'history.json', 'w') as f:
            # Convert numpy types for JSON serialization
            json_history = {
                'train_loss': [float(x) for x in history['train_loss']],
                'train_accuracy': [float(x) for x in history['train_accuracy']],
                'epochs': [int(x) for x in history['epochs']],
                'seq_len_schedule': [int(x) for x in history['seq_len_schedule']],
                'val_metrics': {
                    str(k): [{kk: float(vv) for kk, vv in m.items()} for m in v]
                    for k, v in history['val_metrics'].items()
                },
            }
            json.dump(json_history, f, indent=2)

    return history


def load_model(
    checkpoint_path: str,
    model_type: str = 'vanilla',
    device: str = 'cpu',
) -> Tuple[nn.Module, Dict]:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Type of model ('vanilla', 'lstm', 'gru')
        device: Device to load to

    Returns:
        model: Loaded model
        checkpoint: Full checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Infer model dimensions from state dict
    state_dict = checkpoint['model_state_dict']
    if model_type == 'vanilla':
        input_dim = state_dict['W_in.weight'].shape[1]
        hidden_dim = state_dict['W_in.weight'].shape[0]
        output_dim = state_dict['W_out.weight'].shape[0]
    elif model_type == 'gru':
        # GRU weight_ih_l0 has shape (3*hidden_dim, input_dim)
        hidden_dim = state_dict['gru.weight_ih_l0'].shape[0] // 3
        input_dim = state_dict['gru.weight_ih_l0'].shape[1]
        output_dim = state_dict['W_out.weight'].shape[0]
    elif model_type == 'lstm':
        # LSTM weight_ih_l0 has shape (4*hidden_dim, input_dim)
        hidden_dim = state_dict['lstm.weight_ih_l0'].shape[0] // 4
        input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
        output_dim = state_dict['W_out.weight'].shape[0]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = create_model(model_type, input_dim, hidden_dim, output_dim)
    model.load_state_dict(state_dict)
    model = model.to(device)

    return model, checkpoint


if __name__ == "__main__":
    # Quick training test
    from task import SequenceWorkingMemoryTask

    # Create task and model
    task = SequenceWorkingMemoryTask(n_stimuli=8, seq_lengths=[2, 3, 4])
    model = VanillaRNN(
        input_dim=task.input_dim,
        hidden_dim=128,
        output_dim=task.output_dim,
    )

    print(f"Task input_dim: {task.input_dim}")
    print(f"Task output_dim: {task.output_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Short training test
    history = train_model(
        model,
        task,
        n_epochs=50,
        batch_size=32,
        use_curriculum=True,
        eval_every=10,
        verbose=True,
    )

    print("\nFinal validation accuracies:")
    for sl in task.seq_lengths:
        if history['val_metrics'][sl]:
            acc = history['val_metrics'][sl][-1]['item_accuracy']
            print(f"  seq_len={sl}: {acc:.2%}")
