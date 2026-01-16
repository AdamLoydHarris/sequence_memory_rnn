"""
Task generation for sequence working memory task.

Inspired by Xie et al. Science 2022 - trains RNNs to remember and recall
sequences of stimuli in order.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


class SequenceWorkingMemoryTask:
    """
    Sequence working memory task.

    Trial structure:
    1. Fixation period
    2. Sequential stimulus presentation (2-4 items)
    3. Delay period (maintain sequence)
    4. Response period (output sequence in order)
    """

    def __init__(
        self,
        n_stimuli: int = 8,
        seq_lengths: List[int] = [2, 3, 4],
        fixation_duration: int = 5,
        stimulus_duration: int = 5,
        delay_duration: int = 10,
        response_duration: int = 5,
        dt: float = 0.1,
    ):
        """
        Args:
            n_stimuli: Number of distinct stimuli
            seq_lengths: Possible sequence lengths
            fixation_duration: Duration of fixation in timesteps
            stimulus_duration: Duration of each stimulus in timesteps
            delay_duration: Duration of delay period in timesteps
            response_duration: Duration for each response in timesteps
            dt: Timestep size (for potential continuous dynamics)
        """
        self.n_stimuli = n_stimuli
        self.seq_lengths = seq_lengths
        self.fixation_duration = fixation_duration
        self.stimulus_duration = stimulus_duration
        self.delay_duration = delay_duration
        self.response_duration = response_duration
        self.dt = dt

        # Input dimension: one-hot stimuli + timing signals
        # Timing signals: fixation, stimulus_on, delay, response cue for each position
        self.max_seq_len = max(seq_lengths)
        self.n_timing_signals = 3 + self.max_seq_len  # fixation, stim_on, delay, response_cues
        self.input_dim = n_stimuli + self.n_timing_signals

        # Output dimension: n_stimuli (predict which item at each response)
        self.output_dim = n_stimuli

    def _get_trial_length(self, seq_len: int) -> int:
        """Calculate total trial length for a given sequence length."""
        return (
            self.fixation_duration +
            seq_len * self.stimulus_duration +
            self.delay_duration +
            seq_len * self.response_duration
        )

    def generate_trial(
        self,
        seq_len: Optional[int] = None,
        sequence: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Generate a single trial.

        Args:
            seq_len: Sequence length (randomly sampled if None)
            sequence: Specific sequence to use (randomly generated if None)

        Returns:
            inputs: (trial_length, input_dim) tensor
            targets: (trial_length,) tensor of target indices (-1 for no response)
            mask: (trial_length,) tensor indicating when to compute loss
            info: dict with trial information
        """
        # Sample sequence length if not specified
        if seq_len is None:
            seq_len = np.random.choice(self.seq_lengths)

        # Generate random sequence if not specified
        if sequence is None:
            sequence = np.random.choice(self.n_stimuli, size=seq_len, replace=False).tolist()

        trial_length = self._get_trial_length(seq_len)

        # Initialize tensors
        inputs = torch.zeros(trial_length, self.input_dim)
        targets = torch.full((trial_length,), -1, dtype=torch.long)  # -1 = no target
        mask = torch.zeros(trial_length)

        t = 0

        # Fixation period
        inputs[t:t+self.fixation_duration, self.n_stimuli] = 1.0  # fixation signal
        t += self.fixation_duration

        # Stimulus presentation period
        for i, stim in enumerate(sequence):
            inputs[t:t+self.stimulus_duration, stim] = 1.0  # one-hot stimulus
            inputs[t:t+self.stimulus_duration, self.n_stimuli + 1] = 1.0  # stimulus_on signal
            t += self.stimulus_duration

        # Delay period
        inputs[t:t+self.delay_duration, self.n_stimuli + 2] = 1.0  # delay signal
        t += self.delay_duration

        # Response period
        for i, stim in enumerate(sequence):
            # Response cue for position i
            inputs[t:t+self.response_duration, self.n_stimuli + 3 + i] = 1.0
            # Target is the item at this position
            targets[t:t+self.response_duration] = stim
            mask[t:t+self.response_duration] = 1.0
            t += self.response_duration

        info = {
            'sequence': sequence,
            'seq_len': seq_len,
            'trial_length': trial_length,
        }

        return inputs, targets, mask, info

    def generate_batch(
        self,
        batch_size: int,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
        """
        Generate a batch of trials.

        Note: All trials in batch must have same length, so seq_len must be specified
        or all trials will use a randomly sampled (but consistent) length.

        Args:
            batch_size: Number of trials
            seq_len: Sequence length (sampled once for batch if None)

        Returns:
            inputs: (batch_size, trial_length, input_dim)
            targets: (batch_size, trial_length)
            mask: (batch_size, trial_length)
            infos: List of trial info dicts
        """
        if seq_len is None:
            seq_len = np.random.choice(self.seq_lengths)

        trial_length = self._get_trial_length(seq_len)

        inputs = torch.zeros(batch_size, trial_length, self.input_dim)
        targets = torch.zeros(batch_size, trial_length, dtype=torch.long)
        mask = torch.zeros(batch_size, trial_length)
        infos = []

        for i in range(batch_size):
            inp, tgt, msk, info = self.generate_trial(seq_len=seq_len)
            inputs[i] = inp
            targets[i] = tgt
            mask[i] = msk
            infos.append(info)

        return inputs, targets, mask, infos


class CurriculumScheduler:
    """
    Curriculum learning scheduler for gradually increasing sequence length.
    """

    def __init__(
        self,
        task: SequenceWorkingMemoryTask,
        start_seq_len: int = 2,
        accuracy_threshold: float = 0.9,
        patience: int = 5,
    ):
        """
        Args:
            task: The task object
            start_seq_len: Starting sequence length
            accuracy_threshold: Accuracy needed to progress
            patience: Number of evaluations above threshold before progressing
        """
        self.task = task
        self.current_seq_len = start_seq_len
        self.accuracy_threshold = accuracy_threshold
        self.patience = patience
        self.success_count = 0
        self.max_seq_len = max(task.seq_lengths)

    def get_seq_len(self) -> int:
        """Get current sequence length for training."""
        return self.current_seq_len

    def update(self, accuracy: float) -> bool:
        """
        Update curriculum based on accuracy.

        Args:
            accuracy: Current validation accuracy

        Returns:
            advanced: Whether we advanced to next level
        """
        if accuracy >= self.accuracy_threshold:
            self.success_count += 1
            if self.success_count >= self.patience:
                if self.current_seq_len < self.max_seq_len:
                    self.current_seq_len += 1
                    self.success_count = 0
                    return True
        else:
            self.success_count = 0
        return False


def create_datasets(
    task: SequenceWorkingMemoryTask,
    n_train: int = 1000,
    n_val: int = 200,
    n_test: int = 200,
) -> dict:
    """
    Create train/val/test datasets for each sequence length.

    Args:
        task: Task object
        n_train: Number of training trials per sequence length
        n_val: Number of validation trials per sequence length
        n_test: Number of test trials per sequence length

    Returns:
        datasets: Dict with 'train', 'val', 'test' keys, each containing
                  dict mapping seq_len to (inputs, targets, mask, infos)
    """
    datasets = {'train': {}, 'val': {}, 'test': {}}

    for seq_len in task.seq_lengths:
        datasets['train'][seq_len] = task.generate_batch(n_train, seq_len=seq_len)
        datasets['val'][seq_len] = task.generate_batch(n_val, seq_len=seq_len)
        datasets['test'][seq_len] = task.generate_batch(n_test, seq_len=seq_len)

    return datasets


if __name__ == "__main__":
    # Test the task
    task = SequenceWorkingMemoryTask(n_stimuli=8, seq_lengths=[2, 3, 4])

    print(f"Input dimension: {task.input_dim}")
    print(f"Output dimension: {task.output_dim}")

    # Generate a single trial
    inputs, targets, mask, info = task.generate_trial(seq_len=3)
    print(f"\nTrial with sequence {info['sequence']}:")
    print(f"  Trial length: {info['trial_length']}")
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Response timesteps: {mask.sum().item()}")

    # Generate a batch
    inputs, targets, mask, infos = task.generate_batch(32, seq_len=2)
    print(f"\nBatch of 32 trials (seq_len=2):")
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Targets shape: {targets.shape}")
