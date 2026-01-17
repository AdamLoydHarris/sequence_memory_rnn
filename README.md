# RNN Sequence Working Memory

A PyTorch implementation of recurrent neural networks trained on a sequence working memory task, with comprehensive representation geometry analysis. Inspired by [Xie et al. (2022) Science](https://www.science.org/doi/10.1126/science.abm3030).

## Overview

This project trains RNNs to remember and recall sequences of items (2-4 items), then analyzes how the network represents sequence information in its hidden state dynamics. The analysis focuses on:

- **State-space geometry**: PCA trajectories through stimulus, delay, and response periods
- **Linear decoding**: Item identity decoding at each sequence position throughout the delay
- **Subspace analysis**: Whether different sequence positions occupy orthogonal subspaces
- **Representational similarity**: RDMs for item and position coding

## Installation

```bash
# Clone the repository
git clone https://github.com/AdamLoydHarris/RNN_abstraction.git
cd RNN_abstraction

# Install dependencies
pip install torch numpy matplotlib seaborn scikit-learn tqdm
```

## Quick Start

```bash
# Run a full experiment (train + analyze)
python run_experiment.py --model_type gru --hidden_dim 128 --n_epochs 2000 --learning_rate 0.01

# Analyze a pre-trained model
python run_experiment.py --analyze_only results/exp_XXXXXXXX/final_model.pt --model_type gru
```

## Project Structure

```
RNN_abstraction/
├── task.py           # Sequence working memory task generation
├── model.py          # RNN architectures (Vanilla, GRU, LSTM)
├── train.py          # Training loop with curriculum learning
├── analysis.py       # Representation geometry analysis
├── utils.py          # Plotting and helper functions
└── run_experiment.py # Main entry point
```

## Task Design

The sequence working memory task consists of:

1. **Fixation** (5 timesteps): Network receives fixation signal
2. **Stimulus presentation** (5 timesteps each): 2-4 items presented sequentially
3. **Delay period** (10 timesteps): Network must maintain sequence in memory
4. **Response period** (5 timesteps per item): Network outputs items in order

```
Trial structure (seq_len=3):
[FIX][STIM1][STIM2][STIM3][---DELAY---][RESP1][RESP2][RESP3]
```

Stimuli are one-hot encoded (default: 8 unique items). The network receives timing cues indicating when to respond.

## Models

Three RNN architectures are supported:

- **Vanilla RNN**: Simple tanh nonlinearity, good for interpretable dynamics
- **GRU**: Gated recurrent unit, easier to train, single hidden state for analysis
- **LSTM**: Long short-term memory, separate hidden and cell states

```python
from model import create_model

model = create_model('gru', input_dim=15, hidden_dim=128, output_dim=8)
```

## Training

Training uses curriculum learning with mixed replay to prevent catastrophic forgetting:

```python
from task import SequenceWorkingMemoryTask
from model import create_model
from train import train_model

task = SequenceWorkingMemoryTask(n_stimuli=8, seq_lengths=[2, 3, 4])
model = create_model('gru', task.input_dim, 128, task.output_dim)

history = train_model(
    model, task,
    n_epochs=2000,
    learning_rate=0.01,
    use_curriculum=True,
    curriculum_mix_ratio=0.3,  # Replay previous lengths to prevent forgetting
    curriculum_threshold=0.9,   # Advance when accuracy exceeds this
)
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_epochs` | 500 | Number of training epochs |
| `learning_rate` | 1e-3 | Adam learning rate (use 0.01 for GRU) |
| `use_curriculum` | True | Start with short sequences, progress to longer |
| `curriculum_mix_ratio` | 0.3 | Fraction of batch from previous sequence lengths |
| `curriculum_threshold` | 0.9 | Accuracy threshold to advance curriculum |

## Analysis

The analysis module provides several geometry-focused analyses:

### 1. PCA Trajectories

Visualize how hidden states evolve through the trial, with markers for stimulus presentations.

```python
from analysis import extract_hidden_states
from utils import plot_pca_trajectories

hidden_states = extract_hidden_states(model, inputs, device)
plot_pca_trajectories(hidden_states, infos, task, seq_len=3)
```

### 2. Linear Decoding

Decode item identity at each sequence position throughout the delay period.

```python
from analysis import decode_item_and_position

results = decode_item_and_position(hidden_states, infos, task, seq_len=3)
# results['delay_timecourse'][pos] = accuracy at each delay timestep
```

### 3. Subspace Analysis

Measure whether different sequence positions are encoded in orthogonal subspaces.

```python
from analysis import subspace_analysis

results = subspace_analysis(hidden_states, infos, task, seq_len=3)
# results['delay_subspace_angles'] = principal angles between position subspaces
```

### 4. Geometry Metrics

Compute dimensionality (participation ratio) and representational dissimilarity.

```python
from analysis import compute_geometry_metrics

metrics = compute_geometry_metrics(hidden_states, infos, task, seq_len=3)
# metrics['dimensionality'] = effective dimensionality of representations
```

## Command Line Options

```bash
python run_experiment.py --help

# Task parameters
--n_stimuli 8              # Number of unique stimuli
--seq_lengths 2 3 4        # Sequence lengths to train on

# Model parameters
--model_type gru           # vanilla, gru, or lstm
--hidden_dim 128           # Hidden layer size

# Training parameters
--n_epochs 2000            # Training epochs
--learning_rate 0.01       # Learning rate
--batch_size 64            # Batch size
--use_curriculum           # Enable curriculum learning
--curriculum_mix_ratio 0.3 # Replay ratio for previous lengths
--curriculum_threshold 0.9 # Accuracy to advance curriculum

# Output
--save_dir results/exp1    # Where to save results
--seed 42                  # Random seed
```

## Example Results

After training a GRU for 2000 epochs:

```
Final validation accuracies:
  seq_len=2: item_acc=100.00%, seq_acc=100.00%
  seq_len=3: item_acc=99.48%, seq_acc=98.96%
  seq_len=4: item_acc=98.44%, seq_acc=94.27%
```

Subspace analysis shows position subspaces are partially overlapping (~60° mean angle, where 90° would be orthogonal):

```
Delay subspace angles (seq_len=4):
  Pos 1 vs Pos 2: mean 38.5°
  Pos 1 vs Pos 3: mean 56.2°
  Pos 1 vs Pos 4: mean 71.3°
```

## Output Files

After running an experiment, the save directory contains:

```
results/exp_XXXXXXXX/
├── config.json           # Experiment configuration
├── final_model.pt        # Trained model checkpoint
├── history.json          # Training history
├── training_history.png  # Loss/accuracy plots
├── analysis_results.json # All analysis metrics
└── figures/
    ├── pca_trajectories_sl*.png
    ├── mean_trajectory_sl*.png
    ├── decoding_sl*.png
    └── subspace_sl*.png
```

## References

- Xie, Y., Hu, P., Li, J., et al. (2022). Geometry of sequence working memory in macaque prefrontal cortex. *Science*, 375(6581), 632-639.

## License

MIT License
