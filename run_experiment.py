#!/usr/bin/env python
"""
Main script to run RNN sequence working memory experiment.

Usage:
    python run_experiment.py --hidden_dim 128 --n_epochs 500 --save_dir results/exp1

This script:
1. Creates the task and model
2. Trains the model
3. Runs representation geometry analysis
4. Generates figures
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from task import SequenceWorkingMemoryTask
from model import create_model
from train import train_model, load_model
from analysis import (
    extract_hidden_states,
    run_all_analyses,
    decode_item_and_position,
    subspace_analysis,
    compute_geometry_metrics,
    analyze_dynamics,
)
from utils import (
    set_plotting_style,
    plot_training_history,
    plot_pca_trajectories,
    plot_mean_trajectory_by_phase,
    plot_decoding_results,
    plot_subspace_analysis,
    create_analysis_report,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train RNN on sequence working memory task'
    )

    # Task parameters
    parser.add_argument('--n_stimuli', type=int, default=8,
                        help='Number of distinct stimuli')
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[2, 3, 4],
                        help='Possible sequence lengths')
    parser.add_argument('--fixation_duration', type=int, default=5,
                        help='Fixation period duration (timesteps)')
    parser.add_argument('--stimulus_duration', type=int, default=5,
                        help='Stimulus presentation duration (timesteps)')
    parser.add_argument('--delay_duration', type=int, default=10,
                        help='Delay period duration (timesteps)')
    parser.add_argument('--response_duration', type=int, default=5,
                        help='Response period duration (timesteps)')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='vanilla',
                        choices=['vanilla', 'lstm', 'gru'],
                        help='Type of RNN')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden layer dimension')
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='Noise std added to hidden state during training')

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--use_curriculum', action='store_true', default=True,
                        help='Use curriculum learning')
    parser.add_argument('--no_curriculum', action='store_false', dest='use_curriculum',
                        help='Disable curriculum learning')
    parser.add_argument('--curriculum_mix_ratio', type=float, default=0.3,
                        help='Fraction of batch from previous seq lengths (prevents forgetting)')
    parser.add_argument('--curriculum_threshold', type=float, default=0.9,
                        help='Accuracy threshold to advance curriculum')
    parser.add_argument('--eval_every', type=int, default=20,
                        help='Evaluate every N epochs')

    # Analysis parameters
    parser.add_argument('--n_analysis_trials', type=int, default=200,
                        help='Number of trials for analysis')

    # Output
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results (auto-generated if not specified)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cpu, cuda, mps, or auto)')

    # Mode
    parser.add_argument('--analyze_only', type=str, default=None,
                        help='Path to checkpoint to analyze (skip training)')

    return parser.parse_args()


def get_device(device_str: str) -> str:
    """Get the appropriate device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_str


def main():
    """Main function."""
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = f'results/exp_{timestamp}'
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = vars(args)
    config['device'] = device
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Saving results to: {save_path}")

    # Create task
    task = SequenceWorkingMemoryTask(
        n_stimuli=args.n_stimuli,
        seq_lengths=args.seq_lengths,
        fixation_duration=args.fixation_duration,
        stimulus_duration=args.stimulus_duration,
        delay_duration=args.delay_duration,
        response_duration=args.response_duration,
    )

    print(f"\nTask configuration:")
    print(f"  n_stimuli: {task.n_stimuli}")
    print(f"  seq_lengths: {task.seq_lengths}")
    print(f"  input_dim: {task.input_dim}")
    print(f"  output_dim: {task.output_dim}")

    # Training or load model
    if args.analyze_only is None:
        # Create model
        model = create_model(
            args.model_type,
            input_dim=task.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=task.output_dim,
            noise_std=args.noise_std,
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel configuration:")
        print(f"  type: {args.model_type}")
        print(f"  hidden_dim: {args.hidden_dim}")
        print(f"  parameters: {n_params:,}")

        # Train model
        print(f"\nTraining for {args.n_epochs} epochs...")
        history = train_model(
            model,
            task,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_curriculum=args.use_curriculum,
            curriculum_mix_ratio=args.curriculum_mix_ratio,
            curriculum_threshold=args.curriculum_threshold,
            eval_every=args.eval_every,
            save_dir=str(save_path),
            device=device,
            verbose=True,
        )

        # Plot training history
        set_plotting_style()
        fig = plot_training_history(history, save_path=save_path / 'training_history.png')

        print("\nFinal validation accuracies:")
        for sl in task.seq_lengths:
            if history['val_metrics'][sl]:
                item_acc = history['val_metrics'][sl][-1]['item_accuracy']
                seq_acc = history['val_metrics'][sl][-1]['sequence_accuracy']
                print(f"  seq_len={sl}: item_acc={item_acc:.2%}, seq_acc={seq_acc:.2%}")

    else:
        # Load pre-trained model
        print(f"Loading model from {args.analyze_only}")
        model, checkpoint = load_model(args.analyze_only, args.model_type, device)
        history = checkpoint.get('history', None)

    # Run analysis
    print("\nRunning representation analysis...")
    model = model.to(device)
    model.eval()

    analysis_results = run_all_analyses(
        model,
        task,
        n_trials=args.n_analysis_trials,
        device=device,
    )

    # Save analysis results
    with open(save_path / 'analysis_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, tuple):
                return str(obj)  # Convert tuple keys to strings
            return obj
        json.dump(convert(analysis_results), f, indent=2)

    # Generate analysis figures
    print("\nGenerating figures...")
    figures_path = save_path / 'figures'
    figures_path.mkdir(exist_ok=True)

    for seq_len in task.seq_lengths:
        print(f"  Figures for seq_len={seq_len}...")

        # Generate trials for visualization
        inputs, targets, mask, infos = task.generate_batch(
            args.n_analysis_trials, seq_len=seq_len
        )
        inputs = inputs.to(device)
        hidden_states = extract_hidden_states(model, inputs, device).numpy()

        # PCA trajectories
        fig = plot_pca_trajectories(
            hidden_states, infos, task, seq_len, n_trials=20,
            save_path=figures_path / f'pca_trajectories_sl{seq_len}.png'
        )

        # Mean trajectory by phase
        fig = plot_mean_trajectory_by_phase(
            hidden_states, task, seq_len,
            save_path=figures_path / f'mean_trajectory_sl{seq_len}.png'
        )

        # Decoding results
        if seq_len in analysis_results:
            fig = plot_decoding_results(
                analysis_results[seq_len]['decoding'],
                seq_len,
                save_path=figures_path / f'decoding_sl{seq_len}.png'
            )

            # Subspace analysis
            fig = plot_subspace_analysis(
                analysis_results[seq_len]['subspace'],
                seq_len,
                save_path=figures_path / f'subspace_sl{seq_len}.png'
            )

    # Create summary report
    create_analysis_report(analysis_results, str(figures_path))

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    for seq_len, results in analysis_results.items():
        print(f"\nSequence length = {seq_len}")
        print("-" * 40)

        # Geometry
        geom = results['geometry']
        print(f"  Dimensionality (participation ratio): {geom['dimensionality']:.2f}")
        if 'item_rdm_correlation' in geom:
            print(f"  Item RDM correlation: {geom['item_rdm_correlation']:.3f}")
        if 'position_rdm_correlation' in geom:
            print(f"  Position RDM correlation: {geom['position_rdm_correlation']:.3f}")

        # Decoding
        decoding = results['decoding']
        print(f"  Decoding accuracy at stimulus offset:")
        for pos in range(seq_len):
            acc = decoding['item_at_position'][pos]['accuracy']
            print(f"    Position {pos+1}: {acc:.2%}")

    print("\n" + "="*60)
    print(f"Results saved to: {save_path}")
    print("="*60)


if __name__ == "__main__":
    main()
