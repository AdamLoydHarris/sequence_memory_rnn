"""
Vanilla RNN model for sequence working memory task.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class VanillaRNN(nn.Module):
    """
    Vanilla RNN with tanh nonlinearity for sequence working memory.

    Stores hidden states during forward pass for analysis.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        noise_std: float = 0.0,
    ):
        """
        Args:
            input_dim: Dimension of input
            hidden_dim: Number of hidden units
            output_dim: Dimension of output
            noise_std: Standard deviation of noise added to hidden state
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.noise_std = noise_std

        # RNN weights
        self.W_in = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, output_dim, bias=True)

        # Initialize weights
        self._init_weights()

        # Storage for hidden states (set during forward)
        self.hidden_states = None

    def _init_weights(self):
        """Initialize weights with reasonable defaults."""
        # Input weights: Xavier initialization
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.zeros_(self.W_in.bias)

        # Recurrent weights: scaled to have spectral radius ~1
        nn.init.orthogonal_(self.W_rec.weight)
        with torch.no_grad():
            self.W_rec.weight.mul_(0.9)  # Scale to spectral radius ~0.9

        # Output weights: Xavier initialization
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        store_hidden: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN.

        Args:
            inputs: (batch_size, seq_len, input_dim) input tensor
            h0: (batch_size, hidden_dim) initial hidden state (zeros if None)
            store_hidden: Whether to store hidden states for analysis

        Returns:
            outputs: (batch_size, seq_len, output_dim) output tensor
            h_final: (batch_size, hidden_dim) final hidden state
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        # Initialize hidden state
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            h = h0

        # Storage for outputs and hidden states
        outputs = []
        if store_hidden:
            hidden_states = [h.detach().clone()]

        # Run through time
        for t in range(seq_len):
            x_t = inputs[:, t, :]

            # RNN update: h_t = tanh(W_in @ x_t + W_rec @ h_{t-1})
            h = torch.tanh(self.W_in(x_t) + self.W_rec(h))

            # Add noise during training
            if self.training and self.noise_std > 0:
                h = h + self.noise_std * torch.randn_like(h)

            # Compute output
            y_t = self.W_out(h)
            outputs.append(y_t)

            if store_hidden:
                hidden_states.append(h.detach().clone())

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, output_dim)

        if store_hidden:
            self.hidden_states = torch.stack(hidden_states, dim=1)  # (batch, seq_len+1, hidden_dim)

        return outputs, h

    def get_hidden_states(self) -> Optional[torch.Tensor]:
        """
        Get stored hidden states from last forward pass.

        Returns:
            hidden_states: (batch_size, seq_len+1, hidden_dim) or None
        """
        return self.hidden_states


class LSTM(nn.Module):
    """
    LSTM model for comparison.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.W_out = nn.Linear(hidden_dim, output_dim)

        self.hidden_states = None

    def forward(
        self,
        inputs: torch.Tensor,
        h0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        store_hidden: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM.
        """
        if h0 is None:
            hidden_seq, (h_n, c_n) = self.lstm(inputs)
        else:
            hidden_seq, (h_n, c_n) = self.lstm(inputs, h0)

        outputs = self.W_out(hidden_seq)

        if store_hidden:
            self.hidden_states = hidden_seq.detach()

        return outputs, h_n[-1]

    def get_hidden_states(self) -> Optional[torch.Tensor]:
        return self.hidden_states


class GRU(nn.Module):
    """
    GRU model for comparison.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.W_out = nn.Linear(hidden_dim, output_dim)

        self.hidden_states = None

    def forward(
        self,
        inputs: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        store_hidden: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GRU.
        """
        if h0 is None:
            hidden_seq, h_n = self.gru(inputs)
        else:
            hidden_seq, h_n = self.gru(inputs, h0)

        outputs = self.W_out(hidden_seq)

        if store_hidden:
            self.hidden_states = hidden_seq.detach()

        return outputs, h_n[-1]

    def get_hidden_states(self) -> Optional[torch.Tensor]:
        return self.hidden_states


def create_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: 'vanilla', 'lstm', or 'gru'
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        **kwargs: Additional model-specific arguments

    Returns:
        model: The created model
    """
    if model_type == 'vanilla':
        vanilla_kwargs = {k: v for k, v in kwargs.items() if k in ['noise_std']}
        return VanillaRNN(input_dim, hidden_dim, output_dim, **vanilla_kwargs)
    elif model_type == 'lstm':
        lstm_kwargs = {k: v for k, v in kwargs.items() if k in ['num_layers']}
        return LSTM(input_dim, hidden_dim, output_dim, **lstm_kwargs)
    elif model_type == 'gru':
        gru_kwargs = {k: v for k, v in kwargs.items() if k in ['num_layers']}
        return GRU(input_dim, hidden_dim, output_dim, **gru_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the model
    batch_size = 32
    seq_len = 50
    input_dim = 15
    hidden_dim = 128
    output_dim = 8

    model = VanillaRNN(input_dim, hidden_dim, output_dim)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    outputs, h_final = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Final hidden shape: {h_final.shape}")
    print(f"Stored hidden states shape: {model.get_hidden_states().shape}")
