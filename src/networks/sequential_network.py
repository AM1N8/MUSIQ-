import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class SequentialStateEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_layers: int = 2, 
        nhead: int = 4, 
        dropout: float = 0.1,
        max_len: int = 20
    ):
        """
        Transformer-based encoder for sequential state data.
        
        Args:
            input_dim: Dimension of input features per step
            hidden_dim: Dimension of internal transformer state
            num_layers: Number of transformer layers
            nhead: Number of attention heads
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Encoded state vector of shape (batch_size, hidden_dim)
            (Uses the embedding of the last element in the sequence)
        """
        # Project input to hidden dim
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply Transformer
        # Shape: (batch_size, seq_len, hidden_dim)
        x = self.transformer_encoder(x)
        
        # Aggregate output
        # Option 1: Average pooling
        # output = x.mean(dim=1)
        
        # Option 2: Last token (common for causal/sequential tasks)
        output = x[:, -1, :]
        
        return output
