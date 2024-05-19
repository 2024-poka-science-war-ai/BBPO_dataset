import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

SEQ_LEN = 4

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.01, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_embed_size, target_embed_size, num_classes, num_layers, nhead, dropout=0.01, max_seq_length=5000):
        super(TransformerClassifier, self).__init__()

        # Initial embedding layer
        self.embedding = nn.Linear(input_embed_size, target_embed_size)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(target_embed_size, dropout, max_seq_length)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=target_embed_size,
            nhead=nhead,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.output_layer = nn.Linear(target_embed_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_length, input_embed_size]
        
        # Embedding
        x = self.embedding(x)  # [batch_size, seq_length, target_embed_size]
        
        # Permute x to [seq_length, batch_size, target_embed_size] for Transformer and Positional Encoding
        x = x.permute(1, 0, 2)
        
        # Apply Positional Encoding
        x = self.positional_encoding(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Permute x back to [batch_size, seq_length, target_embed_size]
        x = x.permute(1, 0, 2)
        
        # Average pooling across the sequence dimension
        x = x.mean(dim=1)
        
        return x


embed_size = 256
num_classes = 45
num_layers = 6
input_embed_size = 37  # Original embedding size
target_embed_size = 1024 # Desired embedding size
nhead = 8

class Core(nn.Module):
    def __init__(self):
        super(Core, self).__init__()
        # self.transformer = TransformerClassifier(21, target_embed_size, num_classes, num_layers, nhead)
        self.state_conv = nn.Conv1d(in_channels=SEQ_LEN, out_channels=1, kernel_size=1, stride=1)
        self.action_conv = nn.Conv1d(in_channels=SEQ_LEN-1, out_channels=1, kernel_size=1, stride=1)
        
        self.embed_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, stride=1, padding=1)
        
        self.flatten = nn.Flatten()
        self.input_dim = 80
        self.output_dim = 45
        dim = target_embed_size
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, self.output_dim)   
        )

    def forward(self, x, y):
        B = x.shape[0]
        x = self.state_conv(x)
        y = self.action_conv(y)
        z = torch.cat([x, y], dim=2).reshape(B, 1, 81)
        z = self.embed_conv(z).reshape(B, 80)
        logits = self.model(z)
        return logits
    
    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
