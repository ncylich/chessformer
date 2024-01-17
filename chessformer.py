import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Creating positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ChessTransformer(nn.Module):
    def __init__(self, inp_dict: int, out_dict: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Chess-former'

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder Layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Embedding layers for input
        self.embedding = nn.Embedding(inp_dict, d_model)
        self.x_embedding = nn.Embedding(8, d_model)
        self.y_embedding = nn.Embedding(8, d_model)
        self.d_model = d_model

        # Output linear layer
        self.linear_output = nn.Linear(d_model, out_dict)

        # Initialization of weights
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.x_embedding.weight.data.uniform_(-initrange, initrange)
        self.y_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear_output.bias.data.zero_()
        self.linear_output.weight.data.uniform_(-initrange, initrange)

    def forward(self, board: Tensor, src_mask: Tensor = None) -> Tensor:
        board_emb = self.embedding(board)

        # Generating input for positional embeddings
        batch_size = board.shape[0]
        x_inp = torch.arange(8).unsqueeze(0).repeat(batch_size, 8).to(board.device)
        y_inp = torch.arange(8).repeat_interleave(8).unsqueeze(0).repeat(batch_size, 1).to(board.device)

        x_emb = self.x_embedding(x_inp)
        y_emb = self.y_embedding(y_inp)

        # Combining embeddings
        combined_emb = board_emb + x_emb + y_emb

        # Scaling and passing through transformer
        combined_emb = combined_emb * math.sqrt(self.d_model)
        output = self.transformer_encoder(combined_emb.permute(1, 0, 2)).permute(1, 0, 2)

        # Applying linear layer to the output
        output = self.linear_output(output)
        return output
