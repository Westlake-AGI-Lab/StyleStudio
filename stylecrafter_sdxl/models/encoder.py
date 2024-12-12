import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformers import Transformer


class QueryTransformer(nn.Module):
    def __init__(self, embedding_dim=1024, output_dim=1024, num_heads=8, num_queries=4, n_layers=2):
        super().__init__()
        scale = embedding_dim ** -0.5
        self.num_queries = num_queries
        self.query_emb = nn.Parameter(torch.randn(1, num_queries, embedding_dim) * scale)
        self.transformer_blocks = Transformer(
            width=embedding_dim,
            layers=n_layers,
            heads=num_heads,
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.proj = nn.Parameter(torch.randn(embedding_dim, output_dim) * scale)
    
    def forward(self, x):
        query_emb = self.query_emb.repeat(x.shape[0], 1, 1)
        x = torch.cat([query_emb, x], dim=1)
        # x = torch.cat([x, style_emb], dim=1)
        x = self.ln1(x)
        
        x = x.permute(1, 0, 2)
        x = self.transformer_blocks(x)
        x = x.permute(1, 0, 2)

        x = self.ln2(x[:, :self.num_queries, :])
        x = x @ self.proj
        return x

