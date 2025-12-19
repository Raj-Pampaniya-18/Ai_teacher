import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                batch_first=True
            )
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.size()
        device = x.device

        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)

        causal_mask = torch.triu(
            torch.ones(T, T, device=device),
            diagonal=1
        ).bool()

        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)

        x = self.ln(x)
        return self.fc(x)
