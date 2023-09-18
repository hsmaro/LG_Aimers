import torch.nn as nn
import torch

class SalesForecastNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, use_layernorm):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_layernorm = use_layernorm

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=False,
        )

        # if use_layernorm is True
        if self.use_layernorm:
            self.layernorms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.hidden_size // 2, self.output_size)
        )
        self.actv = nn.ReLU()


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Apply layer norm if use_layernorm True
        if self.use_layernorm:
            for i in range(self.num_layers):
                out = self.layernorms[i](out)
        out = out[:, -1, :]
        out = self.actv(self.fc(out))
        out = out.squeeze(1)
        return out