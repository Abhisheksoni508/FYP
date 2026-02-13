import torch
import torch.nn as nn

class RUL_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(RUL_LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # The LSTM Layer
        # batch_first=True means input shape is (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # The Output Layer (Regressor)
        # Takes the LSTM features and outputs a single number (RUL)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [Batch, Window_Size, Features]
        
        # Forward propagate LSTM
        # out shape: [Batch, Window_Size, Hidden_Dim]
        out, _ = self.lstm(x)
        
        # We only care about the LAST time step for prediction
        # out[:, -1, :] takes the last slice of the sequence
        last_step_out = out[:, -1, :]
        
        # Push through the linear layer
        prediction = self.fc(last_step_out)
        
        return prediction