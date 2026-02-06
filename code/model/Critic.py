import torch
import torch.nn as nn

class ValueCritic(nn.Module):
    """
  V(s), forevaluationusercurrent. 
 input: User Embedding (from OneRec)
 output: Scalar Value ( Reward)
 """
    def __init__(self, input_dim, hidden_dim=128):
        super(ValueCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, user_emb):
        value = self.net(user_emb) # [B, 1]
        return value.squeeze(-1)