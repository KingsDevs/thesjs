import torch
import torch.nn as nn
from self_attention import SelfAttention
from custom_feature_extractor import CustomFeatureExtractor
from typing import Tuple

class AttentionBiLSTM(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, num_layers: int, num_frames: int, dropout_prob, feature_dim: int, last_layer_dim_pi: int = 64, last_layer_dim_vf: int = 64, **kwargs):
        super(AttentionBiLSTM, self).__init__(**kwargs)

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_frames = num_frames 
        self.num_classes = num_classes 

        # Value Network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_size[0] // 2, device=self.device),
            nn.ReLU(),
            nn.Linear(hidden_size[0] // 2, last_layer_dim_vf, device=self.device),
        )
        
        # Fully connected layer for classification
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_size[0], device=self.device),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[0] // 2, device=self.device),
            nn.ReLU(),
            nn.Linear(hidden_size[0] // 2, last_layer_dim_pi, device=self.device),
        )


    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)
