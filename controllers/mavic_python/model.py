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

        # Feature Extractor
        # self.fe = CustomFeatureExtractor()

        # # Single BiLSTM
        # self.bilstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # self.hidden = self.init_hidden()

        # # Attention mechanism
        # self.attention = SelfAttention(hidden_size * 2)

        # # Dropout layer
        # self.dropout = nn.Dropout(dropout_prob)

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

        

    def extract_feature(self, image_depth):
        image_depth = image_depth.unsqueeze(0)
        extracted_feature = self.fe(image_depth)
        return extracted_feature
    
    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
        return (h0, c0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # images_depth = observations['frames']
        # other_observations = observations['other_observations']
        # other_observations = other_observations.view(self.num_frames, 1, 5)

        # # Extract features
        # images_depth = images_depth.squeeze(0)
        # images_depth = torch.stack([self.extract_feature(images_depth[i]) for i in range(images_depth.shape[0])], dim=0)
        
        # images_depth = images_depth.squeeze(1)
        # images_depth = images_depth.view(1, self.num_frames, 64)
        # # other_observations = other_observations.repeat(1, images_depth.size(-1), 1)

        # # combined_input = torch.cat([images_depth, other_observations], dim=-1)
        # # combined_input = combined_input.view(1, self.num_frames, 36)

        # # Apply LSTM
        # lstm_out, hidden = self.bilstm(images_depth, self.hidden)
        # self.hidden = hidden

        # # Apply Attention
        # weighted_context = self.attention(lstm_out)
        # weighted_context = weighted_context.view(-1)

        # Fully connected layer
        # logits = self.fc(weighted_context)
        # logits = logits.unsqueeze(0)

        # state_value = self.value_head(weighted_context)
        # state_value = state_value.squeeze(0)
        
        # softmax_prob = torch.softmax(logits, dim=1)

        # action = torch.multinomial(softmax_prob.squeeze(0), 1)
        # log_probs = torch.log(softmax_prob[:,action.item()])
        # log_probs = log_probs.view(-1, 1)

        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)
