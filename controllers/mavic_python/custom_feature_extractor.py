import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from self_attention import SelfAttention

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64, hidden_size: int = 256, num_layers: int = 4, num_frames: int = 7):
        super().__init__(observation_space, features_dim=num_frames * hidden_size * 2)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8, stride=8),
        )

        self.bilstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.hidden = self.init_hidden()

        self.attention = SelfAttention(hidden_size * 2)

    def extract_feature(self, image_depth):
        image_depth = image_depth.unsqueeze(0)
        extracted_feature = self.cnn(image_depth)
        return extracted_feature
    
    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
        return (h0, c0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        images_depth = observations['frames']

        # Get batch size dynamically
        batch_size = images_depth.size(0)

        images_depth = images_depth.view(batch_size * self.num_frames, 1, 64, 64)

        # Apply CNN
        cnn_outputs = self.cnn(images_depth)

        # Reshape to (batch_size, num_frames, channels, height, width)
        cnn_outputs = cnn_outputs.view(batch_size, self.num_frames, -1, 8, 8)

        # Swap axes to (batch_size, channels, num_frames, height, width)
        cnn_outputs = cnn_outputs.permute(0, 2, 1, 3, 4)

        # Flatten the channels and num_frames dimensions
        cnn_outputs = cnn_outputs.view(batch_size, -1, 64)

        # Apply LSTM
        lstm_out, _ = self.bilstm(cnn_outputs)

        # Apply Attention
        weighted_context = self.attention(lstm_out)
        weighted_context = weighted_context.view(batch_size, -1)

        return weighted_context


