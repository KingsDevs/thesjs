import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64, hidden_size: int = 256, num_layers: int = 4, num_frames: int = 7):
        super().__init__(observation_space, features_dim=num_frames * hidden_size * 2)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.num_frames = num_frames
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 64)  # Adjusted fully connected layer
        )

        self.bilstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.5)

        self.query = nn.Linear(hidden_size * 2, hidden_size * 2, device=self.device)
        self.key = nn.Linear(hidden_size * 2, hidden_size * 2, device=self.device)
        self.value = nn.Linear(hidden_size * 2, hidden_size * 2, device=self.device)

        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=4, batch_first=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        images_depth = observations['frames']

        # Get batch size dynamically
        batch_size = images_depth.size(0)

        # images_depth = images_depth.view(batch_size * self.num_frames, 1, 64, 64)

        # # Apply CNN
        # cnn_outputs = self.cnn(images_depth)
        # cnn_outputs = cnn_outputs.view(batch_size, self.num_frames, -1)

        cnn_outputs = torch.stack([self.cnn(images_depth[i, j, :, :].unsqueeze(0).unsqueeze(0)) for i in range(batch_size) for j in range(self.num_frames)])
        cnn_outputs = cnn_outputs.view(batch_size, self.num_frames, 64)

        # Apply LSTM
        lstm_out, _ = self.bilstm(cnn_outputs)

        # Apply Attention
        queries = self.query(lstm_out)
        keys = self.key(lstm_out)
        values = self.value(lstm_out)
        weighted_context, _ = self.multihead_attention(queries, keys, values)
        weighted_context = weighted_context.reshape(batch_size, -1)

        return weighted_context
    

class CustomFeatureExtractorCNNOnly(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64, hidden_size: int = 256, num_layers: int = 4):
        super().__init__(observation_space, features_dim=64)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 64)  # Adjusted fully connected layer
        )


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        image_depth = observations

        # Get batch size dynamically
        batch_size = image_depth.size(0)

        cnn_outputs = self.cnn(image_depth.unsqueeze(0))
        features = cnn_outputs.reshape(batch_size, -1)

        return features
