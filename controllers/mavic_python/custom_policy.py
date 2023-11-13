from typing import Callable
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy, ActorCriticPolicy

from model import AttentionBiLSTM



class CustomPolicy(MultiInputActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        

        # self.attention_biLSTM = AttentionBiLSTM(self.num_classes, self.hidden_size, self.num_layers, self.num_frames, self.dropout_prob)
        

    # def forward(self, obs):
    #     action, value, log_probs = self.attention_biLSTM(obs)
        
    #     return action, value, log_probs

    def _build_mlp_extractor(self) -> None:
        num_classes=4,
        hidden_size=256,
        num_layers=4,
        num_frames=7,
        dropout_prob=0.3

        self.mlp_extractor = AttentionBiLSTM(num_classes, hidden_size, num_layers, num_frames, dropout_prob, self.features_dim)