import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

from cnn_only import OpenAIGymEnvironmentOnlyCNN
from stable_baselines3.common.vec_env import VecFrameStack


vec_env = VecFrameStack(OpenAIGymEnvironmentOnlyCNN(), n_stack=8)

model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=1)
model.learn(5000)



model.save("ppo_recurrent")
del model # remove to demonstrate saving and loading
model = RecurrentPPO.load("ppo_recurrent")

obs = vec_env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    episode_starts = dones