from stable_baselines3 import PPO
from lstm_attention import OpenAIGymEnvironmentCNNLSTM


if __name__ == "__main__":
    num_frames = 8
    env = OpenAIGymEnvironmentCNNLSTM(number_of_stack_images=num_frames)

    model = PPO.load("dfd")
    obs, info = env.reset()
    for _ in range(100000):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        print(action)
        if done:
            obs, info = env.reset()