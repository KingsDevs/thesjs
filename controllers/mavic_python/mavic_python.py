from stable_baselines3.common.env_checker import check_env
from environment import OpenAIGymEnvironment
from stable_baselines3 import PPO
from custom_policy import CustomPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_feature_extractor import CustomFeatureExtractor
import os


def main():
    num_frames = 7

    env = OpenAIGymEnvironment(number_of_stack_images=num_frames)
    check_env(env)

    print(f"observation space: {env.observation_space}")
    print(f"action space: {env.action_space}")

    # policy_kwargs = dict(
    #     num_classes=4,
    # #     hidden_size=256,
    # #     num_layers=4,
    # #     num_frames=num_frames,
    # #     dropout_prob=0.3
    # # )

    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=64,
            hidden_size=256,
            num_layers=4,
            num_frames=num_frames
        ),
    )


    model = PPO(
        'MultiInputPolicy', 
        env, 
        n_steps=1000, 
        verbose=1,
        learning_rate=0.001,
        clip_range=0.35,
        ent_coef=0.0001, 
        tensorboard_log="./PPO_Policy_Mavic", 
        policy_kwargs=policy_kwargs,
        batch_size=20
    )
    total_timestep = 1000000

    log_dir = "results/train1/"
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    model.learn(total_timesteps=total_timestep, progress_bar=True, callback=checkpoint_callback, tb_log_name="AttentionBiLSTM")
    model.save('attention_bilstm_policy')

    # model = PPO.load("results/train3/rl_model_200000_steps")
    

    obs, info = env.reset()
    for _ in range(100000):
        # action, _ = model.predict(obs, deterministic=True)
        # action = random.randint(0,7)
        action = 0
        obs, reward, done, truncated, info = env.step(action)
        print(reward)
        if done:
            obs, info = env.reset()

if __name__ == '__main__':
    main()



