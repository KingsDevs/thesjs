import sys
from controller import Supervisor

import gymnasium as gym
import numpy as np
from stable_baselines3.common.envs import SimpleMultiObsEnv
import cv2
from gymnasium import spaces
import math

import time

class OpenAIGymEnvironmentCNNLSTM(Supervisor, SimpleMultiObsEnv):
    def __init__(self, max_episode_steps=1000, number_of_stack_images=5):
        super().__init__()

        self.__camera = self.getDevice("camera")
        self.__range_finder = self.getDevice("range-finder")

        self.spec = gym.envs.registration.EnvSpec(
            id='WebotsThesisEnv-v0', 
            max_episode_steps=max_episode_steps,
            entry_point='environment:OpenAIGymEnvironment'
        )

        self.action_space = spaces.Discrete(4) 
        self.__number_of_stack_images = number_of_stack_images
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(number_of_stack_images, 64, 64),
            dtype=np.float32
        )

        self.__stack_images = np.ones(
            (number_of_stack_images, 64, 64),
            dtype=np.float32
        )
        
        self.state = None

        self.__timestep = int(self.getBasicTimeStep())
        self.__front_left_led = self.getDevice("front left led")
        self.__front_right_led = self.getDevice("front right led")
        self.__imu = self.getDevice("inertial unit")
        self.__gps = self.getDevice("gps")
        self.__compass = self.getDevice("compass")
        self.__gyro = self.getDevice("gyro")

        self.__touch_sensors = []
        for touch_sensor in ['touch sensor']:
            self.__touch_sensors.append(self.getDevice(touch_sensor))
        

        self.__camera.enable(self.__timestep)
        self.__range_finder.enable(self.__timestep)
        self.__imu.enable(self.__timestep)
        self.__gps.enable(self.__timestep)
        self.__compass.enable(self.__timestep)
        self.__gyro.enable(self.__timestep)

        for touch_sensor in self.__touch_sensors:
            touch_sensor.enable(self.__timestep)

        self.__camera_roll_motor = self.getDevice("camera roll")
        self.__camera_pitch_motor = self.getDevice("camera pitch")

        self.__front_left_motor = self.getDevice("front left propeller")
        self.__front_right_motor = self.getDevice("front right propeller")
        self.__rear_left_motor = self.getDevice("rear left propeller")
        self.__rear_right_motor = self.getDevice("rear right propeller")

        self.__enable_motors()

        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)
        
        print("initialized Robot and Environment")
        self.step_count = 0

        self.__target_altitude = 0.3
        # Constants, empirically found.
        self.__k_vertical_thrust = 68.5  
        self.__k_vertical_offset = 0.6  
        self.__k_vertical_p = 3.0  
        self.__k_roll_p = 50.0  
        self.__k_pitch_p = 30.0  

        self.__original_position = np.array(self.getSelf().getField('translation').getSFVec3f())[:2]
        self.previous_distance = 0.0
        self.hasTakeOff = False
        

    def __enable_motors(self):
        motors = [self.__front_left_motor, self.__front_right_motor, self.__rear_left_motor, self.__rear_right_motor]
        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1.0)


    def reset_obstacles(self):
        for obstacle in ["obstacle1"]:
            current_obstacle = self.getFromDef(obstacle)
            current_obstacle_pos = current_obstacle.getField('translation')
            random_y_position = np.random.uniform(low=-1.0, high=1.0)
            current_obstacle_pos.setSFVec3f([current_obstacle_pos.getSFVec3f()[0], random_y_position, current_obstacle_pos.getSFVec3f()[2]])

    def reset_mav_pos(self):
        random_y_position = np.random.uniform(low=-1.0, high=1.0)

        current_pos = self.getSelf().getField('translation')
        current_pos.setSFVec3f([current_pos.getSFVec3f()[0], random_y_position, current_pos.getSFVec3f()[2]])


    def reset(self, seed=None):
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        self.__enable_motors()
        
        self.__target_altitude = 0.4
        self.make_action(7)

        self.reset_mav_pos()
        self.reset_obstacles()
        self.hasTakeOff = False
        super().step(self.__timestep)

        while not abs(self.__gps.getValues()[2] - self.__target_altitude) < 0.09:
            self.make_action(3)
            super().step(self.__timestep)

        self.__original_position = np.array(self.getSelf().getField('translation').getSFVec3f())[:2]
        self.previous_distance = 0.0

        self.__stack_images = np.ones((self.__number_of_stack_images, 64, 64), dtype=np.float32)
        

        observation = self.__stack_images

        self.state = observation
        
        
        return observation, {}
    
    def make_action(self, action):
        roll = self.__imu.getRollPitchYaw()[0]
        pitch = self.__imu.getRollPitchYaw()[1]
        altitude = self.__gps.getValues()[2]
        roll_acceleration = self.__gyro.getValues()[0]
        pitch_acceleration = self.__gyro.getValues()[1]

        led_state = int(time.time()) % 2
        self.__front_left_led.set(led_state)
        self.__front_right_led.set(not led_state)

        self.__camera_roll_motor.setPosition(-0.115 * roll_acceleration)
        self.__camera_pitch_motor.setPosition(-0.1 * pitch_acceleration)

        roll_disturbance = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance = 0.0

        current_altitude = self.__gps.getValues()[2]
        if abs(current_altitude - self.__target_altitude) < 0.09 or self.hasTakeOff:
            if action == 0:
                pitch_disturbance = -2.0
            # elif action == 1:
            #     pitch_disturbance = 2.0
            elif action == 1:
                yaw_disturbance = -1.3
            elif action == 2:
                yaw_disturbance = 1.3
            # elif action == 1:
            #     roll_disturbance = -1.0
            # elif action == 2:
            #     roll_disturbance = 1.0

            self.hasTakeOff = True

            # elif action == 6:
            #     self.__target_altitude += 0.05
            #     print("target altitude: %f [m]" % self.__target_altitude)
            # elif action == 7:
            #     self.__target_altitude -= 0.05

        roll_input = self.__k_roll_p * min(max(roll, -1.0), 1.0) + roll_acceleration + roll_disturbance
        pitch_input = self.__k_pitch_p * min(max(pitch, -1.0), 1.0) + pitch_acceleration + pitch_disturbance
        yaw_input = yaw_disturbance
        clamped_difference_altitude = max(min(self.__target_altitude - altitude + self.__k_vertical_offset, 1.0), -1.0)
        vertical_input = self.__k_vertical_p * clamped_difference_altitude ** 3.0

        front_left_motor_input = self.__k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
        front_right_motor_input = self.__k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
        rear_left_motor_input = self.__k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
        rear_right_motor_input = self.__k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input

        self.__front_left_motor.setVelocity(front_left_motor_input)
        self.__front_right_motor.setVelocity(-front_right_motor_input)
        self.__rear_left_motor.setVelocity(-rear_left_motor_input)
        self.__rear_right_motor.setVelocity(rear_right_motor_input)

    
    def step(self, action):
        self.step_count+=1

        
        #action

        self.make_action(action)

        super().step(self.__timestep)

        #Observations
        image_range = self.__range_finder.getRangeImage(data_type='buffer')
        depth_image_np = np.ctypeslib.as_array(image_range, (self.__range_finder.getWidth() * self.__range_finder.getHeight(),))
        maxRangeValueMask = np.isinf(depth_image_np)
        depth_image_np[maxRangeValueMask] = self.__range_finder.getMaxRange()
        depth_image_np[depth_image_np > self.__range_finder.getMaxRange()] = self.__range_finder.getMaxRange()
        normalized_depth_image = depth_image_np / self.__range_finder.getMaxRange()
        normalized_depth_image = normalized_depth_image.reshape(self.__range_finder.getWidth(), self.__range_finder.getHeight())
        
        if self.hasTakeOff:
            self.__stack_images = np.roll(self.__stack_images, shift=1, axis=0)
            self.__stack_images[0, :, :] = normalized_depth_image

            self.state = self.__stack_images
        

        done = False
        #reward
        reward = 0

        collided = any(touch_sensor.getValue() for touch_sensor in self.__touch_sensors)

        current_position = np.array(self.getSelf().getField('translation').getSFVec3f())[:2]
        current_distance = math.sqrt((current_position[0] - self.__original_position[0]) ** 2 +
                                 (current_position[1] - self.__original_position[1]) ** 2)

        safe_pixels = normalized_depth_image[normalized_depth_image > 0.35].shape[0]
        danger_pixels = normalized_depth_image[normalized_depth_image <= 0.35].shape[0]
 
        if self.hasTakeOff:     
            if current_distance - self.previous_distance > 0.005:
                reward += current_distance + safe_pixels * 0.0001
                self.previous_distance = current_distance

        if danger_pixels >= 1800:
            reward = -danger_pixels * 0.0001
        
        if collided:
            reward = -8
            print("Collided")
            done = True
        elif self.__gps.getValues()[0] < -4:
            reward = 10
            print("finished")
            done = True
            
            



        return self.state, reward, done, False, {}
    


from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from custom_policy import CustomPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_feature_extractor import CustomFeatureExtractorCNNLSTM
import os
import torch


def main():
    num_frames = 7

    env = OpenAIGymEnvironmentCNNLSTM(number_of_stack_images=num_frames)
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
        features_extractor_class=CustomFeatureExtractorCNNLSTM,
        features_extractor_kwargs=dict(
            features_dim=64,
            hidden_size=256,
            num_layers=4,
            num_frames=num_frames
        ),
    )


    model = PPO(
        'MlpPolicy', 
        env, 
        n_steps=1000, 
        verbose=1,
        learning_rate=0.0003,
        clip_range=0.35,
        ent_coef=0.0001, 
        tensorboard_log="./PPO_Policy_Mavic", 
        policy_kwargs=policy_kwargs,
        batch_size=50
    )
    total_timestep = 2000000

    log_dir = "results/train6/"
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    model.learn(total_timesteps=total_timestep, progress_bar=True, callback=checkpoint_callback, tb_log_name="CNN-LSTM")
    model.save('CNN-LSTM-Policy')

    # model = PPO.load("results/train1/rl_model_10000_steps")
    

    obs, info = env.reset()
    for _ in range(100000):
        action, _ = model.predict(obs, deterministic=True)
        # feature = model.policy.extract_features(torch.as_tensor(obs).unsqueeze(0))
        # print(feature)
        # action = random.randint(0,7)
        # action = 0
        obs, reward, done, truncated, info = env.step(action)
        print(reward)
        if done:
            obs, info = env.reset()

if __name__ == '__main__':
    main()




