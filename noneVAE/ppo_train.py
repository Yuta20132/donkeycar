import argparse
import uuid
import gym_donkeycar
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os 
import cv2
from tqdm import tqdm
import psutil


gym.logger.set_level(40)
if __name__ == "__main__":

    log_dir = './log/'
    os.makedirs(log_dir,exist_ok=True)

    log1_dir = './log_board/'
    os.makedirs(log1_dir, exist_ok=True)

    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0",
        "donkey-roboracingleague-track-v0",
        "donkey-waveshare-v0",
        "donkey-minimonaco-track-v0",
        "donkey-warren-track-v0",
        "donkey-thunderhill-track-v0",
        "donkey-circuit-launch-track-v0",
    ]

    parser = argparse.ArgumentParser(description="ppo_train")
    parser.add_argument(
        "--sim",
        type=str,
        default="sim_path",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
    )
    parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
    parser.add_argument("--test", action="store_true", help="load the trained model and play")
    parser.add_argument("--multi", action="store_true", help="start multiple sims at once")
    parser.add_argument(
        "--env_name", type=str, default="donkey-waveshare-v0", help="name of donkey sim environment", choices=env_list
    )

    args = parser.parse_args()

    if args.sim == "sim_path" and args.multi:
        print("you must supply the sim path with --sim when running multiple environments")
        exit(1)

    env_id = args.env_name

    conf = {
        "exe_path": "C:\\Users\yutaxxx\projects\DonkeySimWin\donkey_sim.exe",
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "me",
        "font_size": 100,
        "racer_name": "PPO",
        "country": "USA",
        "bio": "Learning to drive w PPO RL",
        "guid": str(uuid.uuid4()),
        "max_cte": 5,
    }

    if args.test:

        # Make an environment test our trained policy
        env = gym.make(args.env_name, conf=conf)
        #env = gym.make("donkey-generated-track-v0")

        model = PPO.load("ppo_donkey")

        obs = env.reset()
        i = 0
        for _ in range(5000):
            action, _states = model.predict(obs, deterministic=True)

            
            #print("観測空間の型     :", env.observation_space)
            #print("観測空間のサイズ :", env.observation_space.shape)
            #print("行動空間の型     :", env.action_space)
            #print("報酬範囲         :", env.reward_range)

            #cv2.imwrite('data_generated_track/'+str(i)+'.jpg', obs)
            #i += 1
            obs, reward, done, info = env.step(action)
            env.render()
            
            if done:
                obs = env.reset()

        print("done testing")

    else:

        # make gym env
        env = gym.make(args.env_name, conf=conf)
        #Monitorの使用
        env = Monitor(env,log_dir,allow_early_resets=True)
        # create cnn policy
        model = PPO("CnnPolicy", env,n_steps=256, verbose=1,tensorboard_log=log1_dir)

        # set up model in learning mode with goal number of timesteps to complete
        model.learn(total_timesteps=100000)

        obs = env.reset()

        for i in tqdm(range(1000)):

            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, done, info = env.step(action)
            
            #memory = psutil.virtual_memory()
            #f = open("memory_noneVAE.txt","a")
            #f.write(str(memory.used)+"\n")
            #f.close()


            try:
                env.render()
            except Exception as e:
                print(e)
                print("failure in render, continuing...")

            if done:
                obs = env.reset()

            if i % 100 == 0:
                print("saving...")
                model.save("ppo_donkey")

        # Save the agent
        model.save("ppo_donkey")
        print("done training")

    env.close()