import argparse

import gym

import dyn_rl_benchmarks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Demo environment with random actions.")
    parser.add_argument("--env", default = "Platforms-v1", help = "Name of gym environment to be run.")
    args = parser.parse_args()

    env = gym.make(args.env)

    for n in range(10):
        obs = env.reset()
        cum_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action)
            env.render()
            cum_reward += rew
            if done:
                print("Episode return: {}".format(cum_reward))
