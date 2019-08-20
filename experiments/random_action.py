import MiningEnv
import gym
from pdb import set_trace


def main():
    # Test rendering effect
    env = gym.make('MiningEnv-v0')

    for i in range(100000):
        done = False
        rewards = 0
        env.reset()
        while not done:
            truck_action = env.action_space[0].sample()
            excvtr_action = env.action_space[1].sample()
            action = [truck_action, excvtr_action]
            action = [np.array([np.nan, np.nan, ])]
            observation, reward, done_n, info = env.step(action)
            done = all(done_n)
            rewards += reward[0]
        if i % 500 == 0:
            print('Episode {}, Reward {}'.format(i, rewards))
    env.close()


if __name__ == "__main__":
    main()
