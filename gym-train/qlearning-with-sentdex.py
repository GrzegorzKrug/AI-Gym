import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import os

env = gym.make("MountainCar-v0")
run_num = 6

LEARNING_RATE = 0.1
DISCOUNT = 0.90  # weight, how important are future action over current
EPISODES = 25000

SHOW_EVERY = EPISODES // 5
TIME_FRAME = 500

DISCRETE_OBS_SIZE = [30] * len(env.observation_space.high)
discrete_obs_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

eps = 0.4  # not a constant, going to be decayed
END_EPS = 0.05
START_EPSILON_DECAYING = 0
END_EPSILON_DECAYING = EPISODES * 3 // 4

os.mkdir(f"qtables_{run_num}")

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


def get_discrete_state(state):
    dc_state = (state - env.observation_space.low) / discrete_obs_win_size
    return tuple(dc_state.astype(np.int))


class EpsIterator:
    def __init__(self, start, stop, n):
        self.now = start
        self.start = start
        self.stop = stop
        self.n = n

    def __next__(self):
        for cur_eps in np.linspace(self.start, self.stop, self.n):
            self.now = cur_eps
            print('Next')
            yield self.now

    def __iter__(self, *args):
        print(f'Iter, args: {args}')
        print(f"Now: {self.now}")
        return self


def eps_function(start, stop, n):
    for this_eps in np.linspace(start, stop, n):
        yield this_eps


eps_iterator = iter(np.linspace(eps, 0, END_EPSILON_DECAYING - START_EPSILON_DECAYING))


for episode in range(EPISODES):
    _episode_reward = 0
    _reached = False

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        try:
            eps = next(eps_iterator)
            eps += END_EPS
        except StopIteration:
            eps = END_EPS

    if not episode % SHOW_EVERY:
        render = True
        render = False
    else:
        render = False

    if episode == EPISODES - 1:
        render = True
        input("Press to show final agent...")

    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > eps:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        _episode_reward += reward

        if render:
            env.render()
            time.sleep(0.009)

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])  # highest value of q action
            current_q = q_table[discrete_state + (action, )]  # current q + action, q before movement
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action, )] = new_q

        elif new_state[0] >= env.goal_position:
            # print(f"We reached goal at: {episode}")
            _reached = True
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
    if render:
        env.close()

    ep_rewards.append(_episode_reward)

    if not episode % 10:
        average_reward = sum(ep_rewards[-TIME_FRAME:]) / len(ep_rewards[-TIME_FRAME:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-TIME_FRAME:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-TIME_FRAME:]))

        print(f"Episode: {episode:>4d}, reward: {_episode_reward:>5.1f}, "
              f"average: {average_reward:>4.1f}, epsilon: {eps:>5.3f}, "
              f"min: {aggr_ep_rewards['min'][-1]:>4.1f}, max: {aggr_ep_rewards['max'][-1]:>4.1f}, "
              f"reached: {str(_reached):>5s}")

        np.save(f"qtables_{run_num}/{episode}-qtable.npy", q_table)

np.save(f"qtables_{run_num}/aggregated.npy", aggr_ep_rewards)


plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=2)
plt.show()
