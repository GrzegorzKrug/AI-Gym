import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import os

env = gym.make("MountainCar-v0")
run_num = 21

LEARNING_RATE = 0.05
# Discount should be not less than 1! Due to numeric loss
DISCOUNT = 1  # weight, how important are future action over current
EPISODES = 50000
EPISODE_OFFSET = 50000


SHOW_EVERY = EPISODES // 6
TIME_FRAME = 2000
STATE_SPACES = 40

EPS_ON = True
EPS_TOGGLE = True
EPS_INVERVAL = 1000

DISCRETE_OBS_SIZE = [STATE_SPACES] * len(env.observation_space.high)
discrete_obs_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

eps = 0.3  # not a constant, going to be decayed
END_EPS = 0.005
START_EPSILON_DECAYING = 0 + EPISODE_OFFSET
END_EPSILON_DECAYING = (EPISODES) // 2 + EPISODE_OFFSET

q_table = np.random.uniform(low=-20, high=-1, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))
q_table = np.load('qtables_20/49990-qtable.npy')

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': [], 'eps': []}


with open('run_params.txt', 'at') as file:
    file.write(f"RUN: {run_num:>3d}, Episodes: {EPISODES:>6d}, Discount: {DISCOUNT:>4.2f}, Learning-rate: {LEARNING_RATE:>4.2f}, "
               f"Spaces: {STATE_SPACES:>3d}, "
               f"Eps-init: {eps:>2.4f}, Eps-end: {END_EPS:>2.4f}, Eps-decay-at: {END_EPSILON_DECAYING:>6d}, "
               f"Timeframe: {TIME_FRAME:>6d}, Eps-toggle: {str(EPS_TOGGLE):>6}")
    file.write('\n')

os.mkdir(f"qtables_{run_num}")


def get_discrete_state(state):
    dc_state = (state - env.observation_space.low) / discrete_obs_win_size
    return tuple(dc_state.astype(np.int))


def eps_function(start, stop, n):
    for this_eps in np.linspace(start, stop, n):
        yield this_eps


eps_iterator = iter(np.linspace(eps, 0, END_EPSILON_DECAYING - START_EPSILON_DECAYING))


for episode in range(0 + EPISODE_OFFSET, EPISODES+EPISODE_OFFSET):
    _episode_reward = 0
    _reached = False

    if not episode % EPS_INVERVAL:
        if EPS_TOGGLE:
            EPS_ON ^= True

    if EPS_ON:
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            try:
                eps = next(eps_iterator)
                eps += END_EPS
            except StopIteration:
                eps = END_EPS
    else:
        eps = 0

    if not episode % SHOW_EVERY:
        render = True
    else:
        render = False

    # if episode == EPISODES - 1:
    #     render = True
    #     input("Press to show final agent...")

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
            time.sleep(0.009)  # Personal settings, to slow down animation

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
        aggr_ep_rewards['eps'].append(eps)

        print(f"Episode[{run_num:^3}]: {episode:>6d}, reward: {_episode_reward:>6.1f}, "
              f"average: {average_reward:>6.1f}, epsilon: {eps:>5.3f}, "
              f"min: {aggr_ep_rewards['min'][-1]:>6.1f}, max: {aggr_ep_rewards['max'][-1]:>6.1f}, "
              f"goal: {str(_reached):>5s}")

        np.save(f"qtables_{run_num}/{episode}-qtable.npy", q_table)

np.save(f"qtables_{run_num}/aggregated.npy", aggr_ep_rewards)
np.save(f"qtables_{run_num}/rewards.npy", ep_rewards)


plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=2)
plt.show()
