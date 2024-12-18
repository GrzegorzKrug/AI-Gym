# rl_tutorial.tf2.0.py
#
# deep reinforcement learning (DRL)

# TensorFlow's eager execution is an imperative programming environment
# that evaluates operations immediately, without building graphs: operations
# return concrete values instead of constructing a computational graph to run later.
# This makes it easy to get started with TensorFlow and debug models

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
# import random
import gym
import numpy as np
# import math
# import matplotlib.pyplot as plt


# Checking TF version, its ok
tf.__version__  # '2.0.0-rc0'
tf.executing_eagerly()  # True


# Note that we’re now in eager mode by default!
print("1 + 2 + 3 + 4 + 5 =", tf.reduce_sum([1, 2, 3, 4, 5]))
# 1 + 2 + 3 + 4 + 5 = tf.Tensor(15, shape=(), dtype=int32)


# Deep Actor-Critic Methods
#
# While much of the fundamental RL theory was developed on the tabular cases,
# modern RL is almost exclusively done with function approximators,
# such as artificial neural networks. Specifically, an RL algorithm is considered “deep”
# if the policy and value functions are approximated with deep neural networks.


# (Asynchronous) Advantage Actor-Critic
#
# Over the years, a number of improvements have been added to address sample efficiency
# and stability of the learning process.
#
# First, gradients are weighted with returns: discounted future rewards,
# which somewhat alleviates the credit assignment problem, and resolves theoretical issues
# with infinite timesteps.
#
# Second, an advantage function is used instead of raw returns.
# Advantage is formed as the difference between returns
# and some baseline (e.g. state-action estimate)
# and can be thought of as a measure of how good a given action is compared to some average.
#
# Third, an additional entropy maximization term is used in objective
# function to ensure agent sufficiently explores various policies.
# In essence, entropy measures how random a probability distribution is,
# maximized with uniform distribution.
#
# Finally, multiple workers are used in parallel to speed up sample gathering
# while helping decorrelate them during training.


# Incorporating all of these changes with deep neural networks
# we arrive at the two of the most popular modern algorithms:
# (asynchronous) advantage actor critic,
# or A3C/A2C for short.
# The difference between the two is more technical than theoretical:
# as the name suggests, it boils down to how the parallel workers estimate their gradients
# and propagate them to the model.


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # separate hidden layers from the same input tensor
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

# import gym
#
# env = gym.make('CartPole-v0')
# model = Model(num_actions=env.action_space.n)
#
# obs = env.reset()
# # no feed_dict or tf.Session() needed at all
# action, value = model.action_value(obs[None, :])
# print(action, value) # [1] [-0.00145713]

class A2CAgent:
    def __init__(self, model):
        # hyperparameters for loss terms
        # self.params = {'value': 0.5, 'entropy': 0.0001}
        self.params = {'value': 0.5, 'entropy': 0.0001, 'gamma': 0.99}
        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=0.0007),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )

    def test(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward

    def train(self, env, batch_sz=32, updates=1000):
        # storage helpers for a single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)
        # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs = env.reset()

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
        return ep_rews

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value']*kls.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy']*entropy_loss


env = gym.make('CartPole-v0')
model = Model(num_actions=env.action_space.n)
obs = env.reset()

# no feed_dict or tf.Session() needed at all
action, value = model.action_value(obs[None, :])
print(action, value)  # [1] [-0.00145713]

agent = A2CAgent(model)
# rewards_sum = agent.test(env)
# print("%d out of 200" % rewards_sum)  # 18 out of 200

rewards_history = agent.train(env)
print("Finished training, testing...")
print("%d out of 200" % agent.test(env))  # 200 out of 200
