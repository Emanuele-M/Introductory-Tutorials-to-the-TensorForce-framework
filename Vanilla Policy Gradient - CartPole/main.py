# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorforce as tf
import tensorflow as tfl
import matplotlib.pyplot as plt


def train(agent, environment, num_episodes):
    tr_sum = 0.0
    for episode in range(num_episodes):
        ep_states = list()
        ep_actions = list()
        ep_internals = list()
        ep_terminal = list()
        ep_rewards = list()
        states = environment.reset()
        internals=agent.initial_internals()
        terminal = False
        while not terminal:
            ep_states.append(states)
            ep_internals.append(internals)
            actions, internals = agent.act(states=states, internals=internals, independent=True, deterministic=False)
            ep_actions.append(actions)
            states, terminal, reward = environment.execute(actions=actions)
            ep_terminal.append(terminal)
            ep_rewards.append(reward)
            tr_sum += reward
        agent.experience(states=ep_states, actions=ep_actions, terminal=ep_terminal, reward=ep_rewards,
                            internals=ep_internals,)
        agent.update()
        print(str(tr_sum))
    return tr_sum

def evaluate(agent, environment, num_episodes):
    ev_sum = 0.0
    for t in range(num_episodes):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent.act(states=states, internals=internals, independent=True)
            states, terminal, reward = environment.execute(actions=actions)
            ev_sum += reward
        print(str(ev_sum))
    return ev_sum

max_time = 100
my_env = tf.Environment.create(environment='gym', level='CartPole', max_episode_timesteps=max_time, visualize=True)
actionSpace = my_env.actions()
print(str(actionSpace))
n_actions=actionSpace.get('num_values')
print(str(n_actions))

#VPG Agent direttamente definito come da documentazione di TensorForce

#Policy network specification, two fully connected intermediate layers and an output layer
network_spec = [dict(type='dense', size=64, activation='relu'),
                dict(type='dense', size=64, activation='relu'),
                dict(type='dense', size=n_actions, activation='softmax')]
#clip = dict(type='clipping', lower= 1e-8, upper=1-1e-8)
my_agent = tf.Agent.create(agent='reinforce', environment=my_env, batch_size=64, network=network_spec,
                           use_beta_distribution=True, memory=10000, learning_rate=5e-4, discount=0.99, baseline='auto',
                           baseline_optimizer=dict(optimizer='adam', learning_rate=5e-4))

#TensorForce Agent configurato per avere come obiettivo il Policy Gradient


#Train and evaluate the agent by using the act-experience-update TensorForce interface

# Training
my_agent.reset()
train_rewards = []
rewards = []

for i in range(100):
    training = train(my_agent, my_env, 100)
    training = training/100.0
    train_rewards.append(training)
    print(str(training))
    print('Training done at step ', i)
    # Evaluation
    sum_reward = evaluate(my_agent, my_env, 100)
    sum_reward=sum_reward/100.0
    print(str(sum_reward))
    rewards.append(sum_reward)
    print('Cleared step ', i)

my_env.close()
my_agent.close()

plt.plot(rewards)
plt.xlabel('steps')
plt.ylabel('avg_rewards_eval')
plt.show()
plt.plot(train_rewards)
plt.xlabel('steps')
plt.ylabel('avg_rewards_train')
plt.show()
"""runner = tf.execution.Runner(my_agent, my_env)

runner.run(num_episodes=200)
runner.run(num_episodes=100, evaluation=True)"""
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

"""my_agent = tf.Agent.create(agent='tensorforce', environment=my_env, update=dict(unit='episodes', batch_size=64),
                           optimizer=dict(optimizer='adam', learning_rate=1e-3), objective='policy_gradient',
                           reward_estimation=dict(horizon=20, discount=0.99))"""
