# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorforce as tf
import matplotlib.pyplot as plt
max_time = 100
my_env = tf.Environment.create(environment='gym', level='MountainCarContinuous', max_episode_timesteps=max_time)

#VPG Agent direttamente definito come da documentazione di TensorForce

my_agent = tf.Agent.create(agent='reinforce', environment=my_env, batch_size=64, learning_rate=1e-3)

#TensorForce Agent configurato per avere come obiettivo il Policy Gradient

"""my_agent = tf.Agent.create(agent='tensorforce', environment=my_env, update=dict(unit='episodes', batch_size=64),
                           optimizer=dict(optimizer='adam', learning_rate=1e-3), objective='policy_gradient',
                           reward_estimation=dict(horizon=20, discount=0.99))"""

# Training
my_agent.reset()
rewards = []
for i in range(100):
    for episode in range(100):
        ep_states = list()
        ep_actions = list()
        ep_internals = list()
        ep_terminal = list()
        ep_rewards = list()
        states = my_env.reset()
        internals = my_agent.initial_internals()
        terminal = False
        while not terminal:
            ep_states.append(states)
            ep_internals.append(internals)
            action, internals = my_agent.act(states=states, internals=internals, independent=True)
            ep_actions.append(action)
            states, terminal, reward = my_env.execute(actions=action)
            ep_terminal.append(terminal)
            ep_rewards.append(reward)
        my_agent.experience(states=ep_states, internals=ep_internals, actions=ep_actions, reward=ep_rewards,
                            terminal=ep_terminal)
        my_agent.update()

    # Evaluation

    sum_reward = 0.0
    for t in range(100):
        states = my_env.reset()
        internals = my_agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = my_agent.act(states=states, internals=internals, independent=True)
            states, terminal, reward = my_env.execute(actions=actions)
            sum_reward += reward
    sum_reward = sum_reward/100.0
    print(str(sum_reward))
    rewards.append(sum_reward)
    print("Cleared training step ", i)

my_env.close()
my_agent.close()

plt.plot(rewards)
plt.xlabel('steps')
plt.ylabel('avg_rewards')
plt.show()


"""runner = tf.execution.Runner(my_agent, my_env)

runner.run(num_episodes=200)
runner.run(num_episodes=100, evaluation=True)"""
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
