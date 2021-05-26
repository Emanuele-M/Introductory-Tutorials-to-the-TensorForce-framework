# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


"""import gym
print(gym.envs.registry.all())
env = gym.make('FrozenLake8x8-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()"""

import tensorforce as tf

env = tf.environments.OpenAIGym('CartPole-v0', visualize=True)
my_env = tf.Environment.create(env)
my_agent = tf.Agent.create(agent = 'tensorforce', environment= my_env, update = 64,
                           optimizer =dict(optimizer= 'adam', learning_rate = 1e-3),
                           objective = 'policy_gradient', reward_estimation= dict(horizon= 1))
#Training

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
        actions, internals = my_agent.act(states = states, internals = internals, independent=True)
        ep_actions.append(actions)
        states, terminal, reward = my_env.execute(actions=actions)
        ep_terminal.append(terminal)
        ep_rewards.append(reward)
    my_agent.experience(states=ep_states, internals=ep_internals, actions=ep_actions, reward=ep_rewards, terminal=ep_terminal)
    my_agent.update()

#Evaluation

sum_reward = 0.0
for t in range (100):
    states = my_env.reset()
    internals = my_agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = my_agent.act(states=states, internals=internals, independent=True)
        states, terminal, reward = my_env.execute(actions=actions)
        sum_reward += reward

print('Mean evaluation reward:', sum_reward/100.0)
my_env.close()
my_agent.close()


"""runner = tf.execution.Runner(my_agent, my_env)

runner.run(num_episodes=200)
runner.run(num_episodes=100, evaluation=True)"""
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
