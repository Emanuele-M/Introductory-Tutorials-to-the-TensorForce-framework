# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import tensorforce as tf
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
        internals = agent.initial_internals()
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
    tr_sum = tr_sum/num_episodes
    return tr_sum

def evaluate(agent, environment, num_episodes):
    ev_sum = 0.0
    for t in range(num_episodes):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent.act(states=states, internals=internals, independent=True)
            print(actions)
            states, terminal, reward = environment.execute(actions=actions)
            ev_sum += reward
    ev_sum = ev_sum/num_episodes
    return ev_sum


max_time = 450
my_env = tf.Environment.create(environment='gym', level='LunarLander-v2', max_episode_timesteps=max_time, visualize=True)
actionSpace = my_env.actions()
n_actions = actionSpace.get('num_values')
network_spec = [dict(type='dense', size=10, activation='relu'),
                dict(type='dense', size=10, activation='relu'),
                dict(type='dense', size=n_actions, activation='softmax')]

my_agent = tf.Agent.create(agent='reinforce', environment=my_env, batch_size=8, network=network_spec, memory=2000000,
                           learning_rate=5e-4, discount=0.99, exploration=5e-4)


my_agent.reset()
train_rewards = []
rewards = []
for i in range(150):
    # Training
    tr = train(my_agent, my_env, 150)
    print("Training: ", tr)
    print('Cleared training ', i)
    train_rewards.append(tr)
    # Evaluation
    ev = evaluate(my_agent, my_env, 50)
    print(str(ev))
    print('Cleared epoch ', i)
    rewards.append(ev)

my_env.close()
my_agent.close()

plt.plot(rewards)
plt.xlabel('epoch')
plt.ylabel('avg_rewards_eval')
plt.show()
plt.plot(train_rewards)
plt.xlabel('epoch')
plt.ylabel('avg_rewards_train')
plt.show()
