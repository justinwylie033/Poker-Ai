from pokerEnv import FiveCardDrawEnvironment
from pokerAgent import DQNAgent, print_statistics

def evaluate_agent(episodes=1000):
    env = FiveCardDrawEnvironment()
    state_size = 107
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    agent.load("poker-dqn.h5")
    agent.epsilon = 0  # Set epsilon to 0 to disable exploration
    rewards = []

    for e in range(episodes):
        env.reset()
        state = agent.get_state_representation(env)
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = agent.get_state_representation(env)
        rewards.append(total_reward)
        print("episode: {}/{}, reward: {}".format(e, episodes, total_reward))

    print_statistics(rewards)

if __name__ == "__main__":
    evaluate_agent()