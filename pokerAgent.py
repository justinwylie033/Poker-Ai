import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
from pokerEnv import FiveCardDrawEnvironment

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)  # Set memory size to 100,000
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.00 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.0000461
        self.learning_rate = 0.0005
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=107, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(2000, activation='relu'))
        model.add(Dense(2000, activation='relu'))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def get_state_representation(self, env):
        state = []
        for player_id, player in env.players.items():
            hand_vector = [0] * 52
            for card in player.hand:
                index = self.card_to_index(card)
                hand_vector[index] = 1
            state.extend(hand_vector)
            state.append(player.chip_stack)
        state.append(env.pot)
        return np.reshape(np.array(state), [1, -1])

    def card_to_index(self, card):
        rank = '23456789TJQKA'.index(card['rank'])
        suit = '♠♥♦♣'.index(card['suit'])
        return rank * 4 + suit

# Training Loop
def train_agent(episodes=1000, batch_size=32):
    env = FiveCardDrawEnvironment()
    state_size = 107  # Update based on your state representation
    action_size = 3   # Bet, Call, Fold for simplicity
    agent = DQNAgent(state_size, action_size)
    done = False
    rewards = []

    for e in range(episodes):
        env.reset()
        state = agent.get_state_representation(env)
        total_reward = 0
        for time in range(500):  # Adjust the maximum number of steps as needed
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state = agent.get_state_representation(env)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        rewards.append(total_reward)
        print("episode: {}/{}, reward: {}, e: {:.2}"
              .format(e, episodes, total_reward, agent.epsilon))
        if e % 10 == 0 and e != 0:
            print_statistics(rewards)
            agent.save("poker-dqn.h5")


def print_statistics(rewards):
    # Calculate win percentage
    wins = sum(r > 0 for r in rewards)
    win_percentage = wins / len(rewards) * 100
    
    # Calculate average reward
    average_reward = sum(rewards) / len(rewards)

    print("\n--- Statistics ---")
    print("Win Percentage: {:.2f}%".format(win_percentage))
    print("Average Reward: {:.2f}".format(average_reward))
    # Add any other statistics you find interesting here
    print("------------------\n")

if __name__ == "__main__":
    train_agent(episodes=5000)