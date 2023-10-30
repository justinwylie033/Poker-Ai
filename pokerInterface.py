import numpy as np
from pokerEnv import FiveCardDrawEnvironment
from pokerAgent import DQNAgent

class PokerInterface:
    def __init__(self, agent):
        self.agent = agent
        self.env = FiveCardDrawEnvironment()
        self.state = self.env.get_state_representation(self.env)
        self.agent.load("poker-dqn.h5")

    def play(self):
        done = False
        while not done:
            self.display_state()
            if self.env.current_better == 0:  # Assuming player 0 is the user
                action = self.get_user_action()
            else:
                action = self.agent.act(self.state)
                print(f"AI chose action: {action}")
            self.state, _, done, _ = self.env.step(action)

        print("Game Over!")
        self.display_state()

    def get_user_action(self):
        action = None
        while action not in [0, 1, 2]:  # Assuming 0: Bet, 1: Call, 2: Fold
            action = int(input("Choose an action (0: Bet, 1: Call, 2: Fold): "))
        return action

    def display_state(self):
        print("Your Hand: ", self.env.players[0].hand)
        print("AI Hand: ", self.env.players[1].hand)
        print("Your Chip Stack: ", self.env.players[0].chip_stack)
        print("Current Pot: ", self.env.pot)
        print("Current Bets: ", self.env.bets)
0
if __name__ == "__main__":
    state_size = 107  # Update based on your state representation
    action_size = 3   # Bet, Call, Fold for simplicity
    agent = DQNAgent(state_size, action_size)
    interface = PokerInterface(agent)
    interface.play()
