import random
import numpy as np

class Player:
    def __init__(self, chip_stack):
        self.hand = []
        self.chip_stack = chip_stack

class HandEvaluator:
    rank_names = [
        "High Card", "One Pair", "Two Pair", "Three of a Kind", 
        "Straight", "Flush", "Full House", "Four of a Kind", 
        "Straight Flush"
    ]
    
    @staticmethod
    def evaluate_five_card_hand(hand):
        ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        sorted_hand = sorted(hand, key=lambda x: (ranks[x['rank']], x['suit']))
        flush = all(card['suit'] == sorted_hand[0]['suit'] for card in sorted_hand)
        rank_values = [ranks[card['rank']] for card in sorted_hand]
        straight = all(rank_values[i] == rank_values[i - 1] + 1 for i in range(1, 5)) or rank_values == [0, 1, 2, 3, 12]
        
        if flush and straight:
            return (8, sorted_hand) if rank_values != [0, 1, 2, 3, 12] else (8, sorted_hand[1:] + [sorted_hand[0]])
        rank_counts = {rank: rank_values.count(rank) for rank in set(rank_values)}
        sorted_rank_counts = sorted(rank_counts.items(), key=lambda x: (-x[1], -x[0]))
        if sorted_rank_counts[0][1] == 4:
            return (7, sorted_hand)
        if sorted_rank_counts[0][1] == 3:
            if sorted_rank_counts[1][1] == 2:
                return (6, sorted_hand)
            return (3, sorted_hand)
        if flush:
            return (5, sorted_hand)
        if straight:
            return (4, sorted_hand)
        if sorted_rank_counts[0][1] == 2:
            if sorted_rank_counts[1][1] == 2:
                return (2, sorted_hand)
            return (1, sorted_hand)
        return (0, sorted_hand)

    
    def compare_hands(hand1, hand2):
        rank1, sorted_cards1 = hand1
        rank2, sorted_cards2 = hand2
        
        if rank1 > rank2:
            return 1
        elif rank1 < rank2:
            return -1
        else:
            # If the ranks are the same, compare the sorted cards
            for card1, card2 in zip(sorted_cards1, sorted_cards2):
                if card1['rank'] > card2['rank']:
                    return 1
                elif card1['rank'] < card2['rank']:
                    return -1
            return 0


class FiveCardDrawEnvironment:
    def __init__(self, num_players=2, starting_chip_stack=1000):
        self.num_players = num_players
        self.players = {i: Player(starting_chip_stack) for i in range(self.num_players)}
        self.deck = self.initialize_deck()
        self.pot = 0
        self.betting_round = True
        self.active_players = set(range(num_players))
        self.current_better = 0
        self.last_raise = 0
        self.bets = {i: 0 for i in range(self.num_players)}
        self.minimum_bet = 10
        self.start_new_round()
        self.drawing_round = False  # Indicates whether it's the drawing round


    def reset(self):
        self.start_new_round()

    def initialize_deck(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['♠', '♥', '♦', '♣']
        deck = [{'rank': rank, 'suit': suit} for rank in ranks for suit in suits]
        random.shuffle(deck)
        return deck
    
    def deal_cards(self):
        for player_id in self.active_players:
            self.players[player_id].hand = [self.deck.pop() for _ in range(5)]
    
    def start_new_round(self):
        self.drawing_round = False
        self.deck = self.initialize_deck()
        self.pot = 0
        self.betting_round = True
        self.active_players = set(range(self.num_players))
        self.current_better = 0
        self.last_raise = 0
        self.bets = {i: 0 for i in range(self.num_players)}
        self.deal_cards()

    def start_drawing_round(self):
        self.drawing_round = True
        self.current_better = 0
        self.bets = {i: 0 for i in range(self.num_players)}
    
    def bet(self, player_id, amount):
        if player_id not in self.active_players:
            return "Invalid action: Player is not active"
        if not self.betting_round:
            return "Invalid action: Cannot bet during drawing round"
        if amount < self.minimum_bet:
            return f"Invalid bet: Minimum bet is {self.minimum_bet} chips"
        player = self.players[player_id]
        if amount > player.chip_stack:
            return "Invalid bet: Player does not have enough chips"
        player.chip_stack -= amount
        self.pot += amount
        self.bets[player_id] += amount
        self.last_raise = amount
        self.current_better = (player_id + 1) % self.num_players
        return f"Player {player_id} bets {amount}"
    
    def call(self, player_id):
        if player_id not in self.active_players:
            return "Invalid action: Player is not active"
        if not self.betting_round:
            return "Invalid action: Cannot call during drawing round"
        amount_to_call = max(self.bets.values()) - self.bets[player_id]
        return self.bet(player_id, amount_to_call)
    
    def raise_bet(self, player_id, amount):
        if player_id not in self.active_players:
            return "Invalid action: Player is not active"
        if not self.betting_round:
            return "Invalid action: Cannot raise during drawing round"
        amount_to_call = max(self.bets.values()) - self.bets[player_id]
        total_amount = amount_to_call + amount
        return self.bet(player_id, total_amount)
    
    def fold(self, player_id):
        if player_id not in self.active_players:
            return "Invalid action: Player is not active"
        if not self.betting_round:
            return "Invalid action: Cannot fold during drawing round"
        self.active_players.remove(player_id)
        self.current_better = (player_id + 1) % self.num_players
        return f"Player {player_id} folds"
    
    def draw_cards(self, player_id, cards_to_discard):
        if not self.drawing_round:
            return "Invalid action: Can only draw cards during drawing round"
        if player_id not in self.active_players:
            return "Invalid action: Player is not active"
        if self.betting_round:
            return "Invalid action: Can only draw cards during drawing round"
        player = self.players[player_id]
        for card in cards_to_discard:
            if card not in player.hand:
                return "Invalid action: Card not in hand"
            player.hand.remove(card)
            player.hand.append(self.deck.pop())
        return f"Player {player_id} draws {len(cards_to_discard)} cards"

    def showdown(self):
        if len(self.active_players) == 1:
            winner = self.active_players.pop()
            self.players[winner].chip_stack += self.pot
            return winner  # Return the winner's identifier
        best_hand = None
        winner = None
        for player_id in self.active_players:
            current_hand = HandEvaluator.evaluate_five_card_hand(self.players[player_id].hand)
            if best_hand is None or HandEvaluator.compare_hands(current_hand, best_hand) > 0:
                best_hand = current_hand
                winner = player_id
        self.players[winner].chip_stack += self.pot
        return winner  # Return the winner's identifier

            # State Representation Functions
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




    def step(self, action):
        player_id = self.current_better
        reward = 0
        done = False
        info = {}

        if self.drawing_round:
            if action == 3:  # Draw cards
                # Here you would need to specify which cards the player wants to draw
                cards_to_draw = []  # Add logic to determine which cards to draw
                self.draw_cards(player_id, cards_to_draw)
            else:
                return self.get_state_representation(self), 0, False, {"error": "Invalid action during drawing round"}
        else:
            if action == 0:  # Bet
                self.bet(player_id, 10)  # Betting a fixed amount for simplicity
            elif action == 1:  # Call
                self.call(player_id)
            elif action == 2:  # Fold
                self.fold(player_id)
            else:
                return self.get_state_representation(self), 0, False, {"error": "Invalid action during betting round"}

        # Check if the betting round is over
        if self.is_betting_round_over():
            if self.drawing_round:
                done = True
                reward = self.end_round()  # Calculate reward based on the outcome of the round
            else:
                self.start_drawing_round()

        new_state = self.get_state_representation(self)
        return new_state, reward, done, info

    def is_betting_round_over(self):
        if not self.betting_round:
            print("Betting round is not active.")
            return False
        
        highest_bet = max(self.bets.values())
        if len(self.active_players) == 1:
            print("Only one player remains.")
            return True
        
        for player_id in self.active_players:
            player_bet = self.bets[player_id]
            player_stack = self.players[player_id].chip_stack
            if player_bet < highest_bet and player_stack > 0:
                print(f"Player {player_id} has not matched the highest bet.")
                return False
        
        print("Betting round is over.")
        return True



    def end_round(self):
        # Determine the winner and calculate the reward
        winner = self.showdown()
        reward = 0
        if winner == 0:  # Assuming player 0 is our agent
            reward = +1
        else:
            reward = -1
        self.start_new_round()  # Start a new round after the game is over
        return reward

