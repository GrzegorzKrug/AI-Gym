import texttable as tt
import numpy as np
import random  # random
import time
import sys
import os

import card_settings


class GameCards98:
    """
    Piles    1: GoingUp 2: GoingUp'
             3: GoingDown 4: GoingDown'
    Input: hand_number, pile number ; Separator is not necessary'
    """

    def __init__(self, timeout_turn=1000):
        """

        Args:
            timeout_turn:
        """
        self._reset()

        self.timeout_turn = timeout_turn
        self.translator = MapIndexesToNum(4, 8)

        'Rewards'
        self.WIN = 0
        self.SkipMove = card_settings.SKIP_MOVE
        self.GoodMove = card_settings.GOOD_MOVE
        self.EndGame = card_settings.LOST_GAME
        self.InvalidMove = card_settings.INVALID_MOVE

    def reset(self):
        """
        Reset game
        Returns:
            state
        """
        self._reset()
        obs = self.observation()
        return obs

    def _reset(self):
        self.piles = [1, 1, 100, 100]
        self.deck = random.sample(range(2, 100), 98)  # 98)
        self.hand = []
        self.move_count = 0
        self.turn = 0

        self.score = 0
        self.score_gained = 0
        self.last_card_played = 0
        self.history = []
        self.hand_fill()

    def calculate_chance_10(self, cards, round_chance=True):
        """
        Check propabality of playing Card Higher or lower by 10
        Args:
            cards:
            round_chance:

        Returns:

        """
        lower_card_chance = []
        higher_card_chance = []

        if len(self.deck) > 0:
            chance = round(1 / len(self.deck) * 100, 2)
            if round_chance:
                chance = round(chance)
            chance = chance
        else:
            chance = 0

        for card in cards:
            # Checking for cards in deck -> Chance %
            # Checking for cards in hand -> 100%
            # Not Checking piles
            if card - 10 in self.deck:
                lower_card_chance.append(chance)
            # elif (card - 10 in self.piles[2:4]) or card - 10 in self.hand:
            elif card - 10 in self.hand:
                lower_card_chance.append(100)
            else:
                lower_card_chance.append(0)

            if card + 10 in self.deck:
                higher_card_chance.append(chance)
            # elif card + 10 in self.piles[0:2] or card + 10 in self.hand:
            elif card + 10 in self.hand:
                higher_card_chance.append(100)
            else:
                higher_card_chance.append(0)
        return [lower_card_chance, higher_card_chance]

    def conv_piles_to_array(self):
        out = np.array(self.piles.copy())
        out = out / 100
        return out

    def conv_dec_to_array(self):
        cards = np.zeros(98, dtype=int)
        for card_index in self.deck:
            cards[card_index - 2] = 1  # Card '2' is first on list, index 0
        return np.array(cards)

    def conv_hand_to_array(self):
        cards = np.zeros((8, 98), dtype=int)
        for index, card_index in enumerate(self.hand):
            cards[index, card_index - 2] = 1  # Card '2' is first on list, index 0
        cards = cards.ravel()
        return cards

    def check_move(self, hand_id, pile_id):
        """
        Method Checks if move is proper
        Returns True if valid
        Returns False if invalid
        Used in checking for End Conditions
        Copied from Play Card Method
        """
        response = {}.fromkeys(['invalid', 'valid', 'skip'], False)
        valid = False

        if hand_id < 0 or hand_id > 7:
            "Invalid move"

        elif pile_id < 0 or pile_id > 3:
            "Invalid move"

        elif pile_id == 0 or pile_id == 1:
            try:
                if self.hand[hand_id] > self.piles[pile_id]:
                    valid = True
                elif self.hand[hand_id] == (self.piles[pile_id] - 10):
                    valid = True
                    response['skip'] = True
            except IndexError:
                "Invalid move"

        elif pile_id == 2 or pile_id == 3:
            try:
                if self.hand[hand_id] < self.piles[pile_id]:
                    valid = True
                elif self.hand[hand_id] == (self.piles[pile_id] + 10):
                    valid = True
                    response['skip'] = True
            except IndexError:
                "Invalid move"

        if valid:
            response['valid'] = True
            return True, response
        else:
            response['invalid'] = True
            return False, response

    def display_table(self, show_chances=False, show_deck=False):
        """
        Showing Table.
        Showing Hand.
        Showing Chances of next Cards.
        """
        print('\n' + '=' * 5, 'Turn'.center(8), '=', self.move_count)
        print('=' * 5, 'Score'.center(8), '=', self.score)
        if show_deck:
            print('Deck (cheating) :', self.deck)

        piles = tt.Texttable()
        piles.add_row(['↑ Pile ↑', '1# ' + str(self.piles[0]), '2# ' + str(self.piles[1])])
        piles.add_row(['↓ Pile ↓', '3# ' + str(self.piles[2]), '4# ' + str(self.piles[3])])
        print(piles.draw())

        hand = tt.Texttable()
        lower_chance, higher_chance = self.calculate_chance_10(self.hand)

        if show_chances:
            lower_chance_row = [str(i) + '%' for i in lower_chance]  # Making text list, Adding % to number
            higher_chance_row = [str(i) + '%' for i in higher_chance]  # Making text list, Adding % to number
            hand.add_row(['Lower Card Chance'] + lower_chance_row)

        hand_with_nums = [str(i + 1) + '# ' + str(j) for i, j in enumerate(self.hand)]  # Numerated Hand
        hand.add_row(['Hand'] + hand_with_nums)

        if show_chances:
            hand.add_row(['Higher Card Chance'] + higher_chance_row)
        print(hand.draw())

    def end_condition(self):
        """
        Checking end conditions after current move
            Returns:

            end_game: Dict[str, bool],  keys = [loss, win, end, other]

            end_bool: boolean value if game has ended

            reward:
        """
        info = {}.fromkeys(['loss', 'win', 'timeout', 'other'], False)
        end_bool = False
        reward = None
        if self.move_count > self.timeout_turn:
            info['timeout'] = True
            end_bool = True
            return end_bool, info, self.EndGame

        next_move = None
        for hand_id in range(8):
            if next_move:
                break

            for pile_id in range(4):
                next_move, info = self.check_move(hand_id, pile_id)
                if next_move:
                    break

        if next_move:
            pass
        elif len(self.hand) == 0 and len(self.deck) == 0:
            info['win'] = True
            end_bool = True
            reward = self.WIN
            # print(f"Win!")
        else:
            info['loss'] = True
            end_bool = True
            reward = self.EndGame
        return end_bool, info, reward

    def hand_fill(self):
        """
        Fill Hand with cards from deck
        Hand is always 8
        """
        while len(self.hand) < 8 and len(self.deck) > 0:
            self.hand.append(self.deck[0])
            self.deck.pop(0)
        self.hand.sort()

    def step(self, action):
        """
        Play 1 card
        Args:
            action: Tuple(int, int)
        Returns:
            reward, new_state, done, info
        """
        valid, reward = self._play_card(action)
        if valid:
            done, info, end_reward = self.end_condition()
            if done:
                reward = end_reward
        else:
            info = {}.fromkeys(['loss', 'win', 'timeout', 'other'], False)
            info['other'] = True
            done = True

        new_state = self.observation()
        return reward, new_state, done, info

    def _play_card(self, action):
        """

        Args:
            action:

        Returns:

        """
        pile_id, hand_id = action
        self.move_count += 1
        self.score_gained = 0  # reset value

        try:
            valid, info = self.check_move(hand_id, pile_id)
            if valid:
                card = self.hand[hand_id]
                self.piles[pile_id] = card
                self.hand.pop(hand_id)

                if info['skip']:
                    reward = self.SkipMove
                else:
                    reward = self.GoodMove
                self.score += reward
                self.turn += 1
                self.hand_fill()
                return True, reward

            else:
                self.score += self.InvalidMove
                reward = self.InvalidMove
                return False, reward

        except IndexError as ie:
            print(f'INDEX ERROR: {ie}, {action}, {len(self.hand)}')
            self.score += self.InvalidMove
            reward = self.InvalidMove
            return False, reward

    def observation(self):
        """Return cards in deck(asumption we know play history) and in hand"""
        piles = self.conv_piles_to_array()
        hand = self.conv_hand_to_array()
        out = np.concatenate([piles, hand])
        return out


class MapIndexesToNum:
    """
    Maps 2-Dimensional arrays or higher to 1 number index and reverse
    """

    def __init__(self, *dimensions):
        self.shape = list(dimensions)
        self.dims = len(self.shape)

        ind = 1
        for dim in dimensions:
            ind *= dim
        self.max_ind = ind - 1

    def get_num(self, *indexes):
        if type(indexes[0]) is tuple:
            indexes = indexes[0]

        if len(indexes) != self.dims:
            raise ValueError("Dimensions number does not match")

        num = 0
        multi = 1
        for ind, size in zip(indexes, self.shape):
            if ind >= size:
                raise ValueError(f"Index must be lower than size: {ind} >= {size}")
            num += ind * multi
            multi *= size
        return num

    def get_map(self, input_index):
        """Returns tuple of indexes matching index"""
        if 0 <= input_index <= self.max_ind:
            num = int(input_index)
            out = []
            for size in self.shape:
                cur_in = num % size
                num = num // size
                out.append(cur_in)
            return tuple(out)

        else:
            raise ValueError("Index beyond range")


if __name__ == "__main__":
    game = GameCards98()
