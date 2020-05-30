import random  # random
import re  # regex
import json
import numpy as np
import texttable as tt


class GameCards98:
    """
    Piles    1: GoingUp 2: GoingUp'
             3: GoingDown 4: GoingDown'
    Input: hand_number, pile number ; Separator is not necessary'
    """

    def __init__(self):
        """
        self.pile_going_up = [1, 1]
        self.pile_going_down = [100, 100]
        """
        self.piles = [1, 1, 100, 100]
        self.deck = random.sample(range(2, 100), 98)  # 98)
        self.hand = []
        self.move_count = 0
        self.turn = 0
        self.score = 0
        self.score_gained = 0
        self.hand_ind = -1
        self.pile_ind = -1
        self.last_card_played = 0

        # NN Values for feedback
        self.GoodMove = 1
        self.WrongMove = -1
        self.SkipMove = 9  # This move + 8 cards in hand

    def calculate_chance_10(self, cards, round_chance=True):
        """

        Args:
            cards:
            round_chance:

        Returns:

        """
        #
        # Check propabality of playing Card Higher or lower by 10
        #        
        lower_card_chance = []
        higher_card_chance = []

        # if len(cards) != 8:
        #     print("Cards Len =" + str(len(cards)))

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

    def cards_left_in_array(self):
        cards = np.zeros(98, dtype=int)
        for x in self.deck:
            cards[x - 2] = 1  # Card '2' is first on list, index 0
        return np.array(cards)

    def check_move(self, hand_id, pile_id):
        """
        Method Checks if move is proper
        Returns True if valid
        Returns False if invalid
        Used in checking for End Conditions
        Copied from Play Card Method
        """
        if hand_id < 0 or hand_id > 7:
            # print('Error: Invalid hand index')
            return False

        elif pile_id < 0 or pile_id > 3:
            # print('Error: Invalid pile index')
            return False

        elif pile_id == 0 or pile_id == 1:
            try:
                if self.hand[hand_id] > self.piles[pile_id] or \
                        self.hand[hand_id] == (self.piles[pile_id] - 10):
                    return True
                else:
                    return False
            except IndexError:
                return False

        elif pile_id == 2 or pile_id == 3:
            try:
                if self.hand[hand_id] < self.piles[pile_id] or \
                        self.hand[hand_id] == (self.piles[pile_id] + 10):
                    return True
                else:
                    return False
            except IndexError:
                return False

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
        [lower_chance, higher_chance] = self.calculate_chance_10(self.hand)

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
        Checking for game over.
        Checking if any move is valid.
        Checking if any cards left.
        """
        if self.score < -20 or self.turn > 150:
            end_game = False
            comment = 'You lost game!'
            return end_game, comment

        next_move = None
        for hand_id in range(8):
            if next_move:
                break

            for pile_id in range(4):
                next_move = self.check_move(hand_id, pile_id)
                if next_move:
                    break

        if next_move:
            end_game = None
            comment = 'Next Move available.'

        elif len(self.hand) == 0 and len(self.deck) == 0:
            end_game = True
            comment = 'You win!'
        else:
            end_game = False
            comment = 'Game over! No moves available'

        return end_game, comment

    def get_user_input(self):
        """
        Reading numbers from input
        Method Return:
          True:   Move
          None:   Command
          False:  Stop or Interrupts
          Second object is score feedback
        """
        self.hand_ind, self.pile_ind = -1, -1
        print('Select card and pile:')
        game_input = input()
        nums = re.findall(r'\d', game_input)

        if len(nums) == 2:
            self.hand_ind = int(nums[0]) - 1
            self.pile_ind = int(nums[1]) - 1
            return True
        else:
            game_input = game_input.split()
            for word in game_input:
                word = word.lower()

                if 'res' in word or 'new' in word:
                    self.reset()
                    return None

                elif 'end' in word or 'over' in word or 'quit' in word \
                        or 'exit' in word:
                    return False

    def hand_fill(self):
        """
        Fill Hand with cards from deck
        Hand is always 8
        """
        while len(self.hand) < 7 and len(self.deck) > 0:
            self.hand.append(self.deck[0])
            self.deck.pop(0)
        self.hand.sort()

    @staticmethod
    def input_random():
        """
        Random input generators (for testing purposes)
        """
        a = round(random.random() * 7) + 1
        b = round(random.random() * 3) + 1
        return a, b

    def main_loop(self):
        while True:
            self.hand_fill()
            status, comment = self.end_condition()

            if status is not None:
                print('\n' * 5)
                return status

            self.display_table()

            user_input = self.get_user_input()  # Replace user input with NN

            if user_input:
                self.play_card(self.hand_ind, self.pile_ind)
                # score = self.play_card(self.hand_ind, self.pile_ind)
                # self.score += score

            elif user_input is False:
                return False  # Interupted by user            
            else:
                pass

    def play_card(self, hand_id, pile_id):
        """
        Returns List Bool
        Plays Card from hand to pile.
        Checks for Valid move.
        Invalid moves return None.
        Add Turn Counter at proper moves.
        """
        self.move_count += 1
        self.score_gained = 0  # reset value
        try:
            if hand_id < 0 or hand_id > 7:  # Invalid Hand index
                print('Error: Invalid hand index')
                self.score += self.WrongMove
                self.score_gained = self.WrongMove
                return False

            elif pile_id < 0 or pile_id > 3:  # Invalid Pile index
                print('Error: Invalid pile index')
                self.score += self.WrongMove
                self.score_gained = self.WrongMove
                return False

            elif pile_id == 0 or pile_id == 1:  # Rising Piles
                if self.hand[hand_id] > self.piles[pile_id]:  # Played card is higher
                    self.piles[pile_id] = self.hand[hand_id]
                    self.hand.pop(hand_id)
                    self.score += self.GoodMove
                    self.score_gained = self.GoodMove
                    self.turn += 1
                    return True

                elif self.hand[hand_id] == (self.piles[pile_id] - 10):  # Played card is lower by 10
                    self.piles[pile_id] = self.hand[hand_id]
                    self.hand.pop(hand_id)
                    self.score += self.SkipMove
                    self.score_gained = self.SkipMove
                    self.turn += 1
                    return True
                else:  # Invalid Move
                    # print('Not valid move!')
                    self.score += self.WrongMove
                    self.score_gained = self.WrongMove
                    return False

            elif pile_id == 2 or pile_id == 3:  # Lowering Piles
                if self.hand[hand_id] < self.piles[pile_id]:  # Played card is lower
                    self.piles[pile_id] = self.hand[hand_id]
                    self.hand.pop(hand_id)
                    self.score += self.GoodMove
                    self.score_gained = self.GoodMove
                    self.turn += 1
                    return True

                elif self.hand[hand_id] == (self.piles[pile_id] + 10):  # Played card is higher by 10
                    self.piles[pile_id] = self.hand[hand_id]
                    self.hand.pop(hand_id)
                    self.score += self.SkipMove
                    self.score_gained = self.SkipMove
                    return True

                else:  # Invalid Move
                    # print('Not valid move!')
                    self.score += self.WrongMove
                    self.score_gained = self.WrongMove
                    return False
            else:
                input('Impossible! How did u get here?!')

        except IndexError:
            print('Not valid move!')
            self.score += self.WrongMove
            self.score_gained = self.WrongMove
            return False

    def reset(self):
        """
        Reset game
        Returns:

        """
        self.__init__()

    def start_game(self, load_save=False):
        """
        Start New Game or Load Save
        """
        self.reset()

        if load_save:
            file = open('data/98CardsGame_SaveFile.json', 'r')
            self.deck = json.load(file)
            file.close()

        comment = self.main_loop()
        print(comment)


if __name__ == '__main__':
    print('Start')
    app = GameCards98()
    app.start_game(load_save=False)
