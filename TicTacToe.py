import os


class TicTacToe:

    def __init__(self, size):

        self.size = size
        self.tile_count = size * size

        self.reset()

    def __str__(self):

        out = ""

        row_break = "\n" + "- " * (self.size) + "\n"

        col_count = 0
        row_count = 0
        for tile in self.board:

            if tile == 1:
                out += "X"
            elif tile == -1:
                out += "O"
            else:
                out += " "

            if col_count < self.size - 1:
                out += "|"

            col_count += 1
            if col_count == self.size and row_count < self.size - 1:
                out += row_break
                col_count = 0
                row_count += 1

        return out

    def reset(self):

        self.board = [0] * self.size * self.size
        self.turn = 1
        self.moves_made = 0

    def make_move(self, tile):

        legal_move = False
        if tile >= 0 and tile < len(self.board):

            if self.board[tile] == 0:
                self.board[tile] = self.turn
                self.moves_made += 1
                self.turn *= -1
                legal_move = True

        return legal_move

    def check_for_winner(self):

        winner = 0

        if winner == 0:
            # check each row
            for cell in range(0, len(self.board), self.size):
                count = sum(self.board[cell:cell + self.size])
                if abs(count) == self.size:
                    winner = 1 if count > 0 else -1
                    break

        if winner == 0:
            # check each col
            for cell in range(self.size):
                count = sum(self.board[cell::self.size])
                if abs(count) == self.size:
                    winner = 1 if count > 0 else -1
                    break

        # check left to right digonal
        if winner == 0:
            count = sum(self.board[::self.size + 1])
            if abs(count) == self.size:
                winner = 1 if count > 0 else -1

        # check right to left diagonal
        if winner == 0:
            count = sum(self.board[len(self.board) - self.size:0:-(self.size - 1)])
            if abs(count) == self.size:
                winner = 1 if count > 0 else -1

        # check for draw
        if winner == 0:
            if self.moves_made == len(self.board):
                winner = 2

        return winner


if __name__ == "__main__":

    b = TicTacToe(size=3)

    while b.check_for_winner() == 0:
        os.system('cls')
        print(b)
        move = int(input("Make move "))
        b.make_move(move)

    os.system('cls')
    print(b)
    if b.check_for_winner() == 2:
        print("The game was a draw")
    else:
        print("Player {} is the winner".format(b.check_for_winner()))
