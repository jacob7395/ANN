import copy
import os
import time
from multiprocessing import Process, cpu_count, Queue

import numpy as np

from PythonNetwork import NeuralNetwork
from TicTacToe import TicTacToe

MUTRATION_RATE = 1
MOVE_BONUS = 15
GEN_SIZE = 2 ** 10


class player:

    def __init__(self, network):
        self.network = network
        self.round = 0


class match:

    def __init__(self, players, game):

        self.players = players
        self.turn = 1
        self.winner = None
        self.loser = None
        self.valid_move = True
        self.result = 0
        self.moves_made = 0
        self.game = copy.deepcopy(game)

    def play(self):
        # run the game till a winner or an invalid move is made
        while self.result == 0:
            # construct the game info
            game_info = [self.game.turn] + self.game.board
            # calculate the player move
            move = list(self.players[self.turn].network.run(game_info).T[0])
            move = move.index(max(move))
            # make the move
            self.valid_move = self.game.make_move(move)
            # break if the move was invalid
            if not self.valid_move:
                break
            # switch the play turn
            self.turn = int(not self.turn)
            # check if there was a winner
            self.result = self.game.check_for_winner()

        for player in self.players:
            player.round += 1

        self.moves_made = self.game.moves_made

        self.loser = self.players.pop(self.turn)
        self.winner = self.players.pop(0)


class tournament:

    def __init__(self, networks, game, game_queue, complete_game_queue):

        self.players = []

        for network in networks:
            self.players.append(player(network))

        self.games_left = len(self.players) - 1
        self.waiting_players = self.players
        self.game = game
        self.match_list = []
        self.worker_load = 5
        self.game_queue = game_queue
        self.complete_games = []
        self.complete_game_queue = complete_game_queue

    def play(self):

        while self.games_left != 0:
            self.setup_games()

            self.fill_queue()

            finished_games = []
            try:
                finished_games = self.complete_game_queue.get(False)
            except:
                pass

            for game in finished_games:
                self.waiting_players.append(game.winner)
                self.complete_games.append(game)
                self.games_left -= 1

    def setup_games(self):

        while len(self.waiting_players) > 1:
            player1 = self.waiting_players.pop(0)
            player2 = self.waiting_players.pop(0)
            self.match_list.append(match(players=[player1, player2], game=game))

    def fill_queue(self):

        while len(self.match_list) >= self.worker_load or (
                self.games_left <= self.worker_load and self.match_list != []):
            match_slice = self.match_list[:self.worker_load]
            self.game_queue.put(match_slice)
            del self.match_list[:self.worker_load]


def work_game(game_queue, complete_game_queue):
    while True:
        game_list = game_queue.get()
        for game in game_list:
            game.play()
        complete_game_queue.put(game_list)


if __name__ == "__main__":
    game = TicTacToe(3)

    game_queue = Queue()
    complete_game_queue = Queue()
    workerforce = []
    for _ in range(cpu_count()):
        worker = Process(target=work_game, args=(
            game_queue, complete_game_queue))
        worker.start()
        workerforce.append(worker)

    input_count = game.tile_count + 1
    output_count = game.tile_count

    base_network = NeuralNetwork([input_count,
                                  int(input_count * 2.5),
                                  int(input_count * 2.5),
                                  output_count],
                                 learning_rate=MUTRATION_RATE,
                                 bias=False)

    player_count = GEN_SIZE

    spawn_ratio = [0.55, 0.20, 0.1, 0.1, 0.05]
    spawn_amounts = []
    running_payer_count = player_count

    for ratio in spawn_ratio[:-1]:
        spawn_count = int(player_count * ratio)

        if running_payer_count - spawn_count <= 0:
            spawn_count = running_payer_count

        spawn_amounts.append(spawn_count)
        running_payer_count -= spawn_count

    spawn_amounts.append(running_payer_count)

    players = base_network.spawn(player_count)

    for gen_count in range(50000):
        np.random.shuffle(players)

        torney = tournament(players, game, game_queue, complete_game_queue)
        torney.play()

        os.system('clear')
        print(torney.complete_games[-1].game)
        print("Generation {}".format(gen_count))
        time.sleep(0.1)
