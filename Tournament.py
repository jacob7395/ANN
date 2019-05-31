import copy
import os
import time
from multiprocessing import Process, cpu_count, Queue

import numpy as np

from PythonNetwork import NeuralNetwork
from TicTacToe import TicTacToe

import matplotlib.pyplot as plt

MUTRATION_RATE = 1
MOVE_BONUS = 15
GEN_SIZE = 2 ** 15


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

    def __init__(self, networks, game, game_queue, complete_game_queue, worker_load = 5):

        self.players = []

        for network in networks:
            self.players.append(player(network))

        self.games_left = len(self.players) - 1
        self.waiting_players = self.players
        self.game = game
        self.match_list = []
        self.worker_load = worker_load
        self.game_queue = game_queue
        self.complete_games = []
        self.complete_game_queue = complete_game_queue
        self.tournament_run_time = 0

    def play(self):

        start_time = time.time()
        while self.games_left != 0:
            self.match_maker()

            self.fill_queue()

            finished_games = []
            try:
                while True:
                    finished_games += self.complete_game_queue.get(False)
            except:
                pass
 
            self.games_left -= len(finished_games)

            for game in finished_games:
                self.waiting_players.append(game.winner)
                self.complete_games.append(game)
        self.tournament_run_time = time.time() - start_time
                

    def match_maker(self):

        match_failed = []
        while len(self.waiting_players) > 0:
            new_match = None
            player1 = self.waiting_players.pop(0)
            for player2 in self.waiting_players:
                if player2.round == player1.round:
                    self.waiting_players.remove(player2)
                    new_match = match(players=[player1, player2], game=game)
                    self.match_list.append(new_match)
                    break
            
            if new_match == None:
                match_failed.append(player1)
        
        self.waiting_players = match_failed

    def fill_queue(self):

        if len(self.match_list) < self.worker_load:
            load = max(self.game_queue.qsize(), 1) if self.game_queue.qsize() < self.worker_load else self.worker_load
        else:
            load = self.worker_load

        while len(self.match_list) >= load:
            match_slice = self.match_list[:load]
            self.game_queue.put(match_slice)
            del self.match_list[:load]

    def stats(self):

        pass


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

    # running load test
    timing_data = []
    for gen_count in range(50):
        np.random.shuffle(players)

        torney = tournament(networks = players, game = game, game_queue = game_queue, complete_game_queue = complete_game_queue, worker_load = 13)
        torney.play()

        timing_data.append(torney.tournament_run_time)

        print("Generation {}".format(gen_count))

    plt.boxplot(timing_data)
    plt.show()

    for worker in workerforce:
        worker.terminate()
