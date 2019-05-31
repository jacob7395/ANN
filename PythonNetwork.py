# https://www.python-course.eu/neural_networks_with_python_numpy.php

import copy
import os
import random
import time

import numpy as np
from scipy.special import expit as activation_function
from scipy.stats import truncnorm

from TicTacToe import TicTacToe


def sigmoid(x):
    return 1 / (1 + np.e ** -x)


activation_function = sigmoid


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class NeuralNetwork:
    class _Layer:

        def __init__(self,
                     no_of_nodes,
                     no_of_in_nodes,
                     bias_node=False
                     ):
            self.no_of_nodes = no_of_nodes
            self.bias_node = bias_node
            self.no_of_in_nodes = no_of_in_nodes

            self.init_weight_matrices()

        def init_weight_matrices(self):
            rad = 1 / np.sqrt(self.no_of_in_nodes)
            X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
            self.weights = X.rvs((self.no_of_nodes,
                                  self.no_of_in_nodes))

        def run(self, input_vector):

            output_vector = np.dot(self.weights, input_vector)
            output_vector_activation = activation_function(output_vector)

            if self.bias_node:
                output_vector_activation = np.concatenate(
                    (output_vector_activation, [[1]]))

            return output_vector_activation

        def get_node_count(self):

            return self.no_of_nodes + 1 if self.bias_node else self.no_of_nodes

        def correct_weights(self, correction):

            self.weights += correction

        def shuffle_weights(self, value):

            for weight in self.weights:
                self.weights += random.uniform(-value, value)

    def __init__(self,
                 layout,
                 learning_rate,
                 bias=None,
                 ):

        self.layout = layout

        self.layers = []

        self.learning_rate = learning_rate
        self.bias = bias
        self.fitness = 0
        self.setup_layers()

    def setup_layers(self):
        """ A method to initialize the weight matrices of the neural 
        network with optional bias nodes"""

        bias_node = 1 if self.bias else 0

        previous_layer_nodes = self.layout[0] + bias_node
        for nodes in self.layout[1:-1]:
            self.layers.append(self._Layer(nodes,
                                           previous_layer_nodes,
                                           bias_node))
            previous_layer_nodes = nodes + bias_node

        self.layers.append(self._Layer(self.layout[-1],
                                       self.layers[-1].get_node_count()))

    def spawn(self, amount):

        copy_count = int(amount * 0.25)
        children = [copy.deepcopy(self)] * copy_count
        amount -= copy_count

        for _ in range(amount):

            child = copy.deepcopy(self)

            for layer in child.layers:
                layer.shuffle_weights(self.learning_rate)

            children.append(child)

        return children

    def run(self, input_vector):

        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [self.bias]))

        input_vector = np.array(input_vector, ndmin=2).T

        layer_input_vector = input_vector
        layer_outputs = [input_vector]
        for layer in self.layers:
            layer_outputs.append(layer.run(layer_input_vector))
            layer_input_vector = layer_outputs[-1]

        return layer_outputs[-1]


if __name__ == "__main__":
    game = TicTacToe(3)

    input_count = game.tile_count + 1
    output_count = game.tile_count

    base_network = NeuralNetwork([input_count, int(input_count * 1.5), input_count, input_count, output_count],
                                 learning_rate=2,
                                 bias=False)

    player_count = 2 ** 10

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

        ranking = []
        fitness = []

        while len(players) > 1:
            winners = []
            matches = []

            np.random.shuffle(players)

            players = iter(players)
            for player1, player2 in zip(players, players):
                matches.append([player1, player2])

            for match in matches:

                game.reset()
                match_platers = match
                turn = 1
                turn_count = 0
                winner = 0
                while winner == 0:
                    turn = not turn

                    game_info = [game.turn] + game.board

                    move = list(match_platers[turn].run(game_info).T[0])
                    move = move.index(max(move))

                    if (not game.make_move(move)):
                        turn = not turn
                        break
                    else:
                        match_platers[turn].fitness += 1 * turn_count

                    turn_count += 10
                    winner = game.check_for_winner()

                winners.append(match_platers[turn])
                ranking.append(match_platers[not turn])
                fitness.append(match_platers[turn].fitness)

            players = winners

        ranking.append(players[-1])
        fitness.append(players[-1].fitness)

        players = []
        for spawn in spawn_amounts:
            fittest = fitness.index(max(fitness))

            fitness.pop(fittest)
            parant = ranking.pop(fittest)
            players += parant.spawn(spawn)

        os.system('cls')
        print(game)
        print("Generation {}".format(gen_count))
        time.sleep(0.1)
